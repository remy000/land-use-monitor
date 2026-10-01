import argparse
import json
from pathlib import Path
import torch
from torch import nn
from ml.data import build_dataloaders
from ml.model import build_model
import subprocess
import mlflow
from ml.export import export_onnx, verify_onnx

NUM_CLASSES = 10


def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description="Fine tume Resnet-18 on EuroSAT.")
    parser.add_argument("--data-root", default="data/EuroSAT")
    parser.add_argument("--splits-path", default="configs/eurosat_splits.json")
    parser.add_argument("--stats-path", default="configs/eurosat_stats.json")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)   
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-batches", type=int, default=None, help="Stop each epoch after this many batches (for quick local tests)"
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def train_one_epoch(
        model, 
        loader,
        criterion,
        optimizer,
        device,
        max_batches=None
):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss+=loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return {"loss": total_loss / total, "accuracy": correct / total}

def evaluate(model, loader, criterion, device, max_batches=None):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return {"loss": total_loss / total, "accuracy": correct / total}


def git_commit()->str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main()->None:
    args=parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loaders = build_dataloaders(
        root=args.data_root,
        splits_path=args.splits_path,
        stats_path=args.stats_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    model = build_model(num_classes=NUM_CLASSES, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)   
    best_val_accuracy = 0.0

    history=[]

    mlflow.set_experiment("eurosat_training")
    with mlflow.start_run():
        mlflow.log_params(vars(args))
        mlflow.set_tag("git_commit", git_commit())
        mlflow.set_tag("device", str(device))

        for epoch in range(1, args.epochs+1):
            lr=optimizer.param_groups[0]["lr"]

            train_metrics = train_one_epoch(
                model, loaders["train"], criterion, optimizer, device, max_batches=args.max_batches
            )
            val_metrics = evaluate(
                model, loaders["val"], criterion, device, max_batches=args.max_batches
            )
            scheduler.step()

            history.append({"epoch": epoch, "train_metrics": train_metrics, "val_metrics": val_metrics})

            mlflow.log_metrics(
                {
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "learning_rate": lr,
                },
                step=epoch,
            )
            print(
                    f"Epoch {epoch}/{args.epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f} - "
                    f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}"
                )

            if val_metrics["accuracy"] > best_val_accuracy:
                best_val_accuracy = val_metrics["accuracy"]
                torch.save(model.state_dict(), output_dir / "best_model.pth")
                print(f" Saved  new best model with val accuracy: {best_val_accuracy:.4f}")


        mlflow.log_metrics({"best_val_accuracy": best_val_accuracy})
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))
        

        class_names = loaders["train"].dataset.base.classes
        (output_dir / "class_names.json").write_text(json.dumps(class_names, indent=2))

        best_state = torch.load(output_dir / "best_model.pth", map_location=device, weights_only=True)
        model.load_state_dict(best_state)
        onnx_path = export_onnx(model, output_dir / "model.onnx")
        max_diff = verify_onnx(model, onnx_path)
        print(f"ONNX export verified with max difference {max_diff:.2e}")


        for name in ("best_model.pth", "model.onnx", "history.json", "class_names.json"):
            mlflow.log_artifact(str(output_dir / name))
        mlflow.log_artifact(args.stats_path)

    print(f"Done. Best val accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()
