import argparse
from pathlib import Path
import torch
import onnxruntime as ort
import numpy as np
from ml.model import build_model

NUM_CLASSES = 10
IMAGE_SIZE = 64

def export_onnx(model:torch.nn.Module, path:str | Path, image_size:int=IMAGE_SIZE)-> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model=model.cpu().eval()
    dummy_input=torch.randn(1, 3, image_size, image_size)


    torch.onnx.export(
        model,
        dummy_input,
        str(path),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False
    )
    return path

def verify_onnx(model:torch.nn.Module, onnx_path:str | Path, image_size:int=IMAGE_SIZE)->float:
    model=model.cpu().eval()
    x=torch.randn(4, 3, image_size, image_size)

    with torch.no_grad():
        expected=model(x).numpy()

    session=ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual=session.run(["logits"], {"images": x.numpy()})[0]

    np.testing.assert_allclose(actual,expected, rtol=1e-3, atol=1e-4)
    return float(np.abs(actual-expected).max())


def main()->None:
    parser=argparse.ArgumentParser(description="Export a trained PyTorch model to ONNX")
    parser.add_argument("--checkpoint", default="models/best_model.pth")
    parser.add_argument("--output", default="models/model.onnx")
    args = parser.parse_args()

    model=build_model(num_classes=NUM_CLASSES, pretrained=False)
    state=torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    path=export_onnx(model, args.output)
    max_diff=verify_onnx(model, path)
    size_mb=path.stat().st_size/1e6
    print(f"Exported {path} ({size_mb:.1f} MB). Max difference vs PyTorch: {max_diff:.2e}")


if __name__=="__main__":
    main()