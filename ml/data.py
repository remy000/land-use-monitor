import json
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import EuroSAT
from torchvision.transforms.functional import to_tensor

from ml.splits import load_splits
from ml.transform import eval_transforms, train_transforms


def load_eurosat(root: str | Path) -> EuroSAT:
    """Load EuroSAT RGB from `root`, downloading it the first time."""
    return EuroSAT(root=str(root), download=True)

def compute_channel_stats(dataset, indices: list[int]) -> dict[str, list[float]]:
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sq_sum = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0

    for i in indices:
        image, _ = dataset[i]
        x = to_tensor(image).double()
        channel_sum += x.sum(dim=(1, 2))
        channel_sq_sum += (x**2).sum(dim=(1, 2))
        pixel_count += x.shape[1] * x.shape[2]

    mean = channel_sum / pixel_count
    std = (channel_sq_sum / pixel_count - mean**2).sqrt()
    return {"mean": mean.tolist(), "std": std.tolist()}


def save_stats(stats: dict[str, list[float]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2))


def load_stats(path: str | Path) -> dict[str, list[float]]:
    return json.loads(Path(path).read_text())


class SplitDataset(Dataset):
    """A view of `base` containing only `indices`, with its own transform."""

    def __init__(self, base: Dataset, indices: list[int], transform: Callable):
        self.base = base
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        image, label = self.base[self.indices[i]]
        return self.transform(image), label


def build_dataloaders(
    root: str | Path,
    splits_path: str | Path,
    stats_path: str | Path,
    batch_size: int = 64,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    base = load_eurosat(root)
    splits = load_splits(splits_path)
    stats = load_stats(stats_path)
    mean, std = stats["mean"], stats["std"]

    datasets = {
        "train": SplitDataset(base, splits["train"], train_transforms(mean, std)),
        "val": SplitDataset(base, splits["val"], eval_transforms(mean, std)),
        "test": SplitDataset(base, splits["test"], eval_transforms(mean, std)),
    }

    return {
        name: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for name, ds in datasets.items()
    }