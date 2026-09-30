import json
from pathlib import Path

from sklearn.model_selection import train_test_split


def make_splits(
    labels: list[int],
    seed: int = 42,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> dict[str, list[int]]:
    indices = list(range(len(labels)))

    # First cut: separate the test set from everything else.
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=labels, random_state=seed
    )

    # Second cut: split the remainder into train and validation.
    train_val_labels = [labels[i] for i in train_val_idx]
    relative_val_size = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val_size,
        stratify=train_val_labels,
        random_state=seed,
    )

    return {
        "train": sorted(train_idx),
        "val": sorted(val_idx),
        "test": sorted(test_idx),
    }


def save_splits(splits: dict[str, list[int]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits))


def load_splits(path: str | Path) -> dict[str, list[int]]:
    return json.loads(Path(path).read_text())