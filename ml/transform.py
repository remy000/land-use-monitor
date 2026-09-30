import torch
from torchvision import transforms as T


class RandomRot90:
    """Rotate a tensor image by 0, 90, 180 or 270 degrees at random."""

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        k = int(torch.randint(0, 4, (1,)))
        return torch.rot90(img, k, dims=(1, 2))


def train_transforms(mean: list[float], std: list[float]) -> T.Compose:
    return T.Compose(
        [
            T.ToTensor(),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            RandomRot90(),
            T.Normalize(mean, std),
        ]
    )
def eval_transforms(mean: list[float], std: list[float]) -> T.Compose:
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )