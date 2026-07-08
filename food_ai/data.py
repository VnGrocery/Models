import os
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS

warnings.filterwarnings(
    "ignore",
    message="Palette images with Transparency expressed in bytes should be converted to RGBA images",
    category=UserWarning,
    module=r"PIL\.Image",
)


class NonEmptyImageFolder(ImageFolder):
    """ImageFolder variant that ignores placeholder classes with no images."""

    @staticmethod
    def find_classes(directory):
        root = Path(directory)
        classes = sorted(
            entry.name
            for entry in root.iterdir()
            if entry.is_dir()
            and any(
                item.is_file() and item.suffix.lower() in IMG_EXTENSIONS
                for item in entry.rglob("*")
            )
        )
        if not classes:
            raise FileNotFoundError(f"Không có lớp nào chứa ảnh trong {directory}")
        return classes, {name: index for index, name in enumerate(classes)}


def build_transforms():
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, eval_transform


def create_loaders(
    train_root,
    valid_root,
    batch_size,
    workers,
    test_ratio=0.20,
    seed=42,
):
    train_transform, eval_transform = build_transforms()
    full_train_dataset = NonEmptyImageFolder(train_root, transform=train_transform)
    full_test_dataset = NonEmptyImageFolder(train_root, transform=eval_transform)
    valid_dataset = NonEmptyImageFolder(valid_root, transform=eval_transform)

    if full_train_dataset.classes != valid_dataset.classes:
        raise ValueError(
            "Nhãn train/valid không khớp: "
            f"{full_train_dataset.classes} != {valid_dataset.classes}"
        )

    total = len(full_train_dataset)
    if total < 2:
        raise ValueError("Cần ít nhất hai ảnh để tách train/test")
    test_count = max(1, round(total * test_ratio))
    test_count = min(test_count, total - 1)
    indices = torch.randperm(
        total, generator=torch.Generator().manual_seed(seed)
    ).tolist()
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    train_dataset = Subset(full_train_dataset, train_indices)
    test_dataset = Subset(full_test_dataset, test_indices)

    use_cuda = torch.cuda.is_available()
    options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": use_cuda,
    }
    if workers > 0:
        options.update(persistent_workers=True, prefetch_factor=2)

    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=use_cuda, **options
    )
    valid_loader = DataLoader(valid_dataset, shuffle=False, **options)
    test_loader = DataLoader(test_dataset, shuffle=False, **options)
    return train_loader, valid_loader, test_loader, full_train_dataset.classes
