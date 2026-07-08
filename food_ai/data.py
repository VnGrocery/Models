import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


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


def create_loaders(train_root, valid_root, batch_size, workers):
    train_transform, eval_transform = build_transforms()
    train_dataset = ImageFolder(train_root, transform=train_transform)
    valid_dataset = ImageFolder(valid_root, transform=eval_transform)

    if train_dataset.classes != valid_dataset.classes:
        raise ValueError(
            "Nhãn train/valid không khớp: "
            f"{train_dataset.classes} != {valid_dataset.classes}"
        )

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
    return train_loader, valid_loader, train_dataset.classes

