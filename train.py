import argparse
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from food_ai.config import CATEGORIES, DATA_DIR, MODEL_DIR, freshness_model
from food_ai.data import create_loaders
from food_ai.model import create_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = DEVICE.type == "cuda"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện AI phân tầng")
    subparsers = parser.add_subparsers(dest="task", required=True)

    subparsers.add_parser("router", help="Huấn luyện nhận diện 5 nhóm")
    expert = subparsers.add_parser(
        "freshness", help="Huấn luyện expert đánh giá độ tươi"
    )
    expert.add_argument("--category", required=True, choices=CATEGORIES)

    parser.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", 15)))
    parser.add_argument(
        "--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", 32))
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("NUM_WORKERS", min(4, os.cpu_count() or 1))),
    )
    return parser.parse_args()


def resolve_task(args):
    if args.task == "router":
        return DATA_DIR / "train", DATA_DIR / "valid", MODEL_DIR / "router.pth"
    return (
        DATA_DIR / "train" / args.category,
        DATA_DIR / "valid" / args.category,
        freshness_model(args.category),
    )


def move_batch(images, labels):
    memory_format = torch.channels_last if USE_CUDA else torch.preserve_format
    images = images.to(
        DEVICE, non_blocking=USE_CUDA, memory_format=memory_format
    )
    labels = labels.to(DEVICE, non_blocking=USE_CUDA)
    return images, labels


def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = count = 0
    context = nullcontext() if training else torch.inference_mode()

    with context:
        for images, labels in loader:
            images, labels = move_batch(images, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)

            amp = (
                torch.autocast("cuda", dtype=torch.float16)
                if USE_CUDA
                else nullcontext()
            )
            with amp:
                outputs = model(images)
                loss = criterion(outputs, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            size = labels.size(0)
            loss_sum += loss.detach().item() * size
            correct += outputs.argmax(1).eq(labels).sum().item()
            count += size

    return loss_sum / count, 100 * correct / count


def main():
    args = parse_args()
    train_root, valid_root, output_path = resolve_task(args)

    try:
        train_loader, valid_loader, classes = create_loaders(
            train_root, valid_root, args.batch_size, args.workers
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Dữ liệu chưa sẵn sàng: {error}") from error

    if len(train_loader.dataset) == 0 or len(valid_loader.dataset) == 0:
        raise SystemExit("Dữ liệu train/valid đang rỗng.")

    print(f"Task: {args.task} | lớp: {classes}")
    print(
        f"Train: {len(train_loader.dataset)} | Valid: {len(valid_loader.dataset)}"
    )
    print(f"Thiết bị: {torch.cuda.get_device_name(0) if USE_CUDA else 'CPU'}")

    if USE_CUDA:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    model = create_model(len(classes), pretrained=True).to(DEVICE)
    if USE_CUDA:
        model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer_options = {"lr": 5e-4, "weight_decay": 1e-4}
    if USE_CUDA:
        optimizer_options["fused"] = True
    try:
        optimizer = optim.AdamW(model.parameters(), **optimizer_options)
    except (TypeError, RuntimeError):
        optimizer_options.pop("fused", None)
        optimizer = optim.AdamW(model.parameters(), **optimizer_options)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_CUDA)
    best_loss = float("inf")
    stale_epochs = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler
        )
        valid_loss, valid_acc = run_epoch(model, valid_loader, criterion)
        scheduler.step(valid_loss)

        improved = valid_loss < best_loss
        if improved:
            best_loss = valid_loss
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": classes,
                    "task": args.task,
                    "category": getattr(args, "category", None),
                },
                output_path,
            )
        else:
            stale_epochs += 1

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train {train_acc:.2f}%/{train_loss:.4f} | "
            f"valid {valid_acc:.2f}%/{valid_loss:.4f} | "
            f"{time.perf_counter() - started:.1f}s"
            + (" | saved" if improved else ""),
            flush=True,
        )
        if stale_epochs >= 3:
            print("Early stopping: 3 epoch không cải thiện.")
            break


if __name__ == "__main__":
    main()
