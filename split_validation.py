import argparse
import random
import shutil
import sys
from pathlib import Path

from torchvision.datasets.folder import IMG_EXTENSIONS

from food_ai.config import CATEGORIES, DATA_DIR, FRESHNESS_CLASSES

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def image_files(path: Path):
    return [
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMG_EXTENSIONS
    ]


def split_product(source, destination, ratio, seed):
    files = image_files(source)
    if not files:
        return 0

    destination.mkdir(parents=True, exist_ok=True)
    existing_files = image_files(destination)
    existing_names = {item.name for item in existing_files}
    collisions = existing_names.intersection(item.name for item in files)
    if collisions:
        raise FileExistsError(
            f"Có {len(collisions)} tên file trùng tại {destination}"
        )

    total_count = len(files) + len(existing_files)
    target_count = max(1, round(total_count * ratio))
    move_count = max(0, target_count - len(existing_files))
    if move_count == 0:
        return 0

    rng = random.Random(seed)
    rng.shuffle(files)
    for source_file in files[:move_count]:
        shutil.move(source_file, destination / source_file.name)
    return move_count


def main():
    parser = argparse.ArgumentParser(description="Tách dữ liệu validation")
    parser.add_argument("--ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, required=True)
    args = parser.parse_args()

    if not 0 < args.ratio < 0.5:
        raise SystemExit("--ratio phải lớn hơn 0 và nhỏ hơn 0.5")

    total = 0
    for category in args.categories:
        for freshness in FRESHNESS_CLASSES:
            source_class = DATA_DIR / "train" / category / freshness
            if not source_class.exists():
                continue
            for product in sorted(path for path in source_class.iterdir() if path.is_dir()):
                destination = DATA_DIR / "valid" / category / freshness / product.name
                moved = split_product(
                    product,
                    destination,
                    args.ratio,
                    f"{args.seed}:{category}:{freshness}:{product.name}",
                )
                if moved:
                    print(f"{category}/{freshness}/{product.name}: {moved} ảnh")
                    total += moved

    print(f"Đã chuyển tổng cộng {total} ảnh sang validation.")


if __name__ == "__main__":
    main()
