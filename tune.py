import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from food_ai.config import CATEGORIES

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Grid search tham số huấn luyện")
    parser.add_argument("--task", choices=("router", "freshness"), required=True)
    parser.add_argument("--category", choices=CATEGORIES)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def newest_run(previous):
    candidates = set(Path("runs").glob("*/summary.json")) - previous
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main():
    args = parse_args()
    if args.task == "freshness" and not args.category:
        raise SystemExit("Task freshness yêu cầu --category")

    trials = [
        ("mobilenet_v3_small", "head_only", "adamw", 1e-3),
        ("mobilenet_v3_small", "fine_tune", "adamw", 5e-4),
        ("mobilenet_v3_large", "fine_tune", "adamw", 5e-4),
        ("efficientnet_b0", "fine_tune", "adamw", 3e-4),
    ]
    results = []
    for index, (model, method, optimizer, learning_rate) in enumerate(trials, start=1):
        print(f"\n=== Trial {index}/{len(trials)}: {model}, {method}, lr={learning_rate} ===")
        before = set(Path("runs").glob("*/summary.json"))
        command = [
            sys.executable,
            "train.py",
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--workers", str(args.workers),
            "--model", model,
            "--method", method,
            "--optimizer", optimizer,
            "--learning-rate", str(learning_rate),
            "--run-name", f"tune_{index}_{model}_{method}",
            args.task,
        ]
        if args.category:
            command.extend(["--category", args.category])
        completed = subprocess.run(command)
        if completed.returncode != 0:
            print(f"Trial {index} failed; continuing.")
            continue
        summary_path = newest_run(before)
        if not summary_path:
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        results.append({
            "model": model,
            "method": method,
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "best_valid_loss": summary["best_valid_loss"],
            "best_valid_accuracy": summary["best_valid_accuracy"],
            "test_accuracy": summary["test_accuracy"],
            "macro_f1": summary["macro_f1"],
            "run": summary_path.parent.name,
        })

    if not results:
        raise SystemExit("Không có trial nào hoàn thành.")
    results.sort(key=lambda row: row["best_valid_loss"])
    output = Path("runs") / f"tuning_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n=== Leaderboard ===")
    for row in results:
        print(
            f"{row['model']:<22} {row['method']:<10} "
            f"ValLoss={row['best_valid_loss']:.4f} "
            f"ValAcc={row['best_valid_accuracy']:.2f}% | "
            f"TestF1={row['macro_f1']:.4f}"
        )
    print(f"Đã lưu: {output}")


if __name__ == "__main__":
    main()
