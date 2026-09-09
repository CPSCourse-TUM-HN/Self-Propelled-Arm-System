import argparse
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge reviewed images + YOLO labels from new_data/ into the training dataset with 70/15/15 split."
    )
    parser.add_argument("--images", type=Path, default=PROJECT_ROOT / "new_data" / "images", help="Reviewed images dir.")
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "new_data" / "labels", help="Reviewed labels dir.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "paper_detect", help="Dataset root (has train/ valid/ test/).")
    parser.add_argument("--train-frac", type=float, default=0.7, help="Fraction (0.0-1.0) to train. Default: 0.7 (70%%).")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Fraction (0.0-1.0) to valid. Default: 0.15 (15%%).")
    parser.add_argument("--test-frac", type=float, default=0.15, help="Fraction (0.0-1.0) to test. Default: 0.15 (15%%).")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed for the split.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be copied without copying.")
    return parser.parse_args()


def main():
    args = parse_args()
    total = args.train_frac + args.val_frac + args.test_frac
    if not (0.99 <= total <= 1.01):
        raise SystemExit(f"Fractions must sum to 1.0 (got {total}).")
    if not args.images.is_dir():
        raise SystemExit(f"Images dir not found: {args.images}")

    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit(f"No images in {args.images}")

    random.seed(args.seed)
    random.shuffle(images)

    train_count = round(len(images) * args.train_frac)
    val_count = round(len(images) * args.val_frac)

    train_set = set(images[:train_count])
    val_set = set(images[train_count:train_count + val_count])
    test_set = set(images[train_count + val_count:])

    counts = {"train": 0, "valid": 0, "test": 0, "skipped": 0}
    for img in images:
        label = args.labels / f"{img.stem}.txt"
        if not label.exists():
            print(f"SKIP (no label): {img.name}")
            counts["skipped"] += 1
            continue

        if img in train_set:
            split = "train"
        elif img in val_set:
            split = "valid"
        else:
            split = "test"

        dst_img = args.dataset / split / "images" / img.name
        dst_lbl = args.dataset / split / "labels" / label.name
        if args.dry_run:
            print(f"{split}: {img.name} (+ {label.name})")
        else:
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dst_img)
            shutil.copy2(label, dst_lbl)
        counts[split] += 1

    print(f"\nDone. train += {counts['train']}, valid += {counts['valid']}, test += {counts['test']}, skipped = {counts['skipped']}")
    print(f"Split ratio: {counts['train']}/{len(images)-counts['skipped']} train, {counts['valid']}/{len(images)-counts['skipped']} valid, {counts['test']}/{len(images)-counts['skipped']} test")
    if not args.dry_run and (counts["train"] or counts["valid"] or counts["test"]):
        print("Now retrain on the expanded dataset:  python train_paper.py")


if __name__ == "__main__":
    main()
