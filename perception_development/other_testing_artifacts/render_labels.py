import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# distinct BGR colors per class index (cycled)
PALETTE = [(0, 180, 0), (0, 0, 255), (255, 128, 0), (200, 0, 200), (0, 200, 200), (128, 0, 255)]


def load_class_names(data_yaml):
    """Return class names from a dataset YAML, or None if unavailable."""
    data_yaml = Path(data_yaml)
    if not data_yaml.is_file():
        return None
    try:
        import yaml
    except ImportError:
        return None
    with open(data_yaml, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    names = data.get("names")
    if isinstance(names, dict):  # {0: '10', 1: 'paper'}
        names = [names[key] for key in sorted(names)]
    return [str(name) for name in names] if names else None


def _draw_labels(img, label_path, names):
    h, w = img.shape[:2]
    if not label_path.exists():
        return 0
    drawn = 0
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:5])
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        color = PALETTE[cls % len(PALETTE)]
        name = names[cls] if names and 0 <= cls < len(names) else str(cls)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, name, (x1, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        drawn += 1
    return drawn


def render_previews(images_dir, labels_dir, preview_dir, data_yaml=None, verbose=True):
    """Redraw preview images from the CURRENT label files. Returns (images, boxes)."""
    images_dir, labels_dir, preview_dir = Path(images_dir), Path(labels_dir), Path(preview_dir)
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    preview_dir.mkdir(parents=True, exist_ok=True)
    names = load_class_names(data_yaml) if data_yaml else None

    total_boxes = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            if verbose:
                print(f"  SKIP (unreadable): {img_path.name}")
            continue
        boxes = _draw_labels(img, labels_dir / f"{img_path.stem}.txt", names)
        cv2.imwrite(str(preview_dir / img_path.name), img)
        total_boxes += boxes
        if verbose:
            print(f"  {img_path.name}: {boxes} box(es)")
    return len(images), total_boxes


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Redraw preview images from the CURRENT label files (model-free). Run this after "
            "fixing labels so new_data/preview/ reflects your corrections. Does NOT touch labels."
        )
    )
    parser.add_argument("--images", type=Path, default=PROJECT_ROOT / "new_data" / "images", help="Images dir.")
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "new_data" / "labels", help="Labels dir (YOLO .txt).")
    parser.add_argument("--preview", type=Path, default=PROJECT_ROOT / "new_data" / "preview", help="Output preview dir.")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "paper_detect" / "data.yaml", help="Dataset YAML for class names.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.images.is_dir():
        raise SystemExit(f"Images folder not found: {args.images}")
    n_imgs, n_boxes = render_previews(args.images, args.labels, args.preview, args.data)
    if n_imgs == 0:
        raise SystemExit(f"No images in {args.images}")
    print(f"\nRedrew {n_imgs} previews ({n_boxes} boxes) -> {args.preview}")


if __name__ == "__main__":
    main()
