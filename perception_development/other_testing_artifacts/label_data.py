import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Open labelImg on the new_data images to review/fix the auto-generated labels. "
            "Generates classes.txt from the dataset so class names are always correct."
        )
    )
    parser.add_argument("--images", type=Path, default=PROJECT_ROOT / "new_data" / "images", help="Images to label.")
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "new_data" / "labels", help="Where labels are saved.")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "paper_detect" / "data.yaml", help="Dataset YAML (source of class names).")
    parser.add_argument("--no-launch", action="store_true", help="Set up classes.txt only; don't open labelImg.")
    return parser.parse_args()


def load_class_names(data_yaml):
    if not data_yaml.is_file():
        raise SystemExit(f"Dataset YAML not found: {data_yaml}")
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required. Install deps with `pip install -r requirements.txt`.") from exc
    with open(data_yaml, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    names = data.get("names")
    if isinstance(names, dict):  # support {0: '10', 1: 'paper'}
        names = [names[key] for key in sorted(names)]
    if not names:
        raise SystemExit(f"No class names found in {data_yaml}")
    return [str(name) for name in names]


def write_classes(path, names):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + "\n", encoding="utf-8")


def find_labelimg():
    exe = shutil.which("labelImg")
    if exe:
        return exe
    bindir = Path(sys.executable).parent  # venv Scripts/ or bin/
    for name in ("labelImg.exe", "labelImg"):
        candidate = bindir / name
        if candidate.exists():
            return str(candidate)
    return None


def main():
    args = parse_args()
    images_dir = args.images
    labels_dir = args.labels

    if not images_dir.is_dir():
        raise SystemExit(
            f"Images folder not found: {images_dir}\n"
            "Create it, drop images in, and run `python autolabel.py` first."
        )
    images = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    if not images:
        raise SystemExit(f"No images in {images_dir}. Add some and run `python autolabel.py` first.")

    labels_dir.mkdir(parents=True, exist_ok=True)

    names = load_class_names(args.data)
    classes_file = images_dir.parent / "classes.txt"
    write_classes(classes_file, names)
    write_classes(labels_dir / "classes.txt", names)

    labelled = sum(1 for p in images if (labels_dir / f"{p.stem}.txt").exists())
    print(f"Classes : {names}")
    print(f"Images  : {len(images)} in {images_dir}")
    print(f"Labels  : {labelled}/{len(images)} have a label file in {labels_dir}")
    if labelled < len(images):
        print("  (tip: run `python autolabel.py` to pre-label the rest before reviewing)")

    if args.no_launch:
        print("\nSetup done (--no-launch). Open labelImg yourself when ready.")
        return

    exe = find_labelimg()
    if not exe:
        raise SystemExit("labelImg not found. Install it with:  pip install labelImg")

    print("\nIn labelImg: click the format button on the left toolbar until it says YOLO.")
    print("Keys: W=new box, D/A=next/prev, Del=delete box, Ctrl+S=save.")
    print("Launching labelImg... (previews refresh automatically when you close it)\n")
    subprocess.run([exe, str(images_dir), str(classes_file), str(labels_dir)], check=False)

    try:
        from render_labels import render_previews

        preview_dir = images_dir.parent / "preview"
        n_imgs, n_boxes = render_previews(images_dir, labels_dir, preview_dir, args.data, verbose=False)
        print(f"Refreshed {n_imgs} previews ({n_boxes} boxes) -> {preview_dir}")
    except Exception as exc:  # never fail the workflow over a preview refresh
        print(f"(Could not refresh previews: {exc})")


if __name__ == "__main__":
    main()
