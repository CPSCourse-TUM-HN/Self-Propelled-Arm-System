import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(PROJECT_ROOT / ".cache" / "matplotlib"),
)

import cv2

try:
    from ultralytics import YOLO  
except ImportError as exc:
    raise ImportError(
        "The ultralytics package is required. Install it with `pip install -r requirements.txt`."
    ) from exc

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Model-assisted labeling: run the trained model on new images, write YOLO "
            "labels + annotated previews so you can review what it recognized before training."
        )
    )
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "best.pt", help="Trained YOLO .pt weights.")
    parser.add_argument("--images", type=Path, default=PROJECT_ROOT / "new_data" / "images", help="Folder of new images.")
    parser.add_argument("--labels", type=Path, default=None, help="Output labels dir. Default: <images>/../labels.")
    parser.add_argument("--preview", type=Path, default=None, help="Output annotated-preview dir. Default: <images>/../preview.")
    parser.add_argument("--conf", type=float, default=0.25, help="Min confidence to keep a box. Lower surfaces more candidates to review. Default 0.25.")
    parser.add_argument("--imgsz", type=int, default=416, help="Inference image size (match training). Default 416.")
    return parser.parse_args()


def main():
    args = parse_args()
    images_dir = args.images
    labels_dir = args.labels or images_dir.parent / "labels"
    preview_dir = args.preview or images_dir.parent / "preview"

    if not images_dir.is_dir():
        raise SystemExit(
            f"Image folder not found: {images_dir}\n"
            "Create it and drop your new images inside, then re-run."
        )
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        raise SystemExit(f"No images in {images_dir} (extensions: {sorted(IMAGE_EXTS)}).")

    labels_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    names = model.names

    total = 0
    per_class = {}
    empty = 0
    for img_path in image_paths:
        result = model.predict(source=str(img_path), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        boxes = result.boxes
        lines = []
        if boxes is not None and len(boxes):
            xywhn = boxes.xywhn.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            for cls_id, (cx, cy, w, h), conf in zip(classes, xywhn, confs):
                lines.append(f"{int(cls_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                label_name = names[int(cls_id)]
                per_class[label_name] = per_class.get(label_name, 0) + 1
                total += 1
        else:
            empty += 1
        (labels_dir / f"{img_path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        cv2.imwrite(str(preview_dir / img_path.name), result.plot())
        print(f"  {img_path.name}: {len(lines)} detections")

    print(f"\nProcessed {len(image_paths)} images -> {total} detections {per_class}")
    print(f"Images with no detection: {empty} (empty label written = treated as background)")
    print(f"Labels  : {labels_dir}")
    print(f"Preview : {preview_dir}")
    print("\nReview the preview images and FIX the labels (add missed boxes, delete wrong ones)")
    print("in a labeling tool, then merge with: python add_reviewed_data.py")


if __name__ == "__main__":
    main()
