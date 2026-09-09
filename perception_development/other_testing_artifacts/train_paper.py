import argparse
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(PROJECT_ROOT / ".cache" / "matplotlib"),
)

import torch

try:
    import psutil

    _PHYSICAL_CORES = psutil.cpu_count(logical=False) or os.cpu_count() or 1
except ImportError:
    _PHYSICAL_CORES = os.cpu_count() or 1

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train a low-memory paper detector.")
    parser.add_argument("--model", type=Path, default=Path("yolo11n.pt"), help="Base YOLO model path. yolo11n (nano, ~2.6M params) is the variant that runs in real time on the JETANK's Jetson Nano (4 GB shared CPU/GPU memory). Training still happens on the RTX 3060; only the model size is capped for the deployment target. Use yolo11s if the Nano has FPS headroom.")
    parser.add_argument("--data", type=Path, default=Path("paper_detect/data.yaml"), help="Dataset YAML path.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs. Small dataset (226 imgs) benefits from many epochs; ")
    parser.add_argument("--patience", type=int, default=50, )
    parser.add_argument("--imgsz", type=int, default=416, help="Idk hopefully it runs better than last time.")
    parser.add_argument("--batch", default=-1,)
    parser.add_argument("--workers", type=int, default=8, help="Data loader workers (parallel CPU data loading/augmentation).")
    parser.add_argument("--device", default="0", help="Training device. '0' = first CUDA GPU (RTX 3060 Laptop). Pass 'cpu' to force CPU.")
    parser.add_argument(
        "--threads",
        type=int,
        default=_PHYSICAL_CORES,
        help="PyTorch CPU compute threads (intra-op). Default: physical core count.",
    )
    parser.add_argument(
        "--cache",
        choices=["ram", "disk", "none"],
        default="ram",
        help="Cache images to speed up every epoch after the first. Default: ram (32 GB on the training machine easily holds this dataset).",
    )
    parser.add_argument("--project", type=Path, default=Path("paper_detect/runs"), help="Output directory.")
    parser.add_argument("--name", default="train", help="Run name.")
    parser.add_argument(
        "--save-period",
        type=int,
        default=10,
        help="Save a numbered checkpoint (epoch{N}.pt) every N epochs, in addition to "
        "the always-updated last.pt/best.pt. These distinct files survive a crash that "
        "corrupts last.pt mid-write, and let you resume from any earlier point. -1 disables.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="last",
        default=None,
    )
    return parser.parse_args()


def parse_batch(value):
    """Batch may be an int (fixed) or -1 (Ultralytics auto-fit ~60% VRAM)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def project_path(path):
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_resume(resume_value, project_path_value, name):
    """Return the checkpoint path to resume from, or None for a fresh run.
    `--resume` (bare) -> the run's weights/last.pt.
    `--resume <path>` -> that specific checkpoint (e.g. .../weights/epoch50.pt).
    """
    if resume_value is None:
        return None
    if resume_value == "last":
        ckpt = project_path_value / name / "weights" / "last.pt"
    else:
        ckpt = project_path(Path(resume_value))
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Cannot resume: checkpoint not found at {ckpt}. "
            "Check --name/--project, or pass an explicit checkpoint path to --resume."
        )
    return ckpt


def main():
    args = parse_args()
    batch = parse_batch(args.batch)
    torch.set_num_threads(args.threads)
    cuda_ok = torch.cuda.is_available()
    if str(args.device) != "cpu" and not cuda_ok:
        print(
            f"[train] WARNING: device={args.device} requested but CUDA is unavailable. "
            "Install the CUDA build of torch (pip install -r requirements-gpu.txt) "
            "or pass --device cpu."
        )
    gpu = torch.cuda.get_device_name(0) if cuda_ok else "none"
    print(
        f"[train] device={args.device} gpu={gpu} threads={args.threads} "
        f"model={args.model} imgsz={args.imgsz} epochs={args.epochs} "
        f"batch={batch} workers={args.workers} cache={args.cache} "
        f"save_period={args.save_period}"
    )
    model_path = project_path(args.model)
    data_path = project_path(args.data)
    project_path_value = project_path(args.project)

    resume_ckpt = resolve_resume(args.resume, project_path_value, args.name)

    if resume_ckpt is not None:
        print(f"[train] resuming from {resume_ckpt}")
        model = YOLO(str(resume_ckpt))
        model.train(resume=True)
    else:
        model = YOLO(str(model_path))
        model.train(
            data=str(data_path),
            epochs=args.epochs,
            patience=args.patience,
            imgsz=args.imgsz,
            batch=batch,
            workers=args.workers,
            device=args.device,
            cache=(False if args.cache == "none" else args.cache),
            amp=True,
            cos_lr=True,         
            optimizer="auto",    
            close_mosaic=10,     
            mixup=0.1,
            copy_paste=0.1,
            save_period=args.save_period,  # numbered epoch{N}.pt checkpoints for crash recovery
            project=str(project_path_value),
            name=args.name,
            exist_ok=True,
        )

    trained_best = project_path_value / args.name / "weights" / "best.pt"
    default_best = PROJECT_ROOT / "best.pt"
    if trained_best.exists():
        shutil.copy2(trained_best, default_best)
        print(f"Copied trained model to {default_best}")


if __name__ == "__main__":
    main()
