"""Regenerate figures/training_curves.pdf from the YOLO training log.

Reads paper_detect/runs/train/results.csv and writes the two-panel figure used
as Figure "Offline detector training" in SPAS_technical_report.tex.

Run from anywhere:

    python writing/report/figures/make_training_curves.py

Requires matplotlib.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
RESULTS = PROJECT_ROOT / "paper_detect" / "runs" / "train" / "results.csv"


def main():
    if not RESULTS.exists():
        raise SystemExit("training log not found: {}".format(RESULTS))

    with RESULTS.open() as handle:
        rows = list(csv.DictReader(handle))

    epoch = [int(r["epoch"]) for r in rows]
    train_box = [float(r["train/box_loss"]) for r in rows]
    train_cls = [float(r["train/cls_loss"]) for r in rows]
    val_box = [float(r["val/box_loss"]) for r in rows]
    val_cls = [float(r["val/cls_loss"]) for r in rows]
    map50 = [float(r["metrics/mAP50(B)"]) for r in rows]
    map95 = [float(r["metrics/mAP50-95(B)"]) for r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.3))

    ax[0].plot(epoch, train_box, label="train box", lw=1.6, color="#1f4e79")
    ax[0].plot(epoch, train_cls, label="train cls", lw=1.6, color="#c0504d")
    ax[0].plot(epoch, val_box, label="val box", lw=1.4, ls="--", color="#1f4e79")
    ax[0].plot(epoch, val_cls, label="val cls", lw=1.4, ls="--", color="#c0504d")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].set_title("(a) Training and validation loss", fontsize=10)
    ax[0].legend(fontsize=8, frameon=False)
    ax[0].grid(alpha=0.25)

    ax[1].plot(epoch, map50, label="mAP@0.5", lw=1.8, color="#2e7d32")
    ax[1].plot(epoch, map95, label="mAP@0.5:0.95", lw=1.8, color="#ef6c00")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("mAP")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title("(b) Validation detection accuracy", fontsize=10)
    ax[1].legend(fontsize=8, frameon=False, loc="lower right")
    ax[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(HERE / "training_curves.pdf")
    fig.savefig(HERE / "training_curves.png", dpi=160)

    best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
    print("wrote training_curves.pdf and training_curves.png")
    print(
        "best epoch {} : mAP50={} mAP50-95={}".format(
            best["epoch"],
            best["metrics/mAP50(B)"],
            best["metrics/mAP50-95(B)"],
        )
    )


if __name__ == "__main__":
    main()
