"""Charts for the S.P.A.S. v2 deck.

Design rules applied (dataviz skill):
  - form picked first: grouped columns for a 2x2 magnitude comparison,
    horizontal bars for a single-series ranked magnitude.
  - categorical palette validated: #0065BD / #E37222 passes all six checks
    (worst adjacent CVD dE 26.8 protan, normal-vision dE 35.6, contrast >= 3:1).
  - thin marks, 4px rounded data-end square at the baseline, 2px surface gap
    between adjacent bars, recessive hairline axis, no gridlines because every
    value is directly labelled, text in ink tokens rather than series colour.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(OUT, exist_ok=True)

TUM_BLUE = "#0065BD"
TUM_ORANGE = "#E37222"
INK = "#1A1A1A"
INK_2 = "#5A5A5A"
INK_3 = "#8A8A8A"
SURFACE = "#FFFFFF"
HAIRLINE = "#D8D8D8"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

DPI = 300


def rounded_bar(ax, x, y, w, h, color, radius, vertical=True):
    """Bar with a rounded data-end and a square baseline end."""
    r = min(radius, abs(h) / 2 if vertical else abs(w) / 2, (w if vertical else h) / 2)
    if vertical:
        verts = [
            (x, y), (x, y + h - r),
            (x, y + h), (x + r, y + h),
            (x + w - r, y + h),
            (x + w, y + h), (x + w, y + h - r),
            (x + w, y), (x, y),
        ]
        codes = [Path.MOVETO, Path.LINETO,
                 Path.CURVE3, Path.CURVE3,
                 Path.LINETO,
                 Path.CURVE3, Path.CURVE3,
                 Path.LINETO, Path.CLOSEPOLY]
    else:
        verts = [
            (x, y), (x + w - r, y),
            (x + w, y), (x + w, y + r),
            (x + w, y + h - r),
            (x + w, y + h), (x + w - r, y + h),
            (x, y + h), (x, y),
        ]
        codes = [Path.MOVETO, Path.LINETO,
                 Path.CURVE3, Path.CURVE3,
                 Path.LINETO,
                 Path.CURVE3, Path.CURVE3,
                 Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", zorder=3))


def chart_detector_f1():
    """Grouped columns: two detectors x two frozen test sets."""
    groups = ["Photographs", "Robot video"]
    deployed = [0.839, 0.924]      # SSD-MobileNet-V1 @ 0.20, the shipped model
    previous = [0.899, 0.945]      # custom TFOD/TensorRT model @ 0.50

    fig, ax = plt.subplots(figsize=(7.4, 3.4), dpi=DPI)

    band = 1.0
    bw = 0.19          # thin: the band keeps its air
    gap = 0.018        # ~2px surface gap between the touching pair
    for i, (d, p) in enumerate(zip(deployed, previous)):
        left = i * band
        rounded_bar(ax, left - bw - gap / 2, 0, bw, d, TUM_BLUE, 0.022)
        rounded_bar(ax, left + gap / 2, 0, bw, p, TUM_ORANGE, 0.022)
        ax.text(left - bw / 2 - gap / 2, d + 0.025, "{:.3f}".format(d),
                ha="center", va="bottom", fontsize=13, color=INK, fontweight="bold")
        ax.text(left + bw / 2 + gap / 2, p + 0.025, "{:.3f}".format(p),
                ha="center", va="bottom", fontsize=13, color=INK, fontweight="bold")
        ax.text(left, -0.04, groups[i], ha="center", va="top", fontsize=13, color=INK_2)

    ax.set_xlim(-0.58, band + 0.58)
    ax.set_ylim(0, 1.03)
    ax.axhline(0, color=HAIRLINE, linewidth=1.0, zorder=2)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # legend: always present for two or more series, swatch carries identity
    handles = [
        plt.Line2D([0], [0], marker="s", markersize=9, linestyle="none",
                   markerfacecolor=TUM_BLUE, markeredgecolor="none",
                   label="Deployed SSD-MobileNet-V1, threshold 0.20"),
        plt.Line2D([0], [0], marker="s", markersize=9, linestyle="none",
                   markerfacecolor=TUM_ORANGE, markeredgecolor="none",
                   label="Replaced custom TensorRT model, threshold 0.50"),
    ]
    leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10),
                    ncol=1, frameon=False, fontsize=12, handletextpad=0.6,
                    labelspacing=0.5)
    for t in leg.get_texts():
        t.set_color(INK_2)

    ax.text(-0.58, 0.965, "F$_1$ score", fontsize=11.5, color=INK_3, ha="left", va="bottom")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.30)
    fig.savefig(os.path.join(OUT, "chart_detector_f1.png"), dpi=DPI)
    plt.close(fig)


def chart_phase_time():
    """Horizontal bars: commanded base motion per mission phase, one series."""
    phases = [
        ("Align can", 5.68, "72 steps"),
        ("Approach bin", 6.12, "35 pulses"),
        ("Approach can", 5.40, "10 pulses"),
        ("Align bin", 3.20, "9 steps"),
        ("Align near", 2.10, "15 steps"),
        ("Search bin", 1.68, ""),
        ("Push into gripper", 7.00, "one 7.0 s pulse"),
    ]
    phases.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=DPI)
    bh = 0.42
    for i, (name, val, note) in enumerate(phases):
        rounded_bar(ax, 0, i - bh / 2, val, bh, TUM_BLUE, 0.16, vertical=False)
        label = "{:.2f} s".format(val)
        if note:
            label += "    {}".format(note)
        ax.text(val + 0.16, i, label, ha="left", va="center", fontsize=11.5, color=INK)
        ax.text(-0.16, i, name, ha="right", va="center", fontsize=12, color=INK_2)

    ax.set_xlim(0, 11.6)
    ax.set_ylim(-0.75, len(phases) - 0.25)
    ax.axvline(0, color=HAIRLINE, linewidth=1.0, zorder=2)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.subplots_adjust(left=0.22, right=0.99, top=0.98, bottom=0.03)
    fig.savefig(os.path.join(OUT, "chart_phase_time.png"), dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    chart_detector_f1()
    chart_phase_time()
    print("wrote", os.listdir(OUT))
