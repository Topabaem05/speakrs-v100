"""Figures for the speakrs-v100 README, one chart per image, in the figures4papers house
style (https://github.com/ChenLiu-1996/figures4papers, scientific-figure-making/references):
semantic palette (blue = ours, red = baseline, neutral = library), top/right spines off,
frameless legends, black-edged bars with values, tight_layout(pad=2), 300 dpi PNG and SVG.

Baseline is the upstream FP32 trunk kernels built for sm_70 (milestone M1); Ours is the
Volta wmma FP16 trunk (milestone M2). Every number was measured on a Tesla
V100-SXM2-16GB with a 4-core host on 2026-11-01/02; see V100_STATUS.md for the runs.

usage: uv run --no-project --with matplotlib --with numpy python make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "teal": "#42949E",
    "gray_dark": "#4D4D4D",
}
BASELINE = ("Baseline (FP32 kernels)", PALETTE["red_2"])
OURS = ("Ours (Volta FP16 tiles)", PALETTE["blue_main"])
LIBRARY = ("cuDNN 9.10.2", PALETTE["neutral"])
OUT = Path(__file__).resolve().parent


def style(font_size=16, axes_linewidth=2.5):
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": font_size,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": axes_linewidth,
        "legend.frameon": False,
        "svg.fonttype": "none",
    })


def finalize(fig, name, pad=2):
    fig.tight_layout(pad=pad)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)


def label_bars(ax, bars, fmt, fontsize=13, bold=False, color="black"):
    for bar in bars:
        ax.annotate(fmt.format(bar.get_height()),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    fontsize=fontsize, fontweight="bold" if bold else "normal", color=color)


def paired_bars(ax, groups, baseline, ours, fmt, width=0.36, extra=None):
    """Baseline and Ours side by side per group, optional library bars in the middle"""
    x = np.arange(len(groups))
    sets = [(baseline, *BASELINE, "")]
    if extra is not None:
        sets.append((extra, *LIBRARY, "//"))
    sets.append((ours, *OURS, ""))
    n = len(sets)
    w = width if n == 2 else 0.26
    for k, (values, label, color, hatch) in enumerate(sets):
        bars = ax.bar(x + (k - (n - 1) / 2) * w, values, w, color=color, edgecolor="black",
                      linewidth=2, hatch=hatch, label=label)
        is_ours = label == OURS[0]
        label_bars(ax, bars, fmt, 12, bold=is_ours,
                   color=PALETTE["blue_main"] if is_ours else "black")
    ax.set_xticks(x, groups)
    return x


def box(ax, xy, w, h, text, face, fontsize=12, bold=False, color="black"):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=face, edgecolor="black", linewidth=1.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal", color=color)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                                 color="black", linewidth=1.8))


# --- diagrams ----------------------------------------------------------------------------

def diagram_pipeline():
    style(font_size=13)
    fig, ax = plt.subplots(figsize=(15, 2.9))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 2.9)
    ax.set_axis_off()
    box(ax, (0.2, 1.2), 1.6, 1.1, "audio\n16 kHz", "white")
    box(ax, (2.3, 1.2), 2.7, 1.1, "segmentation\nSincNet + 4-layer BiLSTM\nFP32", PALETTE["neutral"])
    box(ax, (5.5, 1.2), 1.6, 1.1, "fbank\nFP32", PALETTE["neutral"])
    box(ax, (7.6, 1.2), 4.3, 1.1, "ResNet34 embedding trunk\nOurs: 3x3 layers on Volta FP16 tiles",
        PALETTE["blue_main"], color="white", bold=True)
    box(ax, (12.4, 1.2), 2.4, 1.1, "PLDA + VBx\nclustering (CPU)", PALETTE["neutral"])
    for x0, x1 in ((1.8, 2.3), (5.0, 5.5), (7.1, 7.6), (11.9, 12.4)):
        arrow(ax, (x0, 1.75), (x1, 1.75))
    ax.text(9.75, 0.65, "Baseline: the same layers on FP32 kernels", ha="center", fontsize=12,
            color=PALETTE["red_strong"])
    ax.text(7.5, 0.15, "cuda-v100 build: the whole pipeline runs on the V100 without cuDNN or cuBLAS",
            ha="center", fontsize=11, color=PALETTE["gray_dark"])
    finalize(fig, "diagram_pipeline", pad=0.5)


def diagram_build():
    style(font_size=13)
    fig, ax = plt.subplots(figsize=(15, 2.9))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 2.9)
    ax.set_axis_off()
    steps = [
        ("Rust kernels\n#[kernel]", "white"),
        ("cuda-oxide 918bbde\n--arch sm_70", PALETTE["green_1"]),
        ("PTX ISA 6.0\nwmma + nounroll\npragma", PALETTE["green_2"]),
        ("ptxas 12.9\n(CUDA 13 dropped\nVolta)", PALETTE["green_3"]),
        ("sm_70 cubin\nspeakrs-ptx-sm70", PALETTE["blue_main"]),
    ]
    x = 0.2
    for i, (label, face) in enumerate(steps):
        box(ax, (x, 1.0), 2.55, 1.4, label, face,
            color="white" if face == PALETTE["blue_main"] else "black")
        if i:
            arrow(ax, (x - 0.45, 1.7), (x, 1.7))
        x += 3.0
    ax.text(7.5, 0.35, "sm75/sm80/sm120 tiers keep CUDA 13.0 ptxas and stay byte-identical; a lint "
            "rejects sm_75+ opcodes in sm70 PTX", ha="center", fontsize=11,
            color=PALETTE["gray_dark"])
    finalize(fig, "diagram_build", pad=0.5)


def diagram_tile():
    style(font_size=13)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_axis_off()
    for r in range(6):
        for c in range(11):
            face = PALETTE["green_2"] if 1 <= r <= 4 and 1 <= c <= 9 else PALETTE["green_1"]
            ax.add_patch(plt.Rectangle((0.3 + c * 0.32, 1.3 + r * 0.6), 0.3, 0.55,
                                       facecolor=face, edgecolor="black", linewidth=0.6))
    ax.text(2.05, 0.6, "input stage: 6 x 66 pixels with halo\n16 channels x FP16 = 32 B per "
            "pixel, double-buffered", ha="center", fontsize=11)
    arrow(ax, (3.95, 3.1), (4.9, 3.1))
    ax.text(4.42, 3.35, "wmma.load.b", ha="center", fontsize=10)
    for w in range(4):
        box(ax, (5.0, 4.6 - w * 1.05), 3.2, 0.85, f"warp {w}: output row {w}\n64 columns x 64 channels",
            PALETTE["blue_main"], color="white", fontsize=11)
    ax.text(6.6, 0.6, "4 x 4 wmma m16n16k16 accumulators\nper warp (FP32)", ha="center", fontsize=11)
    arrow(ax, (8.3, 3.1), (9.1, 3.1))
    ax.text(9.5, 3.1, "epilogue\nx2^-20\n+residual\n+bias\nReLU", ha="center", va="center",
            fontsize=10)
    ax.text(0.2, 5.75, "One CTA of the C64 layers (128 threads)", fontsize=14, fontweight="bold")
    finalize(fig, "diagram_tile", pad=0.5)


def diagram_loop():
    style(font_size=13)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_axis_off()
    lines = [
        ("for each chunk of 16 input channels:", 0),
        ("half 0: load channels 0-7 (global -> registers)", 1),
        ("rolled tap loop 0..4   .pragma \"nounroll\"", 1),
        ("wmma.load.a + 16 x wmma.mma", 2),
        ("convert x2^10 to FP16, range check, store stage", 1),
        ("half 1: channels 8-15, taps 5..8", 1),
        ("bar.sync", 1),
        ("epilogue per channel tile:", 0),
        ("wmma.store -> shared 16 x 68 tile", 1),
        ("x2^-20, + residual, + bias, ReLU, FP16/FP32 out", 1),
    ]
    for i, (text, depth) in enumerate(lines):
        ax.text(0.2 + 0.5 * depth, 5.35 - i * 0.48, text, fontsize=12.5, family="monospace",
                fontweight="bold" if depth == 0 else "normal",
                color=PALETTE["red_strong"] if "nounroll" in text else "black")
    ax.text(0.2, 0.3, "Same operand rounding, range flag and epilogue order as the sm75+ tiers.",
            fontsize=11, color=PALETTE["gray_dark"])
    finalize(fig, "diagram_loop", pad=0.5)


# --- comparison charts -------------------------------------------------------------------

def chart_speedup():
    """End-to-end speed relative to the baseline (A/B medians, 4 long files)"""
    style(font_size=16)
    fig, ax = plt.subplots(figsize=(8, 6))
    base_s, ours_s = np.array([16.8, 9.9]), np.array([5.6, 3.2])
    paired_bars(ax, ["cuda", "cuda-fast"], [1.0, 1.0], base_s / ours_s, "{:.2f}x")
    ax.axhline(1.0, color=PALETTE["red_strong"], linewidth=1.5, linestyle="--")
    ax.set_ylim(0, 3.8)
    ax.set_ylabel("end-to-end speed (Baseline = 1.0x)")
    ax.set_title("End-to-end speedup", loc="left")
    ax.legend(loc="upper left", fontsize=13)
    ax.text(0.5, -0.17, "median wall time: 16.8 s -> 5.6 s (cuda), 9.9 s -> 3.2 s (cuda-fast); "
            "4 files, 69.6 min",
            transform=ax.transAxes, ha="center", fontsize=11, color=PALETTE["gray_dark"])
    finalize(fig, "chart_speedup")


def chart_rtfx():
    """Real-time factor of the same A/B runs"""
    style(font_size=16)
    fig, ax = plt.subplots(figsize=(8, 6))
    paired_bars(ax, ["cuda", "cuda-fast"], [249, 422], [746, 1305], "{:.0f}x")
    ax.set_ylim(0, 1550)
    ax.set_ylabel("real-time factor (audio / processing time)")
    ax.set_title("Throughput", loc="left")
    ax.legend(loc="upper left", fontsize=13)
    finalize(fig, "chart_rtfx")


def chart_der():
    """DER on all 216 VoxConverse dev files"""
    style(font_size=16)
    fig, ax = plt.subplots(figsize=(8, 6))
    paired_bars(ax, ["cuda", "cuda-fast"], [7.012545, 7.367643], [7.011792, 7.367643], "{:.4f}")
    ax.set_ylim(6.5, 7.75)
    ax.set_ylabel("DER (%) - lower is better")
    ax.set_title("Accuracy is unchanged", loc="left")
    ax.legend(loc="upper left", fontsize=13)
    ax.text(0.5, -0.17, "RTTMs byte-identical to Baseline: 213/216 (cuda), 215/216 (cuda-fast)",
            transform=ax.transAxes, ha="center", fontsize=11, color=PALETTE["gray_dark"])
    finalize(fig, "chart_der")


def chart_layers():
    """Per-layer-class speed at batch 32 relative to the baseline kernels"""
    style(font_size=16)
    fig, ax = plt.subplots(figsize=(14, 6))
    classes = ["C32 (x6)", "C64 (x7)", "C128 (x11)", "C256 (x5)", "C32 s2", "C64 s2", "C128 s2"]
    fp32 = np.array([24.634, 25.780, 23.712, 11.134, 2.502, 2.191, 1.996])
    cudnn = np.array([23.915, 20.656, 21.236, 8.249, 1.869, 1.669, 1.626])
    fp16 = np.array([9.104, 7.845, 9.470, 3.784, 1.101, 0.685, 0.545])
    paired_bars(ax, classes, np.ones(7), fp32 / fp16, "{:.1f}x", extra=fp32 / cudnn)
    ax.axhline(1.0, color=PALETTE["red_strong"], linewidth=1.5, linestyle="--")
    ax.set_ylim(0, 4.3)
    ax.set_ylabel("layer speed (Baseline = 1.0x)")
    ax.set_title("3x3 trunk layers at batch 32", loc="left")
    ax.legend(loc="upper left", fontsize=13, ncol=3)
    finalize(fig, "chart_layers")


def chart_batches():
    """Speed of all 32 3x3 trunk layers against the embedding batch"""
    style(font_size=16)
    fig, ax = plt.subplots(figsize=(8, 6))
    batches = np.array([1, 4, 8, 16, 32])
    fp32 = np.array([4.164, 11.689, 23.442, 46.125, 91.949])
    cudnn = np.array([5.464, 19.497, 32.686, 40.139, 79.220])
    fp16 = np.array([2.091, 4.576, 8.399, 16.469, 32.534])
    for values, (label, color), marker, offset in [
        (fp16, OURS, "o", 10), (cudnn, LIBRARY, "s", 10), (fp32, BASELINE, "^", 10),
    ]:
        speed = fp32 / values
        ax.plot(batches, speed, color=PALETTE["gray_dark"] if color == PALETTE["neutral"] else color,
                linewidth=2.8, marker=marker, markersize=10, label=label)
        if label != BASELINE[0]:
            for b, v in zip(batches, speed):
                ax.annotate(f"{v:.2f}x", (b, v), xytext=(0, offset), textcoords="offset points",
                            ha="center", fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_xticks(batches, [str(b) for b in batches])
    ax.set_ylim(0, 3.4)
    ax.set_xlabel("embedding batch")
    ax.set_ylabel("trunk speed (Baseline = 1.0x)")
    ax.set_title("All 32 3x3 layers", loc="left")
    ax.legend(loc="center right", fontsize=12)
    finalize(fig, "chart_batches")


def chart_kernel_time():
    """GPU kernel time by stage for one 607 s file"""
    style(font_size=16)
    fig, ax = plt.subplots(figsize=(11, 5))
    parts = [
        ("3x3 trunk (+stem in Baseline)", [2090.1, 490.6], PALETTE["blue_main"]),
        ("stem + shortcuts", [0, 39.4], PALETTE["blue_secondary"]),
        ("LSTM", [555.3, 314.4], PALETTE["teal"]),
        ("other segmentation", [83.8, 72.9], PALETTE["green_3"]),
        ("fbank + embedding head", [20.7, 23.6], PALETTE["neutral"]),
    ]
    rows = ["Baseline", "Ours"]
    left = np.zeros(2)
    for label, values, color in parts:
        values = np.array(values) / 1000
        ax.barh(rows, values, left=left, color=color, edgecolor="black", linewidth=1.5,
                label=label, height=0.55)
        left += values
    for i, total in enumerate(left):
        ax.text(total + 0.04, i, f"{total:.2f} s", va="center", fontsize=15, fontweight="bold",
                color=PALETTE["blue_main"] if i else "black")
    ax.set_xlim(0, 3.3)
    ax.invert_yaxis()
    ax.set_xlabel("GPU kernel time for one 607 s file (s)")
    ax.set_title("Kernel time by stage", loc="left")
    ax.legend(fontsize=11, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    finalize(fig, "chart_kernel_time")


def chart_nounroll():
    """Ablation inside Ours: tap loop unrolled by ptxas against kept rolled"""
    style(font_size=16)
    fig, ax = plt.subplots(figsize=(8, 6))
    kernels = ["C32", "C64", "C128", "C256"]
    unrolled = np.array([178.9, 130.1, 130.7, 194.7])
    rolled = np.array([126.9, 96.2, 114.4, 193.0])
    x = np.arange(4)
    w = 0.36
    b1 = ax.bar(x - w / 2, unrolled, w, color=PALETTE["red_1"], edgecolor="black", linewidth=2,
                label="Ours without nounroll (spills)")
    b2 = ax.bar(x + w / 2, rolled, w, color=PALETTE["blue_main"], edgecolor="black", linewidth=2,
                label='Ours (.pragma "nounroll")')
    label_bars(ax, b1, "{:.0f}", 12)
    label_bars(ax, b2, "{:.0f}", 12, bold=True, color=PALETTE["blue_main"])
    ax.set_xticks(x, kernels)
    ax.set_ylim(0, 260)
    ax.set_ylabel("kernel time at batch 2 (us)")
    ax.set_title("Ablation: rolled tap loop", loc="left")
    ax.legend(loc="upper left", fontsize=12)
    finalize(fig, "chart_nounroll")


if __name__ == "__main__":
    for old in OUT.glob("*.png"):
        old.unlink()
    for old in OUT.glob("*.svg"):
        old.unlink()
    for draw in (diagram_pipeline, diagram_build, diagram_tile, diagram_loop, chart_speedup,
                 chart_rtfx, chart_der, chart_layers, chart_batches, chart_kernel_time,
                 chart_nounroll):
        draw()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
