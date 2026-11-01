"""Figures for the speakrs-v100 README, in the figures4papers house style
(https://github.com/ChenLiu-1996/figures4papers, scientific-figure-making/references):
semantic palette, top/right spines off, frameless legends, black-edged bars with values,
tight_layout(pad=2), 300 dpi PNG plus SVG.

Every number below was measured on a Tesla V100-SXM2-16GB with a 4-core host on
2026-11-01/02; see V100_STATUS.md for the runs.

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
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "gray_dark": "#4D4D4D",
}
OUT = Path(__file__).resolve().parent


def apply_publication_style(font_size=16, axes_linewidth=2.5):
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


def annotate(ax, bars, fmt, fontsize=12, padding=3):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(fmt.format(height), (bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, padding), textcoords="offset points", ha="center",
                    va="bottom", fontsize=fontsize)


def box(ax, xy, w, h, text, face, edge="black", fontsize=12, bold=False, color="black"):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=face, edgecolor=edge, linewidth=1.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal", color=color)


def arrow(ax, start, end, color="black", style="-|>", lw=1.8):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=16,
                                 color=color, linewidth=lw))


def figure_overview():
    """The pipeline on the V100 and the sm70 build path"""
    apply_publication_style(font_size=13)
    fig, ax = plt.subplots(figsize=(15, 6.6))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.6)
    ax.set_axis_off()

    ax.text(0.2, 6.25, "a  Diarization pipeline on the V100 (cuda-v100 build, no cuDNN needed)",
            fontsize=14, fontweight="bold")
    box(ax, (0.2, 4.6), 1.6, 1.0, "audio\n16 kHz", "white")
    box(ax, (2.3, 4.6), 2.7, 1.0, "segmentation\nSincNet + 4-layer BiLSTM\nFP32", PALETTE["neutral"])
    box(ax, (5.5, 4.6), 1.6, 1.0, "fbank\nFP32", PALETTE["neutral"])
    box(ax, (7.6, 4.6), 4.3, 1.0,
        "ResNet34 embedding trunk\n3x3 layers on sm70 wmma FP16 tiles\n(stem, shortcuts FP32)",
        PALETTE["blue_secondary"], color="white", bold=True)
    box(ax, (12.4, 4.6), 2.4, 1.0, "PLDA + VBx\nclustering (CPU)", PALETTE["neutral"])
    for x0, x1 in ((1.8, 2.3), (5.0, 5.5), (7.1, 7.6), (11.9, 12.4)):
        arrow(ax, (x0, 5.1), (x1, 5.1))
    ax.text(9.75, 4.25, "TF32 mode (embedding default) routes here; FP32 mode keeps FP32 kernels",
            ha="center", fontsize=11, color=PALETTE["blue_main"])

    ax.text(0.2, 3.35, "b  Building the sm70 kernel variant", fontsize=14, fontweight="bold")
    steps = [
        ("Rust kernels\n#[kernel]", "white"),
        ("cuda-oxide 918bbde\n--arch sm_70", PALETTE["green_1"]),
        ("PTX ISA 6.0\nwmma, nounroll\npragma", PALETTE["green_2"]),
        ("ptxas 12.9\n(CUDA 13 dropped\nVolta)", PALETTE["green_3"]),
        ("sm_70 cubin\nin speakrs-ptx-sm70", PALETTE["blue_secondary"]),
    ]
    x = 0.2
    for i, (label, face) in enumerate(steps):
        box(ax, (x, 1.45), 2.55, 1.35, label, face,
            color="white" if face == PALETTE["blue_secondary"] else "black")
        if i:
            arrow(ax, (x - 0.45, 2.12), (x, 2.12))
        x += 3.0
    ax.text(0.2, 0.8,
            "The other tiers (sm75/sm80/sm120) keep the CUDA 13.0 ptxas; their PTX and cubins stay "
            "byte-identical. A lint rejects sm_75+ opcodes (ldmatrix, m16n8k8, cp.async) in sm70 PTX.",
            fontsize=11, color=PALETTE["gray_dark"])
    finalize(fig, "overview", pad=1)


def figure_tile():
    """How one sm70 FP16 CTA computes a 3x3 convolution tile"""
    apply_publication_style(font_size=13)
    fig, ax = plt.subplots(figsize=(15, 6.2))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.2)
    ax.set_axis_off()

    ax.text(0.2, 5.85, "a  One CTA of the C64 layers (128 threads, 4 warps)", fontsize=14, fontweight="bold")
    # input stage: (rows + 2) x 66 pixels, 32 bytes per pixel
    for r in range(6):
        for c in range(11):
            face = PALETTE["green_2"] if 1 <= r <= 4 and 1 <= c <= 9 else PALETTE["green_1"]
            ax.add_patch(plt.Rectangle((0.3 + c * 0.32, 1.2 + r * 0.6), 0.3, 0.55,
                                       facecolor=face, edgecolor="black", linewidth=0.6))
    ax.text(2.05, 0.75, "input stage: 6 x 66 pixels (halo included)\n16 channels x FP16 = 32 B "
            "per pixel, double-buffered", ha="center", fontsize=11)
    arrow(ax, (3.9, 3.0), (4.8, 3.0))
    ax.text(4.35, 3.25, "wmma.load.b", ha="center", fontsize=10)

    for w in range(4):
        box(ax, (4.9, 4.55 - w * 1.05), 3.0, 0.85,
            f"warp {w}: output row {w}\n64 columns x 64 channels", PALETTE["blue_secondary"],
            color="white", fontsize=11)
    ax.text(6.4, 0.75, "per warp: 4 x 4 accumulator tiles\n(m16n16k16, FP32 accumulate)",
            ha="center", fontsize=11)
    arrow(ax, (8.0, 3.0), (8.9, 3.0))
    ax.text(8.45, 3.25, "epilogue", ha="center", fontsize=10)

    ax.text(9.0, 5.85, "b  Loop structure", fontsize=14, fontweight="bold")
    lines = [
        ("for chunk of 16 input channels:", 0),
        ("half 0: load channels 0-7  (global -> registers)", 1),
        ("rolled tap loop 0..4   .pragma \"nounroll\"", 1),
        ("wmma.load.a (weights) + 16 x wmma.mma", 2),
        ("convert x 2^10 to FP16, range check, store stage", 1),
        ("half 1: same for channels 8-15, taps 5..8", 1),
        ("bar.sync", 1),
        ("epilogue per channel tile:", 0),
        ("wmma.store -> shared 16 x 68 tile -> x 2^-20", 1),
        ("+ residual, + bias, ReLU, FP16 or FP32 out", 1),
    ]
    for i, (text, depth) in enumerate(lines):
        bold = depth == 0
        color = PALETTE["red_strong"] if "nounroll" in text else "black"
        ax.text(9.1 + 0.45 * depth, 5.3 - i * 0.47, text, fontsize=11.5,
                fontweight="bold" if bold else "normal", color=color, family="monospace")
    ax.text(9.1, 0.55, "Same operand rounding, range flag and epilogue order as the sm75+ tiers;\n"
            "only the MMA shape (m16n16k16 instead of m16n8k8) and the staging width change.",
            fontsize=10.5, color=PALETTE["gray_dark"])
    finalize(fig, "sm70_tile", pad=1)


def figure_end_to_end():
    """M1 (FP32 trunk) against M2 (Volta FP16 trunk), end to end"""
    apply_publication_style(font_size=16, axes_linewidth=2.5)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), gridspec_kw={"width_ratios": [1.1, 1.1, 1]})
    modes = ["cuda", "cuda-fast"]
    width = 0.36
    x = np.arange(len(modes))

    # A/B, four longest dev files (4,177 s of audio), median of 5 alternating rounds
    ax = axes[0]
    m1, m2 = [16.8, 9.9], [5.6, 3.2]
    b1 = ax.bar(x - width / 2, m1, width, color=PALETTE["red_2"], edgecolor="black", linewidth=2,
                label="FP32 trunk (M1)")
    b2 = ax.bar(x + width / 2, m2, width, color=PALETTE["blue_main"], edgecolor="black",
                linewidth=2, label="Volta FP16 trunk (M2)")
    annotate(ax, b1, "{:.1f} s", 13)
    annotate(ax, b2, "{:.1f} s", 13)
    for i, (a, b) in enumerate(zip(m1, m2)):
        ax.text(i, max(a, b) * 1.13, f"{a / b:.2f}x", ha="center", fontsize=15,
                fontweight="bold", color=PALETTE["blue_main"])
    ax.set_xticks(x, modes)
    ax.set_ylim(0, 21)
    ax.set_ylabel("wall time (s)")
    ax.set_title("a  4 long files, 69.6 min of audio", loc="left", fontsize=15)
    ax.legend(loc="upper right", fontsize=13)

    # real-time factor of the same A/B medians and of the dev-216 harness runs
    ax = axes[1]
    ab_m1, ab_m2, harness, tuned = [249, 422], [746, 1305], [752, 1293], [766, 1312]
    w = 0.22
    sets = [(ab_m1, PALETTE["red_2"], "M1, A/B"), (ab_m2, PALETTE["blue_main"], "M2, A/B"),
            (harness, PALETTE["blue_secondary"], "M2, dev-216 harness"),
            (tuned, PALETTE["teal"], "M2 + V100 tune file")]
    for k, (values, color, label) in enumerate(sets):
        bars = ax.bar(x + (k - 1.5) * w, values, w, color=color, edgecolor="black", linewidth=2,
                      label=label)
        annotate(ax, bars, "{:.0f}", 11)
    ax.set_xticks(x, modes)
    ax.set_ylim(0, 1600)
    ax.set_ylabel("real-time factor (x)")
    ax.set_title("b  Throughput", loc="left", fontsize=15)
    ax.legend(loc="upper left", fontsize=12)

    # accuracy: DER on all 216 dev files
    ax = axes[2]
    der_m1, der_m2 = [7.012545, 7.367643], [7.011792, 7.367643]
    b1 = ax.bar(x - width / 2, der_m1, width, color=PALETTE["red_2"], edgecolor="black",
                linewidth=2, label="M1")
    b2 = ax.bar(x + width / 2, der_m2, width, color=PALETTE["blue_main"], edgecolor="black",
                linewidth=2, label="M2")
    annotate(ax, b1, "{:.4f}", 12)
    annotate(ax, b2, "{:.4f}", 12)
    ax.set_xticks(x, modes)
    ax.set_ylim(6.5, 7.6)
    ax.set_ylabel("DER (%)")
    ax.set_title("c  VoxConverse dev DER (216 files)", loc="left", fontsize=15)
    ax.set_xlabel("RTTMs byte-identical to M1: 213/216 (cuda), 215/216 (cuda-fast)", fontsize=12,
                  color=PALETTE["gray_dark"])
    finalize(fig, "end_to_end")


def figure_layers():
    """Per-layer 3x3 trunk time at batch 32 and the scaling with batch"""
    apply_publication_style(font_size=16, axes_linewidth=2.5)
    fig, axes = plt.subplots(1, 2, figsize=(20, 6.4), gridspec_kw={"width_ratios": [1.6, 1]})

    # CUDA-graph medians summed per layer class at batch 32
    classes = ["C32 (x6)", "C64 (x7)", "C128 (x11)", "C256 (x5)", "C32 s2", "C64 s2", "C128 s2"]
    fp32 = [24.634, 25.780, 23.712, 11.134, 2.502, 2.191, 1.996]
    cudnn = [23.915, 20.656, 21.236, 8.249, 1.869, 1.669, 1.626]
    fp16 = [9.104, 7.845, 9.470, 3.784, 1.101, 0.685, 0.545]
    ax = axes[0]
    x = np.arange(len(classes))
    w = 0.26
    for k, (values, color, label, hatch) in enumerate([
        (fp32, PALETTE["red_2"], "FP32 kernels", ""),
        (cudnn, PALETTE["neutral"], "cuDNN 9.10.2", "//"),
        (fp16, PALETTE["blue_main"], "sm70 FP16 tiles (this work)", ""),
    ]):
        bars = ax.bar(x + (k - 1) * w, values, w, color=color, edgecolor="black", linewidth=1.8,
                      hatch=hatch, label=label)
        if k == 2:
            for bar, base in zip(bars, fp32):
                ax.annotate(f"{base / bar.get_height():.1f}x",
                            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points", ha="center",
                            fontsize=12, fontweight="bold", color=PALETTE["blue_main"])
    ax.set_xticks(x, classes, fontsize=13)
    ax.set_ylabel("summed layer time (ms)")
    ax.set_title("a  3x3 trunk layers at batch 32 (speedup vs FP32 above bars)", loc="left",
                 fontsize=15)
    ax.legend(fontsize=13)

    # all 32 3x3 layers against the embedding batch class
    batches = np.array([1, 4, 8, 16, 32])
    total_fp32 = np.array([4.164, 11.689, 23.442, 46.125, 91.949])
    total_cudnn = np.array([5.464, 19.497, 32.686, 40.139, 79.220])
    total_fp16 = np.array([2.091, 4.576, 8.399, 16.469, 32.534])
    ax = axes[1]
    for values, color, label, marker, offset in [
        (total_fp32 / total_fp16, PALETTE["red_strong"], "vs FP32 kernels", "o", 10),
        (total_cudnn / total_fp16, PALETTE["gray_dark"], "vs cuDNN", "s", -20),
    ]:
        ax.plot(batches, values, color=color, linewidth=2.5, marker=marker, markersize=9,
                label=label)
        for b, v in zip(batches, values):
            ax.annotate(f"{v:.2f}", (b, v), xytext=(0, offset), textcoords="offset points",
                        ha="center", fontsize=11, color=color)
    ax.axhline(1.0, color=PALETTE["neutral"], linewidth=2, linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks(batches, [str(b) for b in batches])
    ax.set_ylim(0.5, 4.8)
    ax.set_xlabel("embedding batch")
    ax.set_ylabel("speedup of FP16 tiles (x)")
    ax.set_title("b  All 32 3x3 layers", loc="left", fontsize=15)
    ax.legend(fontsize=13, loc="lower right")
    finalize(fig, "trunk_layers")


def figure_kernels():
    """Where GPU time goes, and the effect of keeping the tap loop rolled"""
    apply_publication_style(font_size=16, axes_linewidth=2.5)
    fig, axes = plt.subplots(1, 2, figsize=(20, 6), gridspec_kw={"width_ratios": [1.3, 1]})

    # nsys kernel time for one 607 s file (grzbb.wav), idle GPU
    ax = axes[0]
    parts = [
        ("3x3 trunk", [2090.1 - 0, 490.6], PALETTE["blue_main"]),
        ("stem + shortcuts", [0, 39.4], PALETTE["blue_secondary"]),
        ("LSTM", [555.3, 314.4], PALETTE["teal"]),
        ("other segmentation", [83.8, 72.9], PALETTE["green_3"]),
        ("fbank + embedding head", [20.7, 23.6], PALETTE["neutral"]),
    ]
    rows = ["M1 (FP32 trunk)", "M2 (FP16 trunk)"]
    left = np.zeros(2)
    for label, values, color in parts:
        values = np.array(values) / 1000
        ax.barh(rows, values, left=left, color=color, edgecolor="black", linewidth=1.5,
                label=label, height=0.55)
        left += values
    for i, total in enumerate(left):
        ax.text(total + 0.04, i, f"{total:.2f} s", va="center", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 3.3)
    ax.set_xlabel("GPU kernel time (s)")
    ax.set_title("a  One 607 s file: kernel time by stage (M1 trunk row includes stem)",
                 loc="left", fontsize=14)
    ax.legend(fontsize=12, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.invert_yaxis()

    # Nsight Compute at batch 2: tap loop unrolled by ptxas against kept rolled
    ax = axes[1]
    kernels = ["C32", "C64", "C128", "C256"]
    unrolled = [178.9, 130.1, 130.7, 194.7]
    rolled = [126.9, 96.2, 114.4, 193.0]
    x = np.arange(len(kernels))
    w = 0.36
    b1 = ax.bar(x - w / 2, unrolled, w, color=PALETTE["red_1"], edgecolor="black", linewidth=2,
                label="taps unrolled by ptxas (spills)")
    b2 = ax.bar(x + w / 2, rolled, w, color=PALETTE["blue_main"], edgecolor="black", linewidth=2,
                label='rolled, .pragma "nounroll" (no spills)')
    annotate(ax, b1, "{:.0f}", 12)
    annotate(ax, b2, "{:.0f}", 12)
    ax.set_xticks(x, kernels)
    ax.set_ylim(0, 250)
    ax.set_ylabel("kernel time (us)")
    ax.set_title("b  ncu, batch 2: keeping the tap loop rolled", loc="left", fontsize=14)
    ax.legend(fontsize=12, loc="upper left")
    finalize(fig, "kernel_time")


if __name__ == "__main__":
    figure_overview()
    figure_tile()
    figure_end_to_end()
    figure_layers()
    figure_kernels()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
