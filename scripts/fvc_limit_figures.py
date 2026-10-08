#!/usr/bin/env python3
"""Schematics: why random-orientation CF-SMC packing jams near FVC 0.27."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)

NAVY = "#002D4C"
GREEN = "#009682"
GREY = "#4D4D4D"
VOID = "#D9E8E4"
TOW_A = "#3B6B9A"
TOW_B = "#2A9D8F"
TOW_C = "#E9C46A"
TOW_D = "#264653"


def _style(ax):
    ax.set_aspect("equal")
    ax.axis("off")
    for sp in ax.spines.values():
        sp.set_visible(False)


def _tow(ax, xy, w, h, angle=0, fc=TOW_A, ec=NAVY, alpha=0.92, lw=0.8):
    r = Rectangle(xy, w, h, angle=angle, facecolor=fc, edgecolor=ec,
                  linewidth=lw, alpha=alpha, zorder=2)
    ax.add_patch(r)
    return r


def side_view(path):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), facecolor="white")
    fig.subplots_adjust(wspace=0.08, left=0.03, right=0.99, top=0.82, bottom=0.08)

    # --- aligned ---
    ax = axes[0]
    _style(ax)
    ax.set_xlim(-0.4, 10.4)
    ax.set_ylim(-0.6, 4.4)
    ax.add_patch(Rectangle((0, 0), 10, 3.6, fill=False, lw=1.4, edgecolor=NAVY, zorder=3))
    colours = [TOW_A, TOW_B, TOW_C, TOW_A, TOW_B]
    for i, c in enumerate(colours):
        y = 0.08 + i * 0.68
        ax.add_patch(Rectangle((0.15, y), 9.7, 0.58, facecolor=c, edgecolor=NAVY,
                               lw=0.6, alpha=0.95, zorder=2))
    ax.annotate("", xy=(10.55, 3.6), xytext=(10.55, 0.0),
                arrowprops=dict(arrowstyle="<->", color=NAVY, lw=1.2),
                annotation_clip=False)
    ax.text(10.7, 1.8, "H", color=NAVY, fontsize=11, va="center")
    ax.set_title("Gerade Rovings  ·  sie liegen in Lagen", color=NAVY, fontsize=13, fontweight="bold", pad=8)
    ax.text(5.0, -0.45, "So lässt sich 40 % FVG erreichen",
            ha="center", va="top", color=GREY, fontsize=10)

    # --- random with bridging ---
    ax = axes[1]
    _style(ax)
    ax.set_xlim(-0.4, 10.4)
    ax.set_ylim(-0.6, 4.4)
    ax.add_patch(Rectangle((0, 0), 10, 3.6, fill=False, lw=1.4, edgecolor=NAVY, zorder=5))
    # voids
    ax.add_patch(Polygon([(1.7, 0.15), (4.4, 0.15), (4.4, 1.85), (3.2, 2.35), (1.7, 1.4)],
                         facecolor=VOID, edgecolor="none", zorder=1))
    ax.add_patch(Polygon([(6.0, 0.15), (8.6, 0.15), (8.6, 1.55), (7.1, 2.15), (6.0, 1.2)],
                         facecolor=VOID, edgecolor="none", zorder=1))
    # stacked crossings (peaks)
    ax.add_patch(Rectangle((0.25, 0.15), 3.3, 0.50, angle=7, facecolor=TOW_A, edgecolor=NAVY, lw=0.6, zorder=2))
    ax.add_patch(Rectangle((5.5, 0.15), 3.4, 0.50, angle=-5, facecolor=TOW_B, edgecolor=NAVY, lw=0.6, zorder=2))
    ax.add_patch(Rectangle((2.7, 0.58), 3.3, 0.50, angle=16, facecolor=TOW_C, edgecolor=NAVY, lw=0.6, zorder=2))
    ax.add_patch(Rectangle((0.35, 1.05), 3.0, 0.50, angle=-11, facecolor=TOW_D, edgecolor=NAVY, lw=0.6, zorder=2))
    ax.add_patch(Rectangle((6.2, 0.95), 3.2, 0.50, angle=12, facecolor=TOW_A, edgecolor=NAVY, lw=0.6, zorder=2))
    ax.add_patch(Rectangle((0.35, 2.52), 9.3, 0.55, facecolor=TOW_C, edgecolor=NAVY, lw=0.9, zorder=4))
    ax.text(5.0, 3.22, "Roving liegt auf den Kreuzungen", ha="center", fontsize=9, color=NAVY)
    ax.text(3.05, 0.85, "Lücke", color=GREEN, fontsize=10, fontweight="bold", ha="center")
    ax.text(7.3, 0.7, "Lücke", color=GREEN, fontsize=10, fontweight="bold", ha="center")
    ax.set_title("Zufällige Lage  ·  Kreuzungen und Lücken", color=NAVY, fontsize=13,
                 fontweight="bold", pad=8)
    ax.text(5.0, -0.45, "Ein langer Roving deckt die Lücken darunter zu",
            ha="center", va="top", color=GREY, fontsize=10)

    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def top_view(path):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), facecolor="white")
    fig.subplots_adjust(wspace=0.06, left=0.03, right=0.99, top=0.84, bottom=0.10)
    rng = np.random.default_rng(4)

    ax = axes[0]
    _style(ax)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.add_patch(Rectangle((0.2, 0.2), 11.6, 7.6, fill=False, lw=1.4, edgecolor=NAVY))
    colours = [TOW_A, TOW_B, TOW_C, TOW_D]
    y = 0.45
    k = 0
    while y < 7.4:
        x = 0.45
        while x < 11.4:
            w = 3.4
            h = 0.95
            if x + w > 11.55:
                break
            ax.add_patch(Rectangle((x, y), w, h, facecolor=colours[k % 4],
                                   edgecolor=NAVY, lw=0.5, alpha=0.95))
            x += w + 0.08
            k += 1
        y += 1.05
    ax.set_title("Gerade ausgerichtet  ·  Rovings füllen die Fläche", color=NAVY, fontsize=13, fontweight="bold")
    ax.text(6.0, -0.15, "Neue Rovings finden noch Platz", ha="center",
            color=GREY, fontsize=10)

    ax = axes[1]
    _style(ax)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.add_patch(Rectangle((0.2, 0.2), 11.6, 7.6, fill=False, lw=1.4, edgecolor=NAVY))
    poses = [
        (0.5, 1.1, 28, TOW_A), (3.4, 0.6, -18, TOW_B), (7.2, 1.0, 12, TOW_C),
        (1.0, 3.2, -35, TOW_D), (4.8, 2.6, 42, TOW_A), (8.0, 3.4, -22, TOW_B),
        (0.4, 5.4, 15, TOW_C), (3.8, 5.0, -40, TOW_D), (7.4, 5.6, 33, TOW_A),
        (5.6, 4.0, 8, TOW_B), (2.2, 4.6, 55, TOW_C),
    ]
    for x, y, ang, c in poses:
        ax.add_patch(Rectangle((x, y), 3.6, 0.95, angle=ang, facecolor=c,
                               edgecolor=NAVY, lw=0.5, alpha=0.9))
    # leftover pocket too small
    ax.add_patch(Polygon([(9.6, 2.2), (11.3, 2.5), (11.2, 4.3), (9.85, 3.9)],
                         facecolor=VOID, edgecolor=GREEN, lw=1.2, linestyle="--"))
    ax.annotate("Lücke kleiner\nals ein Roving", xy=(10.5, 3.2), xytext=(9.3, 0.55),
                fontsize=9, color=NAVY, ha="center",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.1))
    ax.set_title("Zufällige Lage  ·  Restlücken zu klein", color=NAVY, fontsize=13, fontweight="bold")
    ax.text(6.0, -0.15, "Ein neuer Roving passt nicht mehr hinein",
            ha="center", color=GREY, fontsize=10)

    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    del rng


def compaction_bar(path):
    fig, ax = plt.subplots(figsize=(12.4, 4.4), facecolor="white")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.22)
    labels = [
        "Aligned layers\n(same tows, nested)",
        "This work\nrandom, no compaction",
        "Target 0.40\nneeds compaction\nor alignment",
    ]
    values = [0.70, 0.27, 0.40]
    colors = [TOW_A, GREEN, TOW_C]
    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor=NAVY, linewidth=0.8, zorder=2)
    ax.axhline(0.40, color=NAVY, ls="--", lw=1.0, zorder=1)
    ax.text(2.38, 0.415, "0.40 cap", color=NAVY, fontsize=10, va="bottom")
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Fibre volume content", color=NAVY, fontsize=12)
    ax.set_yticks([0, 0.27, 0.40, 0.70])
    ax.set_yticklabels(["0", "0.27", "0.40", "~0.70"], color=NAVY)
    ax.tick_params(colors=NAVY, length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(NAVY)
    ax.spines["bottom"].set_color(NAVY)
    ax.set_facecolor("white")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.2f}",
                ha="center", color=NAVY, fontsize=13, fontweight="bold")
    ax.set_title("Random sedimentation jams well below the 0.40 cap",
                 color=NAVY, fontsize=14, fontweight="bold")
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    side_view(OUT / "fvc_side_aligned_vs_random.png")
    top_view(OUT / "fvc_top_aligned_vs_random.png")
    compaction_bar(OUT / "fvc_bar_comparison.png")
    print("wrote figures")
