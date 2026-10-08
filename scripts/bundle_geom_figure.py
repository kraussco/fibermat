#!/usr/bin/env python3
"""Bundle width/height, subbundle count, and why FVC stays low."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parents[1] / "outputs"
NAVY = "#002D4C"
GREEN = "#009682"
GOLD = "#C4A35A"
RED = "#C0392B"
GREY = "#4D4D4D"
TOW = "#5B8FB9"
FIB = "#2A9D8F"


def arrow(ax, x1, y1, x2, y2, color=NAVY, lw=1.4):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="<->", color=color, lw=lw, mutation_scale=10),
    )


def main():
    fig = plt.figure(figsize=(13.2, 5.15), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.05, 1.15], wspace=0.22,
                          left=0.03, right=0.99, top=0.88, bottom=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    for ax in (ax1, ax2, ax3):
        ax.set_aspect("equal")
        ax.axis("off")

    # --- 1: one bundle, labeled ---
    ax1.set_xlim(-0.6, 8.4)
    ax1.set_ylim(-1.5, 5.2)
    ax1.set_title("Ein Bündel (Roving)", color=NAVY, fontsize=13, fontweight="bold", pad=6)
    # top face (width x length, shortened)
    ax1.add_patch(Rectangle((0.8, 1.7), 6.2, 1.35, facecolor=TOW, edgecolor=NAVY, lw=1.3, zorder=2))
    ax1.text(3.9, 2.38, "Breite  b = 4 mm\n(gemessen, klein)", ha="center", va="center",
             color="white", fontsize=10, fontweight="bold", zorder=3)
    arrow(ax1, 0.8, 1.48, 7.0, 1.48)
    ax1.text(3.9, 1.12, "Länge  L = 25,5 mm", ha="center", color=NAVY, fontsize=10)
    # thickness bracket on the left
    ax1.add_patch(Rectangle((0.8, 1.35), 6.2, 0.35, facecolor="#3B6B9A", edgecolor=NAVY, lw=1.0, zorder=1))
    ax1.annotate("", xy=(0.55, 1.35), xytext=(0.55, 3.05),
                 arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.5, mutation_scale=10))
    ax1.text(0.18, 2.2, "h = 0,103 mm", ha="center", va="center", color=GREEN,
             fontsize=9, fontweight="bold", rotation=90)
    ax1.text(3.9, -0.55, "h ist die Dicke des Bündels\nund zugleich der Faserdurchmesser",
             ha="center", color=GREY, fontsize=9)

    # --- 2: cross-section -> subbundles ---
    ax2.set_xlim(-0.4, 8.6)
    ax2.set_ylim(-1.7, 5.4)
    ax2.set_title("Daraus folgt die Subbundle-Zahl", color=NAVY, fontsize=13, fontweight="bold", pad=6)
    b, h = 7.4, 0.95
    x0, y0 = 0.6, 2.15
    ax2.add_patch(Rectangle((x0, y0), b, h, fill=False, edgecolor=NAVY, lw=1.6, zorder=3))
    n_draw = 12
    d = b / 14.0
    for i in range(n_draw):
        cx = x0 + 0.45 + i * (d * 1.05)
        ax2.add_patch(Circle((cx, y0 + h / 2), d * 0.48, facecolor=FIB,
                             edgecolor=NAVY, lw=0.5, zorder=2))
    ax2.text(x0 + b + 0.15, y0 + h / 2, "… 38", ha="left", va="center",
             color=NAVY, fontsize=11, fontweight="bold")
    arrow(ax2, x0, y0 - 0.28, x0 + b, y0 - 0.28, color=NAVY)
    ax2.text(x0 + b / 2, y0 - 0.62, "b = 4 mm", ha="center", color=NAVY, fontsize=10)
    ax2.annotate("", xy=(x0 - 0.22, y0), xytext=(x0 - 0.22, y0 + h),
                 arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.4, mutation_scale=9))
    ax2.text(x0 - 0.38, y0 + h / 2, "h", ha="right", va="center", color=GREEN,
             fontsize=12, fontweight="bold")
    ax2.text(4.1, 0.55, r"$n_\mathrm{sub} = \mathrm{floor}(b / h)$",
             ha="center", color=NAVY, fontsize=12, fontweight="bold")
    ax2.text(4.1, -0.15, "floor(4,0 / 0,103) = 38  runde Fasern",
             ha="center", color=GREY, fontsize=10)
    ax2.text(4.1, 4.55, "Vorgabe n  →  Höhe  h = b / n",
             ha="center", color=GREEN, fontsize=11, fontweight="bold")
    ax2.text(4.1, 3.95, "z. B. 15 Subbundles  →  h = 4/15 ≈ 0,27 mm",
             ha="center", color=GREY, fontsize=9)

    # --- 3: why FVC smaller ---
    ax3.set_xlim(-0.5, 8.8)
    ax3.set_ylim(-1.6, 5.5)
    ax3.set_title("Warum das FVG klein bleibt", color=NAVY, fontsize=13, fontweight="bold", pad=6)
    # two crossing ribbons in side view
    ax3.add_patch(Rectangle((0.7, 0.55), 6.6, 0.85, angle=9, facecolor=TOW,
                            edgecolor=NAVY, lw=0.9, zorder=2))
    ax3.add_patch(Rectangle((0.9, 1.55), 6.4, 0.85, angle=-11, facecolor=FIB,
                            edgecolor=NAVY, lw=0.9, zorder=2))
    ax3.add_patch(Rectangle((0.7, 2.65), 6.6, 0.85, angle=8, facecolor=GOLD,
                            edgecolor=NAVY, lw=0.9, zorder=2))
    ax3.plot([0.4, 7.6], [4.15, 4.15], color=RED, lw=2.2, zorder=4)
    ax3.text(7.75, 4.15, "Deckel", color=RED, fontsize=10, va="center")
    ax3.annotate("", xy=(0.35, 0.55), xytext=(0.35, 4.15),
                 arrowprops=dict(arrowstyle="<->", color=NAVY, lw=1.3, mutation_scale=9))
    ax3.text(0.05, 2.35, "H", ha="right", va="center", color=NAVY, fontsize=12, fontweight="bold")
    ax3.text(4.2, -0.55, "Jede Kreuzung kostet die volle Höhe h\nüber die ganze Breite b = 4 mm.\nH / h Lagen – dann ist der Deckel voll,\ndie Fläche aber voller Lücken.",
             ha="center", va="top", color=GREY, fontsize=9)

    path = OUT / "bundle_width_height.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
