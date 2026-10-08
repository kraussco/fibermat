#!/usr/bin/env python3
"""Three routes to higher FVG, and why naive squash is invalid."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon

OUT = Path(__file__).resolve().parents[1] / "outputs"
NAVY = "#002D4C"
GREEN = "#009682"
GOLD = "#C4A35A"
RED = "#C0392B"
GREY = "#4D4D4D"
BOX = "#F4F7F8"


def roundbox(ax, x, y, w, h, text, fc=BOX, ec=NAVY, ts=10, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
        facecolor=fc, edgecolor=ec, linewidth=1.3,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=NAVY, fontsize=ts, fontweight=weight)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.5), facecolor="white")
    fig.subplots_adjust(wspace=0.06, left=0.02, right=0.98, top=0.82, bottom=0.10)

    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("1  Mechanisch pressen", color=NAVY, fontsize=13, fontweight="bold")
    ax.add_patch(Rectangle((2.2, 1.2), 5.6, 7.2, fill=False, lw=1.4, edgecolor=NAVY))
    ax.add_patch(Rectangle((2.5, 1.5), 5.0, 0.7, facecolor="#3B6B9A", edgecolor=NAVY, lw=0.5))
    ax.add_patch(Rectangle((2.7, 2.3), 4.6, 0.7, facecolor="#2A9D8F", edgecolor=NAVY, lw=0.5, angle=4))
    ax.add_patch(Rectangle((2.5, 3.2), 5.0, 0.7, facecolor="#E9C46A", edgecolor=NAVY, lw=0.5))
    ax.add_patch(Rectangle((2.6, 4.1), 4.8, 0.7, facecolor="#264653", edgecolor=NAVY, lw=0.5, angle=-3))
    ax.annotate("", xy=(8.6, 2.0), xytext=(8.6, 8.0),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=2))
    ax.text(5, 0.45, "Fasern biegen sich in die Lücken.\nKeine Durchdringung. FVG steigt.",
            ha="center", color=GREY, fontsize=9)

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("2  Nur in z stauchen", color=NAVY, fontsize=13, fontweight="bold")
    ax.add_patch(Rectangle((2.2, 2.6), 5.6, 4.4, fill=False, lw=1.4, edgecolor=NAVY))
    ax.add_patch(Rectangle((2.5, 2.9), 5.0, 0.9, facecolor="#3B6B9A", edgecolor=NAVY, lw=0.5, alpha=0.85))
    ax.add_patch(Rectangle((2.7, 3.4), 4.6, 0.9, facecolor="#2A9D8F", edgecolor=NAVY, lw=0.5, alpha=0.7, angle=8))
    ax.add_patch(Rectangle((2.5, 4.0), 5.0, 0.9, facecolor="#E9C46A", edgecolor=NAVY, lw=0.5, alpha=0.7))
    ax.text(5.0, 5.5, "Überlappung", color=RED, fontsize=12, fontweight="bold", ha="center")
    ax.text(5, 0.45, "Zahlen-FVG wird 40 %.\nFasern durchdringen sich.\nFür DBS unbrauchbar.",
            ha="center", color=GREY, fontsize=9)

    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("3  Rovings ausrichten", color=NAVY, fontsize=13, fontweight="bold")
    ax.add_patch(Rectangle((2.2, 1.2), 5.6, 7.2, fill=False, lw=1.4, edgecolor=NAVY))
    for i, c in enumerate(["#3B6B9A", "#2A9D8F", "#E9C46A", "#264653", "#3B6B9A"]):
        ax.add_patch(Rectangle((2.45, 1.5 + i * 1.35), 5.1, 1.15, facecolor=c,
                               edgecolor=NAVY, lw=0.5))
    ax.text(5, 0.45, "Lagen ohne Kreuzungen.\nDann geht 40 % schon beim Ablegen.\nAber das ist kein zufälliges C-SMC.",
            ha="center", color=GREY, fontsize=9)

    path = OUT / "fvc_compact_options.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
