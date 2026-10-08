#!/usr/bin/env python3
"""Two-path schematic: Mahé mechanical packing vs CF-SMC pack()."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "outputs"
NAVY = "#002D4C"
GREEN = "#009682"
GOLD = "#C4A35A"
GREY = "#4D4D4D"
BOX = "#F4F7F8"


def box(ax, x, y, w, h, text, fc=BOX, ec=NAVY, ts=10, color=NAVY, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=fc, edgecolor=ec, linewidth=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=color,
            fontsize=ts, fontweight=weight, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.3,
                                mutation_scale=12))


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7), facecolor="white")
    fig.subplots_adjust(wspace=0.08, left=0.03, right=0.99, top=0.84, bottom=0.08)

    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Mahé 2023  ·  dünne Rundfasern", color=NAVY, fontsize=13, fontweight="bold")
    box(ax, 1.5, 8.1, 7, 1.35, "Fasernetz erzeugen\n(dünne Fasern, Kontakte)", ts=11)
    arrow(ax, 5, 8.1, 5, 7.15)
    box(ax, 1.5, 5.5, 7, 1.55, "Mechanisch zusammendrücken\n(Solver)",
        fc="#E6F4F1", ec=GREEN, ts=11)
    arrow(ax, 5, 5.5, 5, 4.55)
    box(ax, 1.5, 2.7, 7, 1.75, "FVG steigt\nweil das Vlies gepresst wird",
        fc="#E6F4F1", ec=GREEN, ts=11, weight="bold")
    ax.text(5, 1.3, "Für Glasfaser-Netze gebaut,\nnicht für breite C-SMC-Rovings", ha="center",
            color=GREY, fontsize=10)

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Unsere C-SMC-Rollen  ·  nur ablegen", color=NAVY, fontsize=13, fontweight="bold")
    box(ax, 1.5, 8.1, 7, 1.35, "Rovings zufällig ablegen\nund auf den Stapel senken", ts=11)
    arrow(ax, 5, 8.1, 5, 7.15)
    box(ax, 1.5, 5.5, 7, 1.55, "40 % ist nur eine Obergrenze\nStopp, sobald nichts mehr passt",
        fc="#FBF3E0", ec=GOLD, ts=11)
    arrow(ax, 5, 5.5, 5, 4.55)
    box(ax, 1.5, 2.7, 7, 1.75, "Stapel voll  ·  FVG ≈ 27 %\nEs wird nicht gepresst",
        fc="#FBF3E0", ec=GOLD, ts=11, weight="bold")
    ax.text(5, 1.3, "So steht es auch in der Anleitung:\nAblegen stoppt, wenn kein Roving passt", ha="center",
            color=GREY, fontsize=10)

    path = OUT / "fvc_mahe_vs_pack.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
