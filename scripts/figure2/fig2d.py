#!/usr/bin/env python3
"""
Figure 2D — Placeholder for auto-interpretation example screenshot (external).

The actual screenshot is produced outside this pipeline.

Output: figures/figure2/panels/fig2d.{png,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from theme.mayo_theme import apply_theme, save_figure, FONT_SIZE_TITLE

OUT_STEM = ROOT / "figures" / "figure2" / "panels" / "fig2d"

apply_theme()


def plot(ax):
    """Draw a placeholder box for the auto-interpretation screenshot."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96,
                                fill=True, facecolor="#F0F0F0",
                                edgecolor="#CCCCCC", linewidth=1.0))
    ax.text(0.5, 0.5,
            "Auto-interpretation example\n(external screenshot)",
            ha="center", va="center", fontsize=FONT_SIZE_TITLE,
            color="#888888", style="italic")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    fig, ax = plt.subplots(figsize=(8, 5))
    plot(ax)
    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
