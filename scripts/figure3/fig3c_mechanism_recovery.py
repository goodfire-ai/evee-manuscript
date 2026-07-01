#!/usr/bin/env python3
"""
Figure 3c — Mechanism-class recovery across allelic-heterogeneity genes.

Grouped bar chart of 5-fold supervised-CV balanced accuracy showing that EVEE
features recover documented allelic-series mechanism classes across four genes
(LDLR, LMNA, MYH7, TP53), versus baselines (random, position-only, CADD,
AlphaMissense) and an EVEE + position combination.

Input:  artifacts/mechanism_recovery.json
Output: figures/figure3/fig3c_mechanism_recovery.{png,pdf}
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from theme.theme import (
    apply_theme, save_figure, COLORS,
    FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_LEGEND,
    FIG_WIDTH_DOUBLE,
)

ARTIFACTS = ROOT / "artifacts"
OUT_STEM = ROOT / "figures" / "figure3" / "fig3c_mechanism_recovery"

apply_theme()

GENES = ["LDLR", "LMNA", "MYH7", "TP53"]

# Ordered baseline → our-method flow. Colours locked to the paper-wide method
# palette: CADD = steel, AlphaMissense = sage, EVEE = gf_orange; EVEE + position
# uses lavender to stay distinct from CADD's steel.
METHODS = [
    ("Random",          COLORS["light_gray"]),
    ("Position only",   COLORS["gray"]),
    ("CADD",            COLORS["steel"]),
    ("AlphaMissense",   COLORS["sage"]),
    ("EVEE",            COLORS["gf_orange"]),
    ("EVEE + position", COLORS["lavender"]),
]


def load_data():
    stats = json.loads((ARTIFACTS / "mechanism_recovery.json").read_text())
    genes = stats["genes"]
    return stats, genes


def plot(ax, stats, genes):
    n_methods = len(METHODS)
    group_width = 0.84
    bar_width = group_width / n_methods
    x = np.arange(len(GENES))

    for i, (label, color) in enumerate(METHODS):
        vals = np.array([genes[g]["metrics"].get(label, np.nan) for g in GENES],
                        dtype=float)
        offset = (i - n_methods / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width,
                      color=color, label=label,
                      edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012,
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=FONT_SIZE_TICK - 1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(GENES, fontsize=FONT_SIZE_TICK + 1, fontweight="semibold")
    ax.tick_params(axis="x", pad=3)
    for xi, g in zip(x, GENES):
        n = genes[g]["n"]
        k = genes[g]["n_classes"]
        ax.text(xi, -0.060, f"N = {int(n)},  k = {int(k)}",
                ha="center", va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=FONT_SIZE_TICK)

    ax.set_ylabel("Balanced accuracy  (5-fold CV)",
                  fontsize=FONT_SIZE_LABEL + 1, fontweight="semibold")
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK + 1.5)
    ax.grid(True, axis="y", lw=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=FONT_SIZE_LEGEND + 1.5, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, 1.12),
              ncol=n_methods, columnspacing=1.2, handletextpad=0.5,
              handlelength=1.4)


def main():
    stats, genes = load_data()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    plot(ax, stats, genes)
    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
