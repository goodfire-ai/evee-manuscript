#!/usr/bin/env python3
"""
Supplementary Figure 6 — DMS AUROC barplot, 2x2 faceted by gene.

Same data and grouping as Figure 1F but showing AUROC instead of Spearman |ρ|.

Source: dms/01_dms_benchmark.py (benchmark_results.csv)
Input:  data/figure1/fig1f/benchmark_results.feather
Output: figures/supplement/supfig6.{png,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from theme.mayo_theme import apply_theme, save_figure

# Reuse shared barplot from fig1f
sys.path.insert(0, str(ROOT / "scripts" / "figure1"))
from fig1f import _load_and_filter, plot_dms_barplot, PANELS
OUT_STEM = ROOT / "figures" / "supplement" / "supfig6"

apply_theme()


def main():
    df = _load_and_filter(PANELS / "fig1f.feather")

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharey=True)
    plot_dms_barplot(axes.flat, df, metric="auroc",
                     ylabel="AUROC", ylim=(0.5, 1.0))
    fig.tight_layout(h_pad=2.0, w_pad=1.5)
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
