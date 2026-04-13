#!/usr/bin/env python3
"""
Supplementary — DMS AUROC barplot, 2x2 faceted by gene.

Same data and grouping as DMS Spearman but showing AUROC instead of Spearman |ρ|.

Source: dms/01_dms_benchmark.py (benchmark_results.csv)
Input:  artifacts/dms_benchmark.feather
Output: figures/supplement/supfig_dms_auroc.{png,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from theme.mayo_theme import apply_theme, save_figure

# Reuse shared barplot from fig_dms_spearman
sys.path.insert(0, str(ROOT / "scripts" / "figure1"))
from fig_dms_spearman import _load_and_filter, plot_dms_barplot, ARTIFACTS
OUT_STEM = ROOT / "figures" / "supplement" / "supfig_dms_auroc"

apply_theme()


def main():
    df = _load_and_filter(ARTIFACTS / "dms_benchmark.feather")

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharey=True)
    plot_dms_barplot(axes.flat, df, metric="auroc",
                     ylabel="AUROC", ylim=(0.5, 1.0))
    fig.tight_layout(h_pad=2.0, w_pad=1.5)
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
