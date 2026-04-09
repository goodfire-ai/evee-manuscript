#!/usr/bin/env python3
"""
Figure 1B — Heatmap: CADD-deconfounded v3 ClinVar.

Source: phase13/06_plot.py → plot_consequence_heatmaps() (deconf branch)
Input:  data/panels/fig1b.csv
Output: figures/figure1/panels/fig1b.{png,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from theme.mayo_theme import apply_theme, save_figure

# Reuse heatmap logic from fig1a
sys.path.insert(0, str(ROOT / "scripts" / "figure1"))
from fig1a import _prepare, plot_heatmap

PANELS = ROOT / "data" / "panels"
OUT_STEM = ROOT / "figures" / "figure1" / "panels" / "fig1b"

apply_theme()


def plot(ax):
    """Plot Figure 1B onto given axes."""
    strat_df = pl.read_csv(PANELS / "fig1b.csv")
    df = _prepare(strat_df)
    ov = df.filter(pl.col("consequence") == "Overall").row(0, named=True)
    title = f"Deconfounded ClinVar (n={int(ov['n_total']):,})"
    plot_heatmap(ax, df, "auroc", title)


def main():
    fig, ax = plt.subplots(figsize=(9, 7))
    plot(ax)
    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
