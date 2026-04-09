#!/usr/bin/env python3
"""
Figure 1D — Line plot: performance vs conservation tier (missense only).

Source: phase13/09_conservation_tiers.py (adapted for missense filter)
Input:  data/panels/fig1d.csv
Output: figures/figure1/panels/fig1d.{png,pdf}
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

# Reuse lineplot from fig1c
sys.path.insert(0, str(ROOT / "scripts" / "figure1"))
from fig1c import plot_lineplot, CONSERVATION_TIERS

PANELS = ROOT / "data" / "panels"
OUT_STEM = ROOT / "figures" / "figure1" / "panels" / "fig1d"

apply_theme()


def plot(ax):
    """Plot Figure 1D onto given axes."""
    df = pl.read_csv(PANELS / "fig1d.csv")
    plot_lineplot(ax, df, "auroc", "", CONSERVATION_TIERS,
                  xlabel="Conservation Tier (phyloP100way)")


def main():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot(ax)
    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
