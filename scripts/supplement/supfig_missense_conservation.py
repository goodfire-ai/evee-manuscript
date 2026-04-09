#!/usr/bin/env python3
"""
Supplementary Figure — Conservation tiers, missense variants only.

AUROC by phyloP100way conservation tier, filtered to missense.
(Demoted from main Figure 1D in the figure overhaul.)

Input:  data/panels/fig1d.csv
Output: figures/supplement/supfig_missense_conservation.{png,pdf}
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

sys.path.insert(0, str(ROOT / "scripts" / "figure1"))
from fig1f import plot_lineplot, CONSERVATION_TIERS

PANELS = ROOT / "data" / "panels"
OUT_STEM = ROOT / "figures" / "supplement" / "supfig_missense_conservation"

apply_theme()


def main():
    df = pl.read_csv(PANELS / "fig1d.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_lineplot(ax, df, "auroc", "Missense Only", CONSERVATION_TIERS,
                  xlabel="Conservation Tier (phyloP100way)")
    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
