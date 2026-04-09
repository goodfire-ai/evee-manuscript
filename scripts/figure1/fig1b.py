#!/usr/bin/env python3
"""
Figure 1B — Heatmap: ClinVar pathogenicity prediction across SNV consequence types.

AUROC heatmap for Evo2 covariance probe, Evo2 mean probe, Evo2 loss,
CADD v1.7, AlphaMissense, GPN-MSA, NTv3, and AlphaGenome
on 833,970 variants from genes <= 100 kb with >= 1 star review status.

Input:  data/panels/fig1a.csv
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

sys.path.insert(0, str(ROOT / "scripts" / "figure1"))
from _heatmap_common import prepare, plot_heatmap

PANELS = ROOT / "artifacts"
OUT_STEM = ROOT / "figures" / "figure1" / "panels" / "fig1b"

apply_theme()


def plot(ax, aspect="equal"):
    """Plot Figure 1B onto given axes."""
    strat_df = pl.read_csv(PANELS / "fig1a.csv")
    df = prepare(strat_df)
    plot_heatmap(ax, df, "auroc", aspect=aspect)


# ── Shared cell sizing (must match fig1c) ─────────────────────────
CELL_SIZE = 0.55          # inches per cell edge
H_PAD     = 2.0           # extra height for x-tick labels + margins
W_PAD     = 2.5           # extra width  for y-tick labels + margins
N_ROWS    = 8             # consequence types
N_COLS    = 8             # methods (may shrink if some missing)


def main():
    fig_h = N_ROWS * CELL_SIZE + H_PAD
    fig_w = N_COLS * CELL_SIZE + W_PAD
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plot(ax)
    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
