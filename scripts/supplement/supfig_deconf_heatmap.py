#!/usr/bin/env python3
"""
Supplementary Figure — Deconfounded ClinVar heatmap.

AUROC heatmap on CADD-deconfounded v3 ClinVar dataset.
(Demoted from main Figure 1B in the figure overhaul.)

Input:  data/panels/fig1b.feather
Output: figures/supplement/supfig_deconf_heatmap.{png,pdf}
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
OUT_STEM = ROOT / "figures" / "supplement" / "supfig_deconf_heatmap"

apply_theme()


def main():
    strat_df = pl.read_ipc(PANELS / "fig1b.feather")
    df = prepare(strat_df)
    ov = df.filter(pl.col("consequence") == "Overall").row(0, named=True)
    title = f"Deconfounded ClinVar (n={int(ov['n_total']):,})"

    fig, ax = plt.subplots(figsize=(9, 7))
    plot_heatmap(ax, df, "auroc", title)
    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
