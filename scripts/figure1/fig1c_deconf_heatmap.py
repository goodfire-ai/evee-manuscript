#!/usr/bin/env python3
"""
Figure 1c — Deconfounded ClinVar heatmap.

AUROC heatmap on CADD-deconfounded v3 ClinVar dataset.

Input:  artifacts/deconf_benchmark.feather
Output: figures/figure1/fig1c_deconf_heatmap.{png,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from theme.theme import apply_theme, save_figure

sys.path.insert(0, str(ROOT / "scripts" / "figure1"))
from _heatmap_common import prepare, plot_heatmap

ARTIFACTS = ROOT / "artifacts"
OUT_STEM = ROOT / "figures" / "figure1" / "fig1c_deconf_heatmap"

apply_theme()


def main():
    strat_df = pl.read_ipc(ARTIFACTS / "deconf_benchmark.feather")
    df = prepare(strat_df)

    fig, ax = plt.subplots(figsize=(9, 7))
    plot_heatmap(ax, df, "auroc")
    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
