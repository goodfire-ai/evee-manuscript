#!/usr/bin/env python3
"""
Supplementary Figure — Top-K position selection sweep.

AUROC vs number of divergent positions (K) used for pathogenicity classification.

Input:  artifacts/topk_sweep_auc.csv
Output: figures/supplement/panels/supfig_topk_sweep.{png,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from theme.mayo_theme import apply_theme, save_figure, add_panel_label, COLORS

ARTIFACTS = ROOT / "artifacts"
OUT_STEM = ROOT / "figures" / "supplement" / "panels" / "supfig_topk_sweep"

apply_theme()


def main():
    df = pl.read_csv(ARTIFACTS / "topk_sweep_auc.csv")

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(df["k"], df["auc"], marker="o", markersize=6, linewidth=2, color=COLORS["gf_orange"])

    ax.set_xlabel("Number of divergent positions (K)")
    ax.set_ylabel("AUROC")
    ax.set_xticks(df["k"].to_list())
    ax.set_ylim(0.90, 0.95)

    fig.tight_layout()
    OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
