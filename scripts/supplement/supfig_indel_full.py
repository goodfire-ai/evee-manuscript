#!/usr/bin/env python3
"""
Supplementary Figure — Full indel benchmark: stratified AUROC heatmap.

All strata including size categories (1 bp, 2-5 bp, 6-20 bp, >20 bp).
Vertical orientation (strata on y-axis, methods on x-axis).

Input:  artifacts/indel_stratified.feather
Output: figures/supplement/supfig_indel_full.{png,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from theme.mayo_theme import (
    apply_theme, save_figure, HEATMAP_CMAP,
    FONT_SIZE_TITLE, FONT_SIZE_TICK, FONT_SIZE_CELL, FONT_SIZE_LABEL,
)

ARTIFACTS = ROOT / "artifacts"
OUT_STEM = ROOT / "figures" / "supplement" / "supfig_indel_full"

apply_theme()

METHOD_COLS = (
    ("evo2_cov_probe_zeroshot",       "Evo2 probe+\n(zero-shot)"),
    ("evo2_mean_probe_supervised",    "Evo2 probe\n(supervised)"),
    ("cadd_v17_indel",                "CADD v1.7 InDel"),
    ("ntv3_subref_probe_supervised",  "NTv3 subref probe\n(supervised)"),
)

# Separator positions (between groups on y-axis)
SEP_POSITIONS = (0.5, 4.5, 8.5)


def main():
    df = pl.read_ipc(ARTIFACTS / "indel_stratified.feather")

    strata = df["stratum"].to_list()
    n_vals = df["n"].to_list()
    pct_vals = df["pct_pathogenic"].to_list()
    n_strata = len(strata)
    n_methods = len(METHOD_COLS)

    # Build matrix [strata x methods]
    mat = np.full((n_strata, n_methods), np.nan)
    for j, (col, _) in enumerate(METHOD_COLS):
        mat[:, j] = df[col].to_numpy().astype(float)

    ylabels = [
        f"{s}\n(n={n:,}, {p:.0f}%P)"
        for s, n, p in zip(strata, n_vals, pct_vals)
    ]
    xlabels = [name for _, name in METHOD_COLS]

    fig, ax = plt.subplots(figsize=(5, 8))

    norm = mcolors.Normalize(vmin=0.5, vmax=1.0)
    im = ax.imshow(mat, cmap=HEATMAP_CMAP, norm=norm, aspect="auto")

    for i in range(n_strata):
        for j in range(n_methods):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "\u2014", ha="center", va="center",
                        fontsize=FONT_SIZE_CELL, color="#aaaaaa")
            else:
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=FONT_SIZE_CELL, fontweight="semibold", color="#222222")

    for sep in SEP_POSITIONS:
        ax.axhline(sep, color="white", linewidth=2.0)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

    ax.set_xticks(range(n_methods))
    ax.set_xticklabels(xlabels, fontsize=FONT_SIZE_TICK, rotation=45, ha="right")
    ax.set_yticks(range(n_strata))
    ax.set_yticklabels(ylabels, fontsize=FONT_SIZE_TICK)

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("AUROC", fontsize=FONT_SIZE_LABEL, fontweight="semibold")
    cbar.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    ax.set_title(
        "ClinVar Indel Pathogenicity \u2014 Full Stratified AUROC\n"
        "N = 73,961 indels, 3,570 genes (\u2264 100 kb)",
        fontsize=FONT_SIZE_TITLE, fontweight="semibold", pad=8,
    )

    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
