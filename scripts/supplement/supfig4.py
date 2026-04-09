#!/usr/bin/env python3
"""
Supplementary Figure 4 — Indel benchmark (star >= 1): stratified AUROC heatmap.

74K ClinVar indels (stars >= 1, genes <= 100 kb), 3,570 genes.
Methods: Evo2 cov probe (zero-shot from SNVs), Evo2 mean probe (supervised on
indels), CADD v1.7 InDel, NTv3 subref probe (supervised on indels).
Strata match Phase 6 layout: Overall | Consequence | Indel size | Direction.

Source: notebooks/2026-03-12-20-35_phase6v2_results_continued_continued.ipynb
Input:  data/supplement/supfig4/indel_stratified_auroc.feather
Output: figures/supplement/supfig4.{png,pdf}
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

PANELS = ROOT / "artifacts"
OUT_STEM = ROOT / "figures" / "supplement" / "supfig4"

apply_theme()

# Method columns → display names (y-axis)
METHOD_COLS = (
    ("evo2_cov_probe_zeroshot",       "Evo2 probe+\n(zero-shot)"),
    ("evo2_mean_probe_supervised",    "Evo2 probe\n(supervised)"),
    ("cadd_v17_indel",                "CADD v1.7 InDel"),
    ("ntv3_subref_probe_supervised",  "NTv3 subref probe\n(supervised)"),
)

# Separator positions (between groups on x-axis)
SEP_POSITIONS = (0.5, 4.5, 8.5)


def main():
    df = pl.read_ipc(PANELS / "supfig4.feather")

    strata = df["stratum"].to_list()
    n_vals = df["n"].to_list()
    pct_vals = df["pct_pathogenic"].to_list()
    n_strata = len(strata)
    n_methods = len(METHOD_COLS)

    # Build matrix [methods x strata]
    mat = np.full((n_methods, n_strata), np.nan)
    for i, (col, _) in enumerate(METHOD_COLS):
        mat[i, :] = df[col].to_numpy().astype(float)

    xlabels = [
        f"{s}\n(n={n:,})\n({p:.0f}%P)"
        for s, n, p in zip(strata, n_vals, pct_vals)
    ]
    ylabels = [name for _, name in METHOD_COLS]

    fig, ax = plt.subplots(figsize=(12, 3.5))

    norm = mcolors.Normalize(vmin=0.5, vmax=1.0)
    im = ax.imshow(mat, cmap=HEATMAP_CMAP, norm=norm, aspect="auto")

    # Annotate cells
    for i in range(n_methods):
        for j in range(n_strata):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "\u2014", ha="center", va="center",
                        fontsize=FONT_SIZE_CELL, color="#aaaaaa")
            else:
                text_color = "#222222"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=FONT_SIZE_CELL, fontweight="semibold", color=text_color)

    # Group separators
    for sep in SEP_POSITIONS:
        ax.axvline(sep, color="white", linewidth=2.0)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

    ax.set_xticks(range(n_strata))
    ax.set_xticklabels(xlabels, fontsize=FONT_SIZE_TICK)
    ax.set_yticks(range(n_methods))
    ax.set_yticklabels(ylabels, fontsize=FONT_SIZE_TICK)

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("AUROC", fontsize=FONT_SIZE_LABEL, fontweight="semibold")
    cbar.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    ax.set_title(
        "ClinVar Indel Pathogenicity \u2014 Stratified AUROC\n"
        "N = 73,961 indels, 3,570 genes (\u2264 100 kb)",
        fontsize=FONT_SIZE_TITLE, fontweight="semibold", pad=8,
    )

    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
