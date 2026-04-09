#!/usr/bin/env python3
"""
Figure 1C — Zero-shot generalization to indels: stratified AUROC heatmap.

AUROC for Evo2 covariance probe (zero-shot, trained on SNVs only),
Evo2 mean probe (supervised), CADD v1.7 InDel, and NTv3 subref probe
on 73,961 ClinVar indels stratified by consequence type and direction.
(Size categories omitted; full version in supplement.)

Input:  data/panels/supfig4.feather
Output: figures/figure1/panels/fig1c.{png,pdf}
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
    FONT_SIZE_TICK, FONT_SIZE_CELL,
)

PANELS = ROOT / "artifacts"
OUT_STEM = ROOT / "figures" / "figure1" / "panels" / "fig1c"

apply_theme()

# Method columns -> display names
METHOD_COLS = (
    ("evo2_cov_probe_zeroshot",       "Evo2 Covariance"),
    ("evo2_mean_probe_supervised",    "Evo2 Mean"),
    ("cadd_v17_indel",                "CADD v1.7"),
    ("ntv3_subref_probe_supervised",  "NTv3"),
)

# Strata to keep (exclude size categories: 1 bp, 2-5 bp, 6-20 bp, >20 bp)
KEEP_STRATA = ("Overall", "Frameshift", "In-frame", "Noncoding", "Splice-adj.",
               "Insertion", "Deletion")

# Separator positions (between groups on y-axis)
SEP_POSITIONS = (0.5, 4.5)


def plot(ax, aspect="equal"):
    """Plot indel benchmark heatmap onto given axes (strata=y, methods=x)."""
    df = pl.read_ipc(PANELS / "supfig4.feather")
    df = df.filter(pl.col("stratum").is_in(list(KEEP_STRATA)))

    strata = df["stratum"].to_list()
    n_vals = df["n"].to_list()
    pct_vals = df["pct_pathogenic"].to_list()
    n_strata = len(strata)
    n_methods = len(METHOD_COLS)

    # Build matrix [strata x methods] (strata on y, methods on x)
    mat = np.full((n_strata, n_methods), np.nan)
    for j, (col, _) in enumerate(METHOD_COLS):
        mat[:, j] = df[col].to_numpy().astype(float)

    xlabels = [name for _, name in METHOD_COLS]

    norm = mcolors.Normalize(vmin=0.5, vmax=1.0)
    im = ax.imshow(mat, cmap=HEATMAP_CMAP, norm=norm, aspect=aspect)

    # Vertical separators between method groups: Evo2 | CADD | NTv3
    ax.axvline(1.5, color="white", linewidth=2.5)   # after Evo2 Mean
    ax.axvline(2.5, color="white", linewidth=2.5)   # after CADD v1.7

    # Annotate cells
    for i in range(n_strata):
        for j in range(n_methods):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "\u2014", ha="center", va="center",
                        fontsize=FONT_SIZE_CELL, color="#aaaaaa")
            else:
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=FONT_SIZE_CELL, fontweight="semibold", color="#222222")

    # Group separators (horizontal lines between strata groups)
    for sep in SEP_POSITIONS:
        ax.axhline(sep, color="white", linewidth=2.5)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

    ax.set_xticks(range(n_methods))
    ax.set_xticklabels(xlabels, fontsize=FONT_SIZE_TICK, fontweight="semibold",
                       rotation=45, ha="right")
    ax.set_yticks(range(n_strata))
    ax.set_yticklabels([])  # Clear default labels
    # Manual two-part labels matching fig1b style: bold name, regular stats
    for i, (s, n, p) in enumerate(zip(strata, n_vals, pct_vals)):
        ax.text(-0.04, i - 0.01, s, transform=ax.get_yaxis_transform(),
                ha="right", va="center",
                fontsize=FONT_SIZE_TICK, fontweight="semibold")
        stat = f"(n={n:,}, {p:.1f}%P)"
        ax.text(-0.04, i + 0.21, stat, transform=ax.get_yaxis_transform(),
                ha="right", va="center",
                fontsize=FONT_SIZE_TICK - 1, fontweight="normal", color="#444444")

    return im


# ── Shared cell sizing ────────────────────────────────────────────
# Both fig1b and fig1c use aspect="equal" (square cells).  To ensure
# the same physical cell size and overall figure height we derive
# figsize from a common CELL_SIZE constant.
CELL_SIZE = 0.55          # inches per cell edge
H_PAD     = 2.0           # extra height for x-tick labels + margins
W_PAD     = 2.0           # extra width  for y-tick labels + margins
N_ROWS    = len(KEEP_STRATA)   # 7
N_COLS    = len(METHOD_COLS)   # 4
# Common figure height — matches fig1b (8 rows) so the two panels align
FIG_H     = 8 * CELL_SIZE + H_PAD


def main():
    fig_h = FIG_H
    fig_w = (N_COLS * CELL_SIZE + W_PAD) * 0.9
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plot(ax)
    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
