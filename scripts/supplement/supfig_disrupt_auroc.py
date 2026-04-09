#!/usr/bin/env python3
"""
Supplementary — Disruption AUROC by annotation category.

Per-annotation AUROC of disruption delta (var − ref) for predicting ClinVar
pathogenicity, grouped by browser annotation category. Only includes variants
where the reference annotation is present (ref > 0.5).

Input:  artifacts/fig2_disrupt_auroc.feather
Output: figures/supplement/supfig_disrupt_auroc.{png,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from theme.mayo_theme import apply_theme, save_figure

ARTIFACTS = ROOT / "artifacts"
OUT_STEM = ROOT / "figures" / "supplement" / "supfig_disrupt_auroc"

apply_theme()

# Browser group colors
GROUP_COLORS = {
    "InterPro":     "#DD8452",
    "ChIP-seq":     "#4C72B0",
    "Chromatin":    "#6495ED",
    "Structure":    "#55A868",
    "Substitution": "#C44E52",
    "ATAC-seq":     "#E74C3C",
    "PTM":          "#DA8BC3",
    "ELM Motif":    "#937860",
    "Region":       "#8C8C8C",
    "Protein":      "#1ABC9C",
    "Splice":       "#FF6600",
    "Other":        "#BBBBBB",
}

# Direction markers
DIR_MARKERS = {"neg": "v", "abs": "D"}  # triangle-down for loss, diamond for |change|


def plot(ax):
    """Horizontal boxplot of disruption AUROC by browser group."""
    df = pl.read_ipc(ARTIFACTS / "fig2_disrupt_auroc.feather")

    # Sort groups by median auroc_best ascending
    group_stats = (
        df.group_by("browser_group")
        .agg(pl.col("auroc_best").median().alias("med"))
        .sort("med")
    )
    group_order = group_stats["browser_group"].to_list()

    data_by_group, labels, counts, color_list = [], [], [], []
    directions_by_group = []
    for g in group_order:
        sub = df.filter(pl.col("browser_group") == g)
        vals = sub["auroc_best"].drop_nulls().to_numpy()
        dirs = sub["best_direction"].to_list()
        if len(vals) > 0:
            data_by_group.append(vals)
            directions_by_group.append(dirs)
            labels.append(g)
            counts.append(len(vals))
            color_list.append(GROUP_COLORS.get(g, "#888888"))

    positions = list(range(len(labels)))
    bp = ax.boxplot(
        data_by_group, positions=positions, vert=False, widths=0.55,
        patch_artist=True, showfliers=False,
        boxprops=dict(linewidth=0.8),
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )
    for patch, color in zip(bp["boxes"], color_list):
        patch.set_facecolor(color + "55")
        patch.set_edgecolor(color)

    # Jittered scatter with direction markers
    rng = np.random.default_rng(42)
    for i, (vals, dirs, color) in enumerate(zip(data_by_group, directions_by_group, color_list)):
        jitter = rng.uniform(-0.22, 0.22, size=len(vals))
        for v, j, d in zip(vals, jitter, dirs):
            marker = DIR_MARKERS.get(d, "o")
            ax.scatter(v, i + j, s=12, alpha=0.5, color=color,
                       marker=marker, edgecolors="none", zorder=3)

    ax.set_yticks(positions)
    ax.set_yticklabels([f"{l}  (n={c})" for l, c in zip(labels, counts)])
    ax.set_xlabel("AUROC")
    ax.axvline(0.5, color="grey", linestyle="--", alpha=0.3)
    ax.set_xlim(0.48, 1.02)
    ax.grid(axis="x", alpha=0.2)

    # Direction legend
    ax.scatter([], [], marker="v", color="gray", s=20, label="Loss predicts path.")
    ax.scatter([], [], marker="D", color="gray", s=15, label="|Change| predicts path.")
    ax.legend(loc="lower right", fontsize=6, framealpha=0.9)


def main():
    fig, ax = plt.subplots(figsize=(8, 6))
    plot(ax)
    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
