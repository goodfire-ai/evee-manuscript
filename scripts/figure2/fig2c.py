#!/usr/bin/env python3
"""
Figure 2C — UMAP of annotation disruption space with semantic clusters
and 30+ annotation labels.

Input:  data/panels/fig2c.csv
Output: figures/figure2/panels/fig2c.{png,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from theme.mayo_theme import (
    apply_theme, save_figure, FONT_SIZE_TICK, FONT_SIZE_LEGEND,
)

PANELS = ROOT / "data" / "panels"
OUT_STEM = ROOT / "figures" / "figure2" / "panels" / "fig2c"

apply_theme()

CLUSTER_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#CCBB44", "#8172B3", "#C44E52", "#937860"]

# Prefixes to strip for short display names
_STRIP_PREFIXES = [
    "interpro_", "chipseq_", "chromhmm_", "atacseq_", "in_", "is_",
    "elm_", "ptm_", "region_", "amino_acid_", "secondary_structure_",
    "ccre_", "fstack_", "dna_shape_", "spliceai_", "gtex_",
]

N_LABELS = 30  # number of annotations to label


def _short_name(head: str) -> str:
    """Shorten a head name by stripping common prefixes."""
    for prefix in _STRIP_PREFIXES:
        if head.startswith(prefix):
            return head[len(prefix):]
    return head


def _pick_labels(df, n=N_LABELS):
    """Select annotation indices to label: extremes from each cluster."""
    coords = df.select("umap1", "umap2").to_numpy()
    cluster_ids = df["cluster_id"].to_numpy()
    unique_ids = sorted(set(cluster_ids))

    selected = []
    per_cluster = max(2, n // len(unique_ids))

    for cid in unique_ids:
        mask = np.where(cluster_ids == cid)[0]
        if len(mask) == 0:
            continue
        pts = coords[mask]
        cx, cy = pts.mean(axis=0)
        dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        # Pick the most extreme points
        top = np.argsort(dists)[-per_cluster:]
        selected.extend(mask[top].tolist())

    # Trim to target count, preferring diversity across clusters
    return selected[:n]


def plot(ax):
    """UMAP scatter with clusters and annotation labels."""
    df = pl.read_csv(PANELS / "fig2c.csv")
    coords = df.select("umap1", "umap2").to_numpy()
    cluster_ids = df["cluster_id"].to_numpy()
    heads = df["head"].to_list()

    # Build cluster info
    unique_ids = sorted(set(cluster_ids))
    id_to_name = {}
    for row in df.select("cluster_id", "cluster_name").unique().iter_rows():
        id_to_name[row[0]] = row[1]

    # Plot clusters
    for cid in unique_ids:
        name = id_to_name.get(cid, f"Cluster {cid}")
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
        mask = cluster_ids == cid
        pts = coords[mask]
        n = mask.sum()

        ax.scatter(pts[:, 0], pts[:, 1], s=25, alpha=0.55, color=color,
                   edgecolors="white", linewidth=0.3, zorder=2)

        # Lighter dashed circle
        cx, cy = np.median(pts, axis=0)
        dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        radius = np.percentile(dists, 90) + 0.3

        circle = Circle(
            (cx, cy), radius,
            facecolor=(*matplotlib.colors.to_rgba(color)[:3], 0.04),
            edgecolor=color, linewidth=1.2, linestyle="--", zorder=1,
        )
        ax.add_patch(circle)

        ax.annotate(
            f"{name}\n(n={n})", xy=(cx, cy + radius + 0.1),
            fontsize=FONT_SIZE_LEGEND, fontweight="bold", color=color,
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=color, alpha=0.9, linewidth=0.8),
            zorder=10,
        )

    # Add annotation labels
    label_idx = _pick_labels(df)
    offsets = [
        (12, 10), (-12, 10), (12, -10), (-12, -10),
        (18, 3), (-18, 3), (3, 18), (-3, -18),
        (20, -8), (-20, 8), (15, -15), (-15, 15),
    ]
    for i, idx in enumerate(label_idx):
        x, y = coords[idx]
        name = _short_name(heads[idx])
        if len(name) > 28:
            name = name[:25] + "…"
        cid = cluster_ids[idx]
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]

        ax.scatter(x, y, s=50, color=color, edgecolors="black",
                   linewidth=0.6, zorder=5)
        ox, oy = offsets[i % len(offsets)]
        ax.annotate(
            name, xy=(x, y), xytext=(ox, oy), textcoords="offset points",
            fontsize=FONT_SIZE_TICK - 1, color="#333333",
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5,
                            lw=0.5, connectionstyle="arc3,rad=0.15"),
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor=color, alpha=0.85, linewidth=0.5),
            zorder=6,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(alpha=0.1)


def main():
    fig, ax = plt.subplots(figsize=(8, 7))
    plot(ax)
    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
