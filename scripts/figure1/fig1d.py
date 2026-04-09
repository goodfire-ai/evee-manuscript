#!/usr/bin/env python3
"""
Figure 1D — UMAP of variant covariance embeddings colored by ClinVar
pathogenicity label (benign, VUS, pathogenic) for SNVs and indels.

Reads pre-computed combined UMAP from data/embeddings/umap_combined.*.
Run scripts/prepare/umap_combined.py first to generate.

Output: figures/figure1/panels/fig1d.{png,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from theme.mayo_theme import (
    apply_theme, save_figure, COLORS,
    FONT_SIZE_TITLE, FONT_SIZE_LEGEND,
)

sys.path.insert(0, str(ROOT / "scripts" / "figure1"))
from _umap_common import load_combined_umap, cleanup_axes, format_legend, SCATTER_KW

OUT_STEM = ROOT / "figures" / "figure1" / "panels" / "fig1d"

apply_theme()


def plot(ax):
    """Plot combined UMAP colored by pathogenicity label."""
    coords, pathogenic, csq, variant_type = load_combined_umap()
    ux, uy = coords[:, 0], coords[:, 1]

    is_snv = (variant_type == "SNV") | (variant_type == "VUS")
    is_indel = variant_type == "Indel"
    is_vus = variant_type == "VUS"

    mask_b_snv = (pathogenic == 0) & is_snv
    mask_b_indel = (pathogenic == 0) & is_indel
    mask_vus = is_vus
    mask_p_snv = (pathogenic == 1) & is_snv
    mask_p_indel = (pathogenic == 1) & is_indel

    # Draw order: benign/VUS first (background), then pathogenic on top
    ax.scatter(ux[mask_b_snv], uy[mask_b_snv], c=COLORS["benign"], alpha=0.5,
               label="Benign SNV", zorder=1, **SCATTER_KW)
    ax.scatter(ux[mask_vus], uy[mask_vus], c=COLORS["vus"], alpha=0.5,
               zorder=1, **SCATTER_KW)
    ax.scatter(ux[mask_b_indel], uy[mask_b_indel], c=COLORS["sage"], alpha=0.5,
               label="Benign indel", zorder=2, **SCATTER_KW)
    ax.scatter(ux[mask_p_snv], uy[mask_p_snv], c=COLORS["crimson"], alpha=0.5,
               label="Pathogenic SNV", zorder=3, **SCATTER_KW)
    ax.scatter(ux[mask_p_indel], uy[mask_p_indel], c=COLORS["pathogenic"], alpha=0.5,
               label="Pathogenic indel", zorder=3, **SCATTER_KW)
    # Invisible scatter for legend entry in correct position
    ax.scatter([], [], c=COLORS["vus"], label="VUS", **SCATTER_KW)

    format_legend(ax, FONT_SIZE_LEGEND, ncol=3)
    cleanup_axes(ax)


def main():
    fig, ax = plt.subplots(figsize=(6, 4.5))
    plot(ax)
    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
