#!/usr/bin/env python3
"""
Supplementary Figure — Combined UMAP of labeled SNVs + indels + VUS.

Joint embedding space from covariance64_pool embeddings.
Two panels: (A) Pathogenic/Benign/VUS, (B) Consequence type.

Reads pre-computed UMAP from data/embeddings/umap_combined.feather.
Run scripts/prepare/umap_combined.py first to generate.

Output: figures/supplement/supfig_combined_umap.{png,pdf}
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
from theme.mayo_theme import (
    apply_theme, save_figure, add_panel_label, COLORS, CONSEQ_COLORS,
    FONT_SIZE_TITLE, FONT_SIZE_LEGEND,
)

EMBED_DIR = ROOT / "artifacts"
OUT_STEM = ROOT / "figures" / "supplement" / "supfig_combined_umap"

CONSEQ_ORDER = ("Missense", "Synonymous", "Nonsense", "Splice", "Intronic",
                "Frameshift", "In-frame", "UTR", "Other")

COMBINED_CONSEQ_COLORS = {
    **CONSEQ_COLORS,
    "Noncoding":  COLORS["light_gray"],
    "In-frame":   COLORS["lavender"],
    "Frameshift":  COLORS["crimson"],
}

apply_theme()


def main():
    path = EMBED_DIR / "umap_combined.feather"
    


    df = pl.read_ipc(path)
    coords = df.select("umap_x", "umap_y").to_numpy()
    pathogenic = df["pathogenic"].to_numpy()
    csq = df["csq"].to_numpy()
    variant_type = df["variant_type"].to_numpy()

    ux, uy = coords[:, 0], coords[:, 1]
    is_snv = (variant_type == "SNV") | (variant_type == "VUS")
    is_indel = variant_type == "Indel"
    is_vus = variant_type == "VUS"

    SCATTER = dict(s=3.0, marker="o", rasterized=True, edgecolors="none")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel A: Label x variant type ---
    mask_b_snv = (pathogenic == 0) & is_snv
    mask_b_indel = (pathogenic == 0) & is_indel
    mask_vus = is_vus
    mask_p_snv = (pathogenic == 1) & is_snv
    mask_p_indel = (pathogenic == 1) & is_indel

    # Draw order: benign/VUS first (background), then pathogenic on top
    ax1.scatter(ux[mask_b_snv], uy[mask_b_snv], c=COLORS["benign"], alpha=0.10,
                label="Benign SNV", zorder=1, **SCATTER)
    ax1.scatter(ux[mask_vus], uy[mask_vus], c=COLORS["vus"], alpha=0.08,
                zorder=1, **SCATTER)  # No label yet, added below for legend order
    ax1.scatter(ux[mask_b_indel], uy[mask_b_indel], c=COLORS["sage"], alpha=0.15,
                label="Benign indel", zorder=2, **SCATTER)
    ax1.scatter(ux[mask_p_snv], uy[mask_p_snv], c=COLORS["pathogenic"], alpha=0.35,
                label="Pathogenic SNV", zorder=3, **SCATTER)
    ax1.scatter(ux[mask_p_indel], uy[mask_p_indel], c=COLORS["crimson"], alpha=0.45,
                label="Pathogenic indel", zorder=3, **SCATTER)
    # Invisible scatter just for legend entry in correct position
    ax1.scatter([], [], c=COLORS["vus"], label="VUS", **SCATTER)

    leg1 = ax1.legend(fontsize=FONT_SIZE_LEGEND, markerscale=3,
                      loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)
    for handle in leg1.legend_handles:
        handle.set_alpha(1.0)

    # --- Panel B: Consequence type (all variants, no SNV/indel split) ---
    for csq_name in CONSEQ_ORDER:
        color = COMBINED_CONSEQ_COLORS.get(csq_name, COLORS["light_gray"])
        mask = csq == csq_name
        if mask.sum() > 0:
            ax2.scatter(ux[mask], uy[mask], c=color, alpha=0.15,
                        label=csq_name, **SCATTER)

    leg2 = ax2.legend(fontsize=FONT_SIZE_LEGEND, markerscale=3,
                      loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=5, frameon=False)
    for handle in leg2.legend_handles:
        handle.set_alpha(1.0)

    # Remove all axes
    for ax in (ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
