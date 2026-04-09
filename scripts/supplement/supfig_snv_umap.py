#!/usr/bin/env python3
"""
Supplementary Figure — SNV-only UMAP of CovarianceProbe embeddings.

Two panels: pathogenic vs benign (left), consequence type (right).
(Demoted from main Figure 1E in the figure overhaul.)

Reads pre-computed UMAP from data/embeddings/umap_snv.*.
Run scripts/prepare/umap_snv.py first to generate.

Output: figures/supplement/supfig_snv_umap.{png,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import safetensors.numpy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from theme.mayo_theme import (
    apply_theme, save_figure, COLORS, CONSEQ_COLORS,
    FONT_SIZE_LEGEND, FONT_SIZE_TITLE,
)

EMBED_DIR = ROOT / "data" / "embeddings"
OUT_STEM = ROOT / "figures" / "supplement" / "supfig_snv_umap"

CONSEQ_ORDER = ("Missense", "Synonymous", "Nonsense", "Splice", "UTR", "Intronic", "Other")

apply_theme()


def main():
    tensors_path = EMBED_DIR / "umap_snv.safetensors"
    meta_path = EMBED_DIR / "umap_snv_meta.feather"

    if not tensors_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Run scripts/prepare/umap_snv.py first")

    tensors = safetensors.numpy.load_file(str(tensors_path))
    coords = tensors["coords"]
    y = tensors["pathogenic"]
    conseq = pl.read_ipc(meta_path)["consequence"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left panel: pathogenic vs benign
    for label, color, name in [(0, COLORS["benign"], "Benign"),
                                (1, COLORS["pathogenic"], "Pathogenic")]:
        mask = y == label
        ax1.scatter(coords[mask, 0], coords[mask, 1],
                    c=color, s=0.8, alpha=0.15, rasterized=True, label=name)

    ax1.set_title("Pathogenic vs Benign", fontsize=FONT_SIZE_TITLE, fontweight="semibold")
    leg1 = ax1.legend(fontsize=FONT_SIZE_LEGEND, markerscale=6,
                      loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    for handle in leg1.legend_handles:
        handle.set_alpha(1.0)

    # Right panel: consequence type
    for ctype in CONSEQ_ORDER:
        mask = conseq == ctype
        if mask.sum() == 0:
            continue
        ax2.scatter(coords[mask, 0], coords[mask, 1],
                    c=CONSEQ_COLORS.get(ctype, COLORS["light_gray"]),
                    s=0.8, alpha=0.15, rasterized=True, label=ctype)

    ax2.set_title("Consequence Type", fontsize=FONT_SIZE_TITLE, fontweight="semibold")
    leg2 = ax2.legend(fontsize=FONT_SIZE_LEGEND, markerscale=6,
                      loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False)
    for handle in leg2.legend_handles:
        handle.set_alpha(1.0)

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
