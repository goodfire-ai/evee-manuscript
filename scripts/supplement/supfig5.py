#!/usr/bin/env python3
"""
Supplementary Figure 5 — UMAP of covariance-probe embeddings for ClinVar indels.

Two panels: (A) Pathogenic vs Benign, (B) Consequence type.

Reads pre-computed UMAP from data/embeddings/umap_indel.{safetensors,feather}.
Run scripts/prepare/umap_indel.py first to generate.

Output: figures/supplement/supfig5.{png,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import safetensors.torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from theme.mayo_theme import (
    apply_theme, save_figure, add_panel_label, COLORS, CONSEQ_COLORS,
    FONT_SIZE_TITLE, FONT_SIZE_LEGEND,
)

EMBED_DIR = ROOT / "data" / "embeddings"
OUT_STEM = ROOT / "figures" / "supplement" / "supfig5"

SCATTER = dict(s=0.4, rasterized=True, edgecolors="none")

CONSEQ_SPEC = (
    ("Frameshift",    CONSEQ_COLORS.get("Frameshift",   COLORS["pathogenic"]), 0.25),
    ("Intronic",      CONSEQ_COLORS.get("Intronic",     COLORS["gray"]),       0.20),
    ("Nonsense",      CONSEQ_COLORS.get("Nonsense",     COLORS["steel"]),      0.25),
    ("Splice",        CONSEQ_COLORS.get("Splice",       COLORS["lavender"]),   0.25),
    ("In-frame del",  CONSEQ_COLORS.get("In-frame del", COLORS["lavender"]),   0.25),
    ("In-frame ins",  CONSEQ_COLORS.get("In-frame ins", COLORS["gf_brown"]),   0.25),
    ("UTR",           CONSEQ_COLORS.get("UTR",          COLORS["sage"]),       0.25),
    ("Other",         CONSEQ_COLORS.get("Other",        COLORS["light_gray"]), 0.15),
)

apply_theme()


def main():
    tensors_path = EMBED_DIR / "umap_indel.safetensors"
    meta_path = EMBED_DIR / "umap_indel_meta.feather"

    if not tensors_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Run scripts/prepare/umap_indel.py first")

    tensors = safetensors.torch.load_file(str(tensors_path))
    coords = tensors["coords"].numpy()
    pathogenic = tensors["pathogenic"].numpy()
    csq = pl.read_ipc(meta_path)["csq_type"].to_numpy()

    ux, uy = coords[:, 0], coords[:, 1]
    n_total = len(ux)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Panel A: Pathogenic vs Benign
    ax = axes[0]
    mask_b = pathogenic == 0
    mask_p = pathogenic == 1
    ax.scatter(ux[mask_b], uy[mask_b], c=COLORS["benign"], alpha=0.15,
               label=f"Benign ({mask_b.sum():,})", **SCATTER)
    ax.scatter(ux[mask_p], uy[mask_p], c=COLORS["pathogenic"], alpha=0.25,
               label=f"Pathogenic ({mask_p.sum():,})", **SCATTER)
    ax.legend(loc="upper right", markerscale=6, framealpha=0.8, fontsize=FONT_SIZE_LEGEND)
    add_panel_label(ax, "a")

    # Panel B: Consequence type
    ax = axes[1]
    for label, color, alpha in CONSEQ_SPEC:
        mask = csq == label
        n = mask.sum()
        if n == 0:
            continue
        ax.scatter(ux[mask], uy[mask], c=color, alpha=alpha,
                   label=f"{label} ({n:,})", **SCATTER)
    ax.legend(loc="upper right", markerscale=6, framealpha=0.8, fontsize=FONT_SIZE_LEGEND)
    add_panel_label(ax, "b")

    # Remove all axes — UMAP dimensions are arbitrary
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        f"UMAP of Indel Covariance-Probe Embeddings (n = {n_total:,})",
        fontsize=FONT_SIZE_TITLE, fontweight="semibold", y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
