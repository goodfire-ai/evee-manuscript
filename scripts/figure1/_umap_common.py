"""Shared UMAP plotting utilities for combined variant embeddings."""
import functools
from pathlib import Path

import numpy as np
import polars as pl
import safetensors.numpy

ROOT = Path(__file__).resolve().parents[2]
EMBED_DIR = ROOT / "data" / "embeddings"

# Default scatter kwargs for UMAP plots
SCATTER_KW = dict(s=5.0, marker="o", rasterized=True, edgecolors="none")

# Consequence draw order
CONSEQ_ORDER = ("Missense", "Synonymous", "Nonsense", "Splice", "Intronic",
                "Frameshift", "In-frame", "UTR", "Other")


@functools.lru_cache(maxsize=1)
def load_combined_umap():
    """Load combined UMAP data (cached so fig1d + fig1e don't reload)."""
    tensors_path = EMBED_DIR / "umap_combined.safetensors"
    meta_path = EMBED_DIR / "umap_combined_meta.feather"

    if not tensors_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Run scripts/prepare/umap_combined.py first")

    tensors = safetensors.numpy.load_file(str(tensors_path))
    coords = tensors["coords"]
    pathogenic = tensors["pathogenic"]
    meta = pl.read_ipc(meta_path)
    csq = meta["csq"].to_numpy()
    variant_type = meta["variant_type"].to_numpy()

    return coords, pathogenic, csq, variant_type


def cleanup_axes(ax):
    """Remove ticks and spines for UMAP plots (dimensions are arbitrary)."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def format_legend(ax, fontsize, ncol=3, markerscale=3):
    """Create a legend with solid markers below the axes."""
    leg = ax.legend(fontsize=fontsize, markerscale=markerscale,
                    loc="upper center", bbox_to_anchor=(0.5, -0.02),
                    ncol=ncol, frameon=False)
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)
    return leg
