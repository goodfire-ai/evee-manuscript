#!/usr/bin/env python3
"""
Figure 1E — UMAP of variant covariance embeddings colored by VEP consequence
type, using a custom 9-color palette.

Consequence types are ordered by functional severity (most damaging → least
specific) and colors are assigned so that functionally similar categories share
visually similar hues.

Reads pre-computed combined UMAP from artifacts/umap_combined.*.
Run scripts/prepare/umap_combined.py first to generate.

Output: figures/figure1/panels/fig1e.{png,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from theme.mayo_theme import apply_theme, save_figure, FONT_SIZE_TITLE, FONT_SIZE_LEGEND

sys.path.insert(0, str(ROOT / "scripts" / "figure1"))
from _umap_common import load_combined_umap, cleanup_axes, format_legend, SCATTER_KW

OUT_STEM = ROOT / "figures" / "figure1" / "panels" / "fig1e"

# ---------------------------------------------------------------------------
# Unified palette — consistent with EVEE website (UmapCanvas.svelte).
# Ordered by functional severity (most damaging → least specific).
# ---------------------------------------------------------------------------
CONSEQ_ORDER_RECOLOR = (
    "Nonsense",
    "Frameshift",
    "Splice",
    "Missense",
    "In-frame",
    "Synonymous",
    "UTR",
    "Intronic",
    "Other",
)

CONSEQ_COLORS_RECOLOR = {
    "Missense":    "#d98033",
    "Synonymous":  "#66b366",
    "Frameshift":  "#cc4040",
    "Nonsense":    "#c97088",
    "Splice":      "#8c4db3",
    "Intronic":    "#8099b3",
    "In-frame":    "#c4a035",
    "UTR":         "#2685d2",
    "Other":       "#b0b0b0",
}

apply_theme()


def plot(ax):
    """Plot combined UMAP colored by consequence type (recolored palette)."""
    coords, pathogenic, csq, variant_type = load_combined_umap()
    ux, uy = coords[:, 0], coords[:, 1]

    for csq_name in CONSEQ_ORDER_RECOLOR:
        color = CONSEQ_COLORS_RECOLOR[csq_name]
        mask = csq == csq_name
        if mask.sum() > 0:
            ax.scatter(ux[mask], uy[mask], c=color, alpha=0.5,
                       label=csq_name, **SCATTER_KW)

    format_legend(ax, FONT_SIZE_LEGEND, ncol=5)
    cleanup_axes(ax)


def main():
    fig, ax = plt.subplots(figsize=(6, 4.5))
    plot(ax)
    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
