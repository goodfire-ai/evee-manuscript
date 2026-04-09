#!/usr/bin/env python3
"""
Figure 1E — UMAP of variant covariance embeddings colored by VEP consequence
type, using a custom 9-color palette.

Consequence types are ordered by functional severity (most damaging → least
specific) and colors are assigned so that functionally similar categories share
visually similar hues.

Reads pre-computed combined UMAP from data/embeddings/umap_combined.*.
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
# Custom palette
# ---------------------------------------------------------------------------
PALETTE = {
    "Cerulean":        "#00709d",
    "Brick Ember":     "#bd2600",
    "Forest Moss":     "#6d9916",
    "Amethyst":        "#b03bb0",
    "Sunflower Gold":  "#ebb000",
    "Cinnamon Wood":   "#ab734d",
    "Brilliant Azure": "#2685d2",
    "Autumn Leaf":     "#eb6200",
    "Grey Olive":      "#858585",
}

# ---------------------------------------------------------------------------
# Logical ordering (most damaging → least specific) and color assignments
#
#   Damaging / loss-of-function  → warm reds/oranges
#     Nonsense   – complete LoF  → Brick Ember  (deep red)
#     Frameshift – LoF           → Autumn Leaf  (orange)
#     Splice     – LoF / altered → Sunflower Gold (gold)
#   Moderate / coding change     → blues/purple
#     Missense   – amino-acid Δ  → Cerulean     (teal-blue)
#     In-frame   – length Δ      → Amethyst     (purple)
#   Benign / non-coding          → greens/neutrals
#     Synonymous – silent        → Forest Moss  (green)
#     UTR        – regulatory    → Brilliant Azure (bright blue)
#     Intronic   – deep intronic → Cinnamon Wood (brown)
#     Other      – catch-all     → Grey Olive   (gray)
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
    "Nonsense":    PALETTE["Brick Ember"],
    "Frameshift":  PALETTE["Autumn Leaf"],
    "Splice":      PALETTE["Sunflower Gold"],
    "Missense":    PALETTE["Cerulean"],
    "In-frame":    PALETTE["Amethyst"],
    "Synonymous":  PALETTE["Forest Moss"],
    "UTR":         PALETTE["Brilliant Azure"],
    "Intronic":    PALETTE["Cinnamon Wood"],
    "Other":       PALETTE["Grey Olive"],
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
