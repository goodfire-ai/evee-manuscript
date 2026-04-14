#!/usr/bin/env python3
"""
Supplementary Figure — Evo 2-7B layer sweep.

Bar chart of IID AUROC per Evo2-7B transformer block (layers 0-31),
with the best layer (L27) highlighted.

Input:  artifacts/layer_sweep_evo2_7b.csv
Output: figures/supplement/supfig1_layer_sweep.{png,pdf}
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from theme.theme import apply_theme, save_figure, COLORS

ARTIFACTS = ROOT / "artifacts"
OUT_STEM = ROOT / "figures" / "supplement" / "supfig1_layer_sweep"

apply_theme()


def main():
    df = pd.read_csv(ARTIFACTS / "layer_sweep_evo2_7b.csv")

    # Filter to evo2-7b-tw, subref embeddings
    df = df[(df["model"] == "evo2-7b-tw") & (df["embedding"].str.endswith("_subref"))].copy()

    # Extract layer index from embedding name (e.g. "l04_subref" -> 4)
    df["layer"] = df["embedding"].apply(lambda s: int(re.search(r"l(\d+)_", s).group(1)))
    df = df.sort_values("layer").reset_index(drop=True)

    layers = df["layer"].values
    aurocs = df["iid_auroc"].values

    best_idx = aurocs.argmax()
    best_layer = layers[best_idx]
    best_auroc = aurocs[best_idx]

    # Colors: gf_orange for all bars, darker accent for best layer
    bar_color = COLORS["gf_orange"]
    highlight_color = "#B0652A"  # Darker warm brown for highlight

    colors = [highlight_color if l == best_layer else bar_color for l in layers]

    fig, ax = plt.subplots(figsize=(7.2, 3))
    ax.bar(layers, aurocs, color=colors, width=0.8, edgecolor="white", linewidth=0.3)

    # Annotate best layer only
    ax.annotate(
        f"L{best_layer}: {best_auroc:.4f}",
        xy=(best_layer, best_auroc),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        fontsize=7,
        fontweight="semibold",
        color=highlight_color,
    )

    ax.set_xlabel("Evo2-7B Layer (block index)")
    ax.set_ylabel("AUROC")
    ax.set_title("Evo 2-7B layer sweep", fontweight="semibold")
    ax.set_xticks(layers)
    ax.set_xlim(-0.8, max(layers) + 0.8)
    ax.set_ylim(0.74, 1.0)

    fig.tight_layout()
    OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
