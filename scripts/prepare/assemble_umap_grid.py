#!/usr/bin/env python3
"""
Assemble UMAP grid search results into a summary montage.

Reads individual PNGs from figures/test/umap_grid/ and assembles
them into a contact sheet grouped by metric, sorted by parameters.

Usage:
    python scripts/prepare/assemble_umap_grid.py
"""
import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GRID_DIR = ROOT / "figures" / "test" / "umap_grid"
OUT_PATH = ROOT / "figures" / "test" / "umap_grid_summary.png"

N_NEIGHBORS = [40, 80, 100]
MIN_DISTS = [0.03, 0.1, 0.2, 0.3]
SPREADS = [1.0, 3.0, 5.0]
METRICS = ["correlation", "cosine"]


def main():
    # Layout: rows = (metric, spread) combos, cols = (nn, md) combos
    # 2 metrics × 3 spreads = 6 row groups
    # 3 nn × 4 md = 12 columns
    row_params = list(itertools.product(METRICS, SPREADS))   # 6 rows
    col_params = list(itertools.product(N_NEIGHBORS, MIN_DISTS))  # 12 cols

    n_rows = len(row_params)
    n_cols = len(col_params)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 2.5))

    missing = 0
    for ri, (metric, spread) in enumerate(row_params):
        for ci, (nn, md) in enumerate(col_params):
            ax = axes[ri, ci]
            tag = f"nn{nn}_md{md}_sp{spread}_{metric}"
            png_path = GRID_DIR / f"{tag}.png"

            if png_path.exists():
                img = mpimg.imread(str(png_path))
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, f"MISSING\n{tag}", ha="center", va="center",
                        fontsize=6, color="red", transform=ax.transAxes)
                missing += 1

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.3)

            # Column headers (top row only)
            if ri == 0:
                ax.set_title(f"nn={nn}\nmd={md}", fontsize=7, fontweight="bold")

            # Row labels (left column only)
            if ci == 0:
                ax.set_ylabel(f"{metric}\nspread={spread}", fontsize=7,
                              fontweight="bold", rotation=0, labelpad=60, va="center")

    fig.suptitle("UMAP Parameter Grid Search", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    total = n_rows * n_cols
    print(f"Assembled {total - missing}/{total} panels into {OUT_PATH}")
    if missing:
        print(f"  ({missing} missing — check SLURM logs in {GRID_DIR}/logs/)")


if __name__ == "__main__":
    main()
