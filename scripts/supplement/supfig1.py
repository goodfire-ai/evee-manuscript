#!/usr/bin/env python3
"""
Supplementary Figure 1 — Per-position activation delta (|alt - ref|)
around SNV site, pathogenic vs benign overlaid on single plot.

Source: notebooks/2026-03-11-17-18_supplement.ipynb, cell 38
Input:  data/panels/supfig1.safetensors
Output: figures/supplement/supfig1.{png,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import safetensors.torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from theme.mayo_theme import (
    apply_theme, save_figure, COLORS,
    FONT_SIZE_LABEL, FONT_SIZE_LEGEND, FONT_SIZE_TICK,
)

PANELS = ROOT / "artifacts"
OUT_STEM = ROOT / "figures" / "supplement" / "supfig1"

apply_theme()


def main():
    data = safetensors.torch.load_file(str(PANELS / "supfig1.safetensors"))
    path_real = data["pathogenic_deltas"].numpy()   # [500, 2001]
    benign_real = data["benign_deltas"].numpy()     # [500, 2001]
    rel_pos = np.arange(2001) - 1000       # -1000 to +1000

    fig, ax = plt.subplots(figsize=(10, 4))

    for arr, color, label in [
        (path_real, COLORS["pathogenic"], "Pathogenic"),
        (benign_real, COLORS["benign"], "Benign"),
    ]:
        n = len(arr)
        mu = arr.mean(axis=0)
        se = arr.std(axis=0) / np.sqrt(n)
        ax.plot(rel_pos, mu, color=color, linewidth=1.0,
                label=f"{label} (n={n})")
        ax.fill_between(rel_pos, mu - 1.96 * se, mu + 1.96 * se,
                        color=color, alpha=0.12)

    ax.axvline(0, color=COLORS["gray"], linestyle="-", linewidth=0.8, alpha=0.6,
               label="SNV site")
    ax.set_ylabel("Mean |alt \u2212 ref|", fontsize=FONT_SIZE_LABEL, fontweight="semibold")
    ax.set_xlabel("Position relative to SNV (bp)", fontsize=FONT_SIZE_LABEL, fontweight="semibold")
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc="upper center",
              bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False)
    ax.grid(True, alpha=0.15)

    ax.text(0.02, 0.40, "\u2190 5\u2032", transform=ax.transAxes, fontsize=FONT_SIZE_TICK,
            color="gray", fontweight="semibold")
    ax.text(0.96, 0.40, "3\u2032 \u2192", transform=ax.transAxes, fontsize=FONT_SIZE_TICK,
            color="gray", fontweight="semibold")

    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
