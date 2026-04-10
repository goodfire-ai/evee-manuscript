#!/usr/bin/env python3
"""
Figure 2E — Auto-interpretation benchmarking: effect of context on
interpretation quality across 3 LLM models.

Shows 3-axis composite score (mechanism rubric v2 + biological accuracy +
specificity) for 5 prompt configurations on 154 expert-reviewed ClinVar
variants with concordant probe predictions.

Progression: Coord only → +Gene → +Other context → +HGVSp → +Evo2 predictions

Input:  artifacts/fig2e.feather
Output: figures/figure2/panels/fig2e.{png,pdf}
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
from theme.mayo_theme import apply_theme, save_figure, FONT_SIZE_TICK, FONT_SIZE_LABEL, FONT_SIZE_TITLE

ARTIFACTS = ROOT / "artifacts"
OUT_STEM = ROOT / "figures" / "figure2" / "panels" / "fig2e"

apply_theme()

MODEL_COLORS = {"haiku": "#7fb3d8", "sonnet": "#f4a261", "opus": "#e76f51"}
MODEL_LABELS = {"haiku": "Haiku 4.5", "sonnet": "Sonnet 4.6", "opus": "Opus 4.6"}

CONFIG_LABELS = {
    "C-1":    "Coord\nonly",
    "C0":     "+Gene",
    "C2b":    "+Other\ncontext",
    "C0_hgvs": "+HGVSp",
    "C_hgvs": "+Evo2\npredictions",
}
CONFIGS = ["C-1", "C0", "C2b", "C0_hgvs", "C_hgvs"]
MODELS = ["haiku", "sonnet", "opus"]


def plot(ax):
    """Line plot with 95% CI of composite score progression."""
    df = pl.read_ipc(ARTIFACTS / "fig2e.feather")

    for model in MODELS:
        sub = df.filter(pl.col("model") == model)
        # Order by config list
        avgs, cis = [], []
        for config in CONFIGS:
            row = sub.filter(pl.col("config") == config)
            if len(row) == 0:
                avgs.append(float("nan"))
                cis.append(float("nan"))
            else:
                avgs.append(row["composite_mean"][0])
                cis.append(row["composite_ci95"][0])

        x = np.arange(len(CONFIGS))
        ax.errorbar(x, avgs, yerr=cis, fmt="o-",
                    color=MODEL_COLORS[model],
                    label=MODEL_LABELS[model],
                    linewidth=2, markersize=7,
                    capsize=3, capthick=1.2)

        for j, v in enumerate(avgs):
            if not np.isnan(v):
                ax.annotate(f"{v:.2f}", (j, v), textcoords="offset points",
                            xytext=(0, 11), ha="center",
                            fontsize=FONT_SIZE_TICK - 1)

    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels([CONFIG_LABELS[c] for c in CONFIGS],
                       fontsize=FONT_SIZE_TICK)
    ax.set_ylabel("Composite Score", fontsize=FONT_SIZE_LABEL)
    ax.set_ylim(1.0, 4.3)
    ax.legend(loc="lower right", fontsize=FONT_SIZE_TICK)
    ax.grid(axis="y", alpha=0.3)

    # Shade no-Evo2 vs Evo2 regions
    ax.axvspan(-0.5, 3.5, alpha=0.04, color="blue")
    ax.axvspan(3.5, 4.5, alpha=0.04, color="red")
    ax.text(1.5, 1.1, "No Evo2", ha="center",
            fontsize=FONT_SIZE_TICK, color="gray", style="italic")
    ax.text(4.0, 1.1, "With Evo2", ha="center",
            fontsize=FONT_SIZE_TICK, color="gray", style="italic")


def main():
    fig, ax = plt.subplots(figsize=(10, 4))
    plot(ax)
    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
