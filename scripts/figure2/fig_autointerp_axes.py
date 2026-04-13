#!/usr/bin/env python3
"""
Auto-interpretation per-axis score breakdown.

Three-panel figure showing mechanism coverage (rubric v2), biological accuracy,
and specificity scores across prompt configurations for 3 LLM models.
Companion to the composite score lineplot.

Progression: Coord only → +Gene → +Other context → +HGVSp → +Evo2 predictions

Input:  artifacts/supfig_autointerp_axes.feather
Output: figures/figure2/panels/fig_autointerp_axes.{png,pdf}
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
    apply_theme, save_figure,
    FONT_SIZE_TICK, FONT_SIZE_LABEL, FONT_SIZE_TITLE, FONT_SIZE_LEGEND,
)

ARTIFACTS = ROOT / "artifacts"
OUT_STEM = ROOT / "figures" / "figure2" / "panels" / "fig_autointerp_axes"

apply_theme()

MODEL_COLORS = {"haiku": "#7fb3d8", "sonnet": "#f4a261", "opus": "#e76f51"}
MODEL_LABELS = {"haiku": "Haiku 4.5", "sonnet": "Sonnet 4.6", "opus": "Opus 4.6"}

CONFIGS = ["C-1", "C0", "C2b", "C0_hgvs", "C_hgvs"]
CONFIG_LABELS = {
    "C-1": "Coord\nonly",
    "C0": "+Gene",
    "C2b": "+Other\ncontext",
    "C0_hgvs": "+HGVSp",
    "C_hgvs": "+Evo2\npredictions",
}
MODELS = ["haiku", "sonnet", "opus"]

AXIS_TITLES = {
    "mechanism_score": "Mechanism Coverage",
    "biological_accuracy": "Biological Accuracy",
    "specificity": "Specificity",
}
AXES = ["mechanism_score", "biological_accuracy", "specificity"]


def plot(axes_3):
    """Plot per-axis score breakdown onto a flat array of 3 axes."""
    df = pl.read_ipc(ARTIFACTS / "supfig_autointerp_axes.feather")

    for idx, axis_name in enumerate(AXES):
        ax = axes_3[idx]
        sub_df = df.filter(pl.col("axis") == axis_name)

        for model in MODELS:
            model_df = sub_df.filter(pl.col("model") == model)
            avgs, cis = [], []
            for config in CONFIGS:
                row = model_df.filter(pl.col("config") == config)
                if len(row) == 0:
                    avgs.append(float("nan"))
                    cis.append(float("nan"))
                else:
                    avgs.append(row["mean"][0])
                    cis.append(row["ci95"][0])

            x = np.arange(len(CONFIGS))
            ax.errorbar(x, avgs, yerr=cis, fmt="o-",
                        color=MODEL_COLORS[model],
                        label=MODEL_LABELS[model],
                        linewidth=1.8, markersize=6,
                        capsize=3, capthick=1.0)

            for j, v in enumerate(avgs):
                if not np.isnan(v):
                    ax.annotate(f"{v:.2f}", (j, v),
                                textcoords="offset points",
                                xytext=(0, 10), ha="center",
                                fontsize=FONT_SIZE_TICK - 1)

        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(0.8, 5.0)
        ax.set_ylabel("Score (1\u20135)", fontsize=FONT_SIZE_LABEL)
        ax.set_title(AXIS_TITLES[axis_name], fontsize=FONT_SIZE_TITLE)
        ax.grid(axis="y", alpha=0.3)

        ax.axvspan(-0.5, 3.5, alpha=0.04, color="blue")
        ax.axvspan(3.5, 4.5, alpha=0.04, color="red")

        if idx == 0:
            ax.legend(loc="upper left", fontsize=FONT_SIZE_LEGEND)

        if idx == len(AXES) - 1:
            ax.set_xticks(range(len(CONFIGS)))
            ax.set_xticklabels([CONFIG_LABELS[c] for c in CONFIGS],
                               fontsize=FONT_SIZE_TICK)


def main():
    fig, axes_arr = plt.subplots(3, 1, figsize=(5.5, 10), sharex=True)
    plot(axes_arr)
    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
