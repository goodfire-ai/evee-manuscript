#!/usr/bin/env python3
"""
Figure 2E — Auto-interpretation benchmarking: score progression across
prompt configurations for 3 LLM models.

Shows composite SCV judge scores (avg of mechanism_coverage,
biological_accuracy, specificity, novel_insights) for 6 ablation
configurations across Haiku 4.5, Sonnet 4.6, and Opus 4.6.

Input:  data/panels/fig2e.feather
Output: figures/figure2/panels/fig2e.{png,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from theme.mayo_theme import apply_theme, save_figure, FONT_SIZE_TICK

PANELS = ROOT / "artifacts"
OUT_STEM = ROOT / "figures" / "figure2" / "panels" / "fig2e"

apply_theme()

MODEL_COLORS = {"haiku": "#7fb3d8", "sonnet": "#f4a261", "opus": "#e76f51"}
MODEL_LABELS = {"haiku": "Haiku 4.5", "sonnet": "Sonnet 4.6", "opus": "Opus 4.6"}

CONFIG_LABELS = {
    "C1": "C1: Baseline\n(gene+csq)",
    "C2": "C2: +Context\n(+HGVSp,LOEUF,GO)",
    "C3": "C3: +Evo2\n(+score+profile)",
    "C4": "C4: Ctx+Evo2\n(current)",
    "C5": "C5: Ctx+Pred\n(no Evo2)",
    "C6": "C6: Full\n(all signals)",
}
CONFIGS = ["C1", "C2", "C3", "C4", "C5", "C6"]
MODELS = ["haiku", "sonnet", "opus"]


def plot(ax):
    """Line plot of composite score progression across prompt configs."""
    df = pl.read_ipc(PANELS / "fig2e.feather")

    for model in MODELS:
        sub = df.filter(pl.col("model") == model).sort("config")
        avgs = sub["composite_mean"].to_list()

        ax.plot(range(len(CONFIGS)), avgs, "o-",
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
                linewidth=2, markersize=8)

        for j, v in enumerate(avgs):
            ax.annotate(f"{v:.2f}", (j, v), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=FONT_SIZE_TICK - 1)

    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels([CONFIG_LABELS[c] for c in CONFIGS])
    ax.set_ylabel("Composite Score")
    ax.set_ylim(2.4, 4.0)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)


def main():
    fig, ax = plt.subplots(figsize=(10, 4))
    plot(ax)
    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
