#!/usr/bin/env python3
"""
Supplementary Figure — Context ablation extended analysis.

Panel 1: Pathogenic vs Benign composite bar chart (1x2)
Panel 2: Score distribution stacked bars for all 3 axes, Opus only (1x3)
Panel 3: Per-axis line plots (3x1 vertical)

Input:  artifacts/context_ablation_eval.feather
Output: figures/supplement/supfig9_autointerp_pathogenic_barchart.{png,pdf}
        figures/supplement/supfig10_autointerp_score_distribution.{png,pdf}
        figures/supplement/supfig9_autointerp_peraxis_lineplot.{png,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from theme.theme import (
    apply_theme, save_figure,
    FONT_SIZE_TICK, FONT_SIZE_LABEL, FONT_SIZE_TITLE, FONT_SIZE_LEGEND,
)

ARTIFACTS = ROOT / "artifacts"
OUT_DIR = ROOT / "figures" / "supplement"

apply_theme()

MODEL_COLORS = {"haiku": "#7fb3d8", "sonnet": "#f4a261", "opus": "#e76f51"}
MODEL_LABELS = {"haiku": "Haiku 4.5", "sonnet": "Sonnet 4.6", "opus": "Opus 4.6"}

CONFIGS = [
    "coord_only",
    "coord_gene",
    "coord_gene_context",
    "coord_gene_context_hgvsp",
    "coord_gene_context_hgvsp_evo2",
]
CONFIG_LABELS = {
    "coord_only": "Coordinates",
    "coord_gene": "+Gene",
    "coord_gene_context": "+Gene\nContext",
    "coord_gene_context_hgvsp": "+HGVSp",
    "coord_gene_context_hgvsp_evo2": "+Evo2",
}
MODELS = ["haiku", "sonnet", "opus"]

AXES = ["mechanism_score", "biological_accuracy", "specificity"]
AXIS_TITLES = {
    "mechanism_score": "Mechanism Coverage",
    "biological_accuracy": "Biological Accuracy",
    "specificity": "Specificity",
}

LEVEL_COLORS = {1: "#d73027", 2: "#fc8d59", 3: "#fee090", 4: "#91bfdb", 5: "#4575b4"}


def _ci_95(values):
    n = len(values)
    if n < 2:
        return np.nan
    return stats.t.ppf(0.975, n - 1) * np.std(values, ddof=1) / np.sqrt(n)


def _add_evo2_shading(ax):
    n_no_evo2 = sum(1 for c in CONFIGS if "evo2" not in c)
    ax.axvspan(-0.5, n_no_evo2 - 0.5, alpha=0.04, color="blue")
    ax.axvspan(n_no_evo2 - 0.5, len(CONFIGS) - 0.5, alpha=0.04, color="red")

    ax.text((n_no_evo2 - 1) / 2, 4.75, "Baselines without Evo2",
            ha="center", va="center", fontsize=FONT_SIZE_LABEL,
            color="#999999", weight="bold")
    ax.text(n_no_evo2 + (len(CONFIGS) - n_no_evo2 - 1) / 2, 4.75, "With Evo2",
            ha="center", va="center", fontsize=FONT_SIZE_LABEL,
            color="#999999", weight="bold")


def _plot_barchart(df_pd, score_col, title, ax, ylabel="Score (1\u20135)"):
    n_models = len(MODELS)
    bar_width = 0.25
    x = np.arange(len(CONFIGS))

    for i, model in enumerate(MODELS):
        avgs, cis = [], []
        for cfg in CONFIGS:
            sub = df_pd[(df_pd["config"] == cfg) & (df_pd["model"] == model)][score_col].dropna()
            avgs.append(sub.mean() if len(sub) > 0 else np.nan)
            cis.append(_ci_95(sub.values) if len(sub) > 1 else 0)

        offset = (i - (n_models - 1) / 2) * bar_width
        ax.bar(x + offset, avgs, bar_width, yerr=cis,
               color=MODEL_COLORS[model], label=MODEL_LABELS[model],
               edgecolor="white", linewidth=0.5,
               capsize=2, error_kw={"linewidth": 1.0})

        for j, v in enumerate(avgs):
            if not np.isnan(v):
                ax.annotate(f"{v:.2f}", (x[j] + offset, v + cis[j]),
                            textcoords="offset points", xytext=(0, 4),
                            ha="center", fontsize=FONT_SIZE_TICK - 1)

    ax.set_xlim(-0.5, len(CONFIGS) - 0.5)
    ax.set_ylim(0, 5.0)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL)
    ax.set_title(title, fontsize=FONT_SIZE_TITLE)
    ax.grid(axis="y", alpha=0.3)
    _add_evo2_shading(ax)


def _plot_lineplot(df_pd, score_col, title, ax, ylabel="Score (1\u20135)"):
    x = np.arange(len(CONFIGS))

    for model in MODELS:
        avgs, cis = [], []
        for cfg in CONFIGS:
            sub = df_pd[(df_pd["config"] == cfg) & (df_pd["model"] == model)][score_col].dropna()
            avgs.append(sub.mean() if len(sub) > 0 else np.nan)
            cis.append(_ci_95(sub.values) if len(sub) > 1 else 0)

        ax.errorbar(x, avgs, yerr=cis, fmt="o-",
                    color=MODEL_COLORS[model], label=MODEL_LABELS[model],
                    linewidth=2, markersize=7, capsize=3, capthick=1.2)

        for j, v in enumerate(avgs):
            if not np.isnan(v):
                ax.annotate(f"{v:.2f}", (j, v), textcoords="offset points",
                            xytext=(0, 11), ha="center",
                            fontsize=FONT_SIZE_TICK - 1)

    ax.set_xlim(-0.5, len(CONFIGS) - 0.5)
    ax.set_ylim(0, 5.0)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL)
    ax.set_title(title, fontsize=FONT_SIZE_TITLE)
    ax.grid(axis="y", alpha=0.3)
    _add_evo2_shading(ax)


def main():
    df = pl.read_ipc(ARTIFACTS / "context_ablation_eval.feather").to_pandas()

    # --- Panel 1: Pathogenic vs Benign bar chart ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, label in zip(axes, ["Pathogenic", "Benign"]):
        sub = df[df["label"].str.lower().isin(
            ["pathogenic", "likely_pathogenic"] if label == "Pathogenic"
            else ["benign", "likely_benign"]
        )]
        _plot_barchart(sub, "composite", f"Composite ({label})", ax,
                       ylabel="Composite Score")
        ax.set_xticks(range(len(CONFIGS)))
        ax.set_xticklabels([CONFIG_LABELS[c] for c in CONFIGS],
                           fontsize=FONT_SIZE_TICK)
        if label == "Pathogenic":
            ax.legend(loc="upper left", fontsize=FONT_SIZE_LEGEND)

    fig.tight_layout()
    save_figure(fig, OUT_DIR / "supfig9_autointerp_pathogenic_barchart")
    print("Saved: supfig9_autointerp_pathogenic_barchart")

    # --- Panel 2: Score distribution stacked bar (Opus, all 3 axes) ---
    opus_df = df[df["model"] == "opus"].copy()
    x = np.arange(len(CONFIGS))
    width = 0.6

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for idx, axis_name in enumerate(AXES):
        ax = axes[idx]
        opus_df["_int"] = opus_df[axis_name].round().astype("Int64")

        bottoms = np.zeros(len(CONFIGS))
        for level in [1, 2, 3, 4, 5]:
            fracs = []
            for cfg in CONFIGS:
                sub = opus_df[opus_df["config"] == cfg]["_int"].dropna()
                n_total = len(sub)
                n_level = (sub == level).sum()
                fracs.append(n_level / n_total * 100 if n_total > 0 else 0)
            ax.bar(x, fracs, width, bottom=bottoms, label=f"L{level}",
                   color=LEVEL_COLORS[level], edgecolor="white", linewidth=0.5)
            bottoms += fracs

        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.set_ylim(0, 100)
        ax.set_title(f"{AXIS_TITLES[axis_name]} (Opus)", fontsize=FONT_SIZE_TITLE)
        ax.set_xticks(x)
        ax.set_xticklabels([CONFIG_LABELS[c] for c in CONFIGS],
                           fontsize=FONT_SIZE_TICK)
        ax.grid(axis="y", alpha=0.3)

        if idx == 0:
            ax.set_ylabel("% of variants", fontsize=FONT_SIZE_LABEL)

        if idx == len(AXES) - 1:
            ax.legend(title="Level", bbox_to_anchor=(1.02, 1), loc="upper left",
                      fontsize=FONT_SIZE_LEGEND, title_fontsize=FONT_SIZE_LEGEND)

    fig.tight_layout()
    save_figure(fig, OUT_DIR / "supfig10_autointerp_score_distribution")
    print("Saved: supfig10_autointerp_score_distribution")

    # --- Panel 3: Per-axis line plots (3x1 vertical) ---
    fig, axes = plt.subplots(3, 1, figsize=(5.5, 10), sharex=True)
    for idx, axis_name in enumerate(AXES):
        ax = axes[idx]
        _plot_lineplot(df, axis_name, AXIS_TITLES[axis_name], ax)

        if idx == 0:
            ax.legend(loc="upper left", fontsize=FONT_SIZE_LEGEND)

        if idx == len(AXES) - 1:
            ax.set_xticks(range(len(CONFIGS)))
            ax.set_xticklabels([CONFIG_LABELS[c] for c in CONFIGS],
                               fontsize=FONT_SIZE_TICK)
        else:
            ax.tick_params(labelbottom=False)

    fig.tight_layout()
    save_figure(fig, OUT_DIR / "supfig9_autointerp_peraxis_lineplot")
    print("Saved: supfig9_autointerp_peraxis_lineplot")


if __name__ == "__main__":
    main()
