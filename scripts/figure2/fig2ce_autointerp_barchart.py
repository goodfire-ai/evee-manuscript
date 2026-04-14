#!/usr/bin/env python3
"""
Auto-interpretation bar charts — context ablation evaluation.

Panel 1: Composite score grouped bar chart (3 models x 5 configs)
Panel 2: Per-axis breakdown (mechanism, bio accuracy, specificity) — 3-panel vertical

Input:  artifacts/context_ablation_eval.feather
Output: figures/figure2/fig2c_autointerp_composite_barchart.{png,pdf}
        figures/figure2/fig2e_autointerp_peraxis_barchart.{png,pdf}
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
PANELS = ROOT / "figures" / "figure2"

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


def _ci_95(values):
    """95% CI half-width using t-distribution."""
    n = len(values)
    if n < 2:
        return np.nan
    return stats.t.ppf(0.975, n - 1) * np.std(values, ddof=1) / np.sqrt(n)


def _add_evo2_shading(ax):
    """Add shading and labels for no-Evo2 / with-Evo2 regions."""
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
    """Grouped bar chart with 95% CI."""
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


def _load_data():
    return pl.read_ipc(ARTIFACTS / "context_ablation_eval.feather").to_pandas()


def plot(ax):
    """Plot composite bar chart onto a single axes (for assembler)."""
    df = _load_data()
    _plot_barchart(df, "composite", "Composite Score", ax, ylabel="Composite Score")
    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels([CONFIG_LABELS[c] for c in CONFIGS], fontsize=FONT_SIZE_TICK)
    ax.legend(loc="upper left", fontsize=FONT_SIZE_LEGEND)


def plot_axes(axes_3):
    """Plot per-axis bar charts onto a flat array of 3 axes (for assembler)."""
    df = _load_data()
    for idx, axis_name in enumerate(AXES):
        ax = axes_3[idx]
        _plot_barchart(df, axis_name, AXIS_TITLES[axis_name], ax)

        if idx == 0:
            ax.legend(loc="upper left", fontsize=FONT_SIZE_LEGEND)

        if idx == len(AXES) - 1:
            ax.set_xticks(range(len(CONFIGS)))
            ax.set_xticklabels([CONFIG_LABELS[c] for c in CONFIGS],
                               fontsize=FONT_SIZE_TICK)
        else:
            ax.tick_params(labelbottom=False)


def main():
    df = _load_data()

    # --- Panel 1: Composite bar chart ---
    fig, ax = plt.subplots(figsize=(7, 4))
    plot(ax)
    fig.tight_layout()
    save_figure(fig, PANELS / "fig2c_autointerp_composite_barchart")
    print("Saved: fig2c_autointerp_composite_barchart")

    # --- Panel 2: Per-axis 3-panel bar chart ---
    fig, axes = plt.subplots(3, 1, figsize=(5.5, 10), sharex=True)
    plot_axes(axes)
    fig.tight_layout()
    save_figure(fig, PANELS / "fig2e_autointerp_peraxis_barchart")
    print("Saved: fig2e_autointerp_peraxis_barchart")


if __name__ == "__main__":
    main()
