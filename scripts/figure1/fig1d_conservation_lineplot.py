#!/usr/bin/env python3
"""
Conservation lineplot — Performance robustness across conservation levels.

AUROC by phyloP100way conservation tier for Evo2 probes, AlphaMissense,
CADD v1.7, Evo2 loss, and GPN-MSA. All variant types.

Input:  artifacts/conservation_benchmark.feather
Output: figures/figure1/fig1d_conservation_lineplot.{png,pdf}
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
    apply_theme, save_figure, COLORS,
    FONT_SIZE_TITLE, FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_LEGEND,
)

ARTIFACTS = ROOT / "artifacts"
OUT_STEM = ROOT / "figures" / "figure1" / "fig1d_conservation_lineplot"

apply_theme()

CONSERVATION_TIERS = (
    ("Fast-evolving",        -np.inf, -1.0),
    ("Weakly conserved",     -1.0,     0.0),
    ("Low conservation",      0.0,     2.0),
    ("Moderate conservation", 2.0,     5.0),
    ("Highly conserved",      5.0,     np.inf),
)

METHOD_RENAME = {
    "Evo2 probe (cov-pool)": "Evo2 Covariance",
    "Evo2 probe (w64 mean)": "Evo2 Mean",
    "Evo2 cov probe": "Evo2 Covariance",
    "Evo2 mean probe": "Evo2 Mean",
    "Evo2 probe+": "Evo2 Covariance",
    "Evo2 probe": "Evo2 Mean",
    "Evo2 loss": "Evo2 Loss",
}

METHOD_ORDER = (
    "Evo2 Covariance", "Evo2 Mean", "Evo2 Loss",
    "AlphaMissense", "CADD v1.7", "GPN-MSA",
)

# ---------------------------------------------------------------------------
# Colors from DMS palette (via METHOD_COLORS / DMS_METHOD_SPEC)
# ---------------------------------------------------------------------------
LINE_COLORS = {
    "Evo2 Covariance": COLORS["gf_orange"],   # #db8a48
    "Evo2 Mean":       COLORS["gf_brown"],     # #bbab8b
    "Evo2 Loss":       COLORS["gf_beige"],     # #D5CDBA
    "AlphaMissense":   COLORS["sage"],         # #7B9E87
    "CADD v1.7":       COLORS["steel"],        # #5A7D9A
    "GPN-MSA":         COLORS["lavender"],     # #9B8EA8
}

LINE_WIDTH = 1.5
LINE_ZORDER = 3

# Markers: triangle-up for probe+, square for probe, circle for everything else
LINE_MARKERS = {
    "Evo2 Covariance": "D",
    "Evo2 Mean":       "s",
}


def plot_lineplot(ax, df, metric, title, tiers=None,
                  xlabel="Conservation Tier (phyloP100way)"):
    """Reusable lineplot for AUROC by conservation tier.

    Parameters
    ----------
    ax : matplotlib Axes
    df : polars DataFrame with columns: method, tier, {metric}, {metric}_lo, {metric}_hi, n_total, n_pathogenic
    metric : column name for the y-axis value
    title : axes title (ignored if None)
    tiers : optional override for tier definitions, defaults to CONSERVATION_TIERS
    xlabel : x-axis label
    """
    if tiers is None:
        tiers = CONSERVATION_TIERS

    df = df.with_columns(
        pl.col("method").replace_strict(METHOD_RENAME, default=pl.first())
    )
    df = df.filter(pl.col("tier") != "Overall")

    tier_names = [t[0] for t in tiers]
    active_tiers = [t for t in tier_names if t in df["tier"].to_list()]
    methods = [m for m in METHOD_ORDER if m in df["method"].to_list()]

    tick_names, tick_stats = [], []
    for t in active_tiers:
        sub = df.filter(pl.col("tier") == t).row(0, named=True)
        n = int(sub.get("n_total", 0))
        p = int(sub.get("n_pathogenic", 0))
        tick_names.append(t)
        tick_stats.append(f"(n={n:,}, {100*p/n:.1f}% P)" if n else "")

    x = np.arange(len(active_tiers))
    lo_col, hi_col = f"{metric}_lo", f"{metric}_hi"
    has_ci = lo_col in df.columns and hi_col in df.columns

    for m in methods:
        sub = df.filter(pl.col("method") == m)
        vals, lo, hi = [], [], []
        for t in active_tiers:
            row = sub.filter(pl.col("tier") == t)
            if len(row) == 1:
                vals.append(float(row[0, metric]))
                lo.append(float(row[0, lo_col]) if has_ci else np.nan)
                hi.append(float(row[0, hi_col]) if has_ci else np.nan)
            else:
                vals.append(np.nan); lo.append(np.nan); hi.append(np.nan)

        vals, lo, hi = np.array(vals), np.array(lo), np.array(hi)
        color = LINE_COLORS.get(m, "#B0B0B0")
        marker = LINE_MARKERS.get(m, "o")

        ax.plot(x, vals, label=m, color=color, linewidth=LINE_WIDTH,
                zorder=LINE_ZORDER, marker=marker, markersize=5,
                markeredgecolor="white", markeredgewidth=0.4)
        if not np.all(np.isnan(lo)):
            ax.fill_between(x, lo, hi, alpha=0.08, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([])
    label_fs = FONT_SIZE_TICK - 1
    stat_fs = FONT_SIZE_TICK - 2
    for i, (name, stat) in enumerate(zip(tick_names, tick_stats)):
        ax.text(x[i], -0.04, name.title(), transform=ax.get_xaxis_transform(),
                ha="right", va="top", rotation=45,
                fontsize=label_fs, fontweight="semibold")
        if stat:
            ax.text(x[i], -0.10, stat, transform=ax.get_xaxis_transform(),
                    ha="right", va="top", rotation=45,
                    fontsize=stat_fs, color="#666666")

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.set_ylabel(metric.upper(), fontsize=FONT_SIZE_LABEL, fontweight="semibold")
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="y", alpha=0.15)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABEL, fontweight="semibold")
    if title:
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight="semibold")
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc="lower right",
              bbox_to_anchor=(0.99, 0.01), ncol=2, frameon=True,
              framealpha=0.9, edgecolor="#cccccc", fancybox=False)


def plot(ax):
    """Plot Figure 1F onto given axes."""
    df = pl.read_ipc(ARTIFACTS / "conservation_benchmark.feather")
    plot_lineplot(ax, df, "auroc", title=None, xlabel=None)


def main():
    fig, ax = plt.subplots(figsize=(3.85, 3.5))
    plot(ax)
    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
