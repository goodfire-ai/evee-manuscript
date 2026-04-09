#!/usr/bin/env python3
"""
Figure 1C — Line plot: performance vs conservation tier (all variants).

Source: phase13/09_conservation_tiers.py → _plot_lineplot()
Input:  data/panels/fig1c.csv
Output: figures/figure1/panels/fig1c.{png,pdf}
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
    METHOD_LINE_STYLES, FONT_SIZE_TITLE, FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_LEGEND,
)

PANELS = ROOT / "data" / "panels"
OUT_STEM = ROOT / "figures" / "figure1" / "panels" / "fig1c"

apply_theme()

CONSERVATION_TIERS = (
    ("Fast-evolving",        -np.inf, -1.0),
    ("Weakly conserved",     -1.0,     0.0),
    ("Low conservation",      0.0,     2.0),
    ("Moderate conservation", 2.0,     5.0),
    ("Highly conserved",      5.0,     np.inf),
)

# Rename data method names to display names matching theme
METHOD_RENAME = {
    "Evo2 probe (cov-pool)": "Evo2 probe+",
    "Evo2 probe (w64 mean)": "Evo2 probe",
    "Evo2 cov probe": "Evo2 probe+",
    "Evo2 mean probe": "Evo2 probe",
}

METHOD_ORDER = (
    "Evo2 probe+", "Evo2 probe", "Evo2 loss",
    "AlphaMissense", "CADD v1.7", "GPN-MSA",
)


def plot_lineplot(ax, strat_df: pl.DataFrame, metric: str, title: str,
                  tier_defs: tuple, xlabel: str = ""):
    """Line plot: tiers (x) vs metric (y), one line per method."""
    strat_df = strat_df.with_columns(
        pl.col("method").replace_strict(METHOD_RENAME, default=pl.first())
    )

    df = strat_df.filter(pl.col("tier") != "Overall")
    tier_names = [t[0] for t in tier_defs]
    tiers = [t for t in tier_names if t in df["tier"].to_list()]
    methods = [m for m in METHOD_ORDER if m in df["method"].to_list()]

    tick_names, tick_stats = [], []
    for t in tiers:
        sub = df.filter(pl.col("tier") == t).row(0, named=True)
        n = int(sub.get("n_total", 0))
        p = int(sub.get("n_pathogenic", 0))
        tick_names.append(t)
        tick_stats.append(f"(n={n:,}, {100*p/n:.0f}%P)" if n else "")

    x = np.arange(len(tiers))

    for m in methods:
        sub = df.filter(pl.col("method") == m)
        vals, lo, hi = [], [], []
        for t in tiers:
            row = sub.filter(pl.col("tier") == t)
            if len(row) == 1:
                vals.append(float(row[0, metric]))
                lo.append(float(row[0, "auroc_lo"]))
                hi.append(float(row[0, "auroc_hi"]))
            else:
                vals.append(np.nan); lo.append(np.nan); hi.append(np.nan)

        vals, lo, hi = np.array(vals), np.array(lo), np.array(hi)
        style = METHOD_LINE_STYLES.get(m, dict(color="#B0B0B0", linewidth=1.2))
        ax.plot(x, vals, label=m, **style)
        if not np.all(np.isnan(lo)):
            ax.fill_between(x, lo, hi, alpha=0.08, color=style.get("color", "gray"))

    ax.set_xticks(x)
    ax.set_xticklabels([])  # Clear default labels
    # Manual two-part labels: bold name, regular stats
    for i, (name, stat) in enumerate(zip(tick_names, tick_stats)):
        ax.text(x[i], -0.04, name, transform=ax.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=FONT_SIZE_TICK, fontweight="semibold")
        if stat:
            ax.text(x[i], -0.08, stat, transform=ax.get_xaxis_transform(),
                    ha="center", va="top",
                    fontsize=FONT_SIZE_TICK - 1, fontweight="normal", color="#666666")
    ax.set_ylabel(metric.upper(), fontsize=FONT_SIZE_LABEL, fontweight="semibold")
    if title:
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight="semibold")
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc="upper center",
              bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.15)
    ax.set_ylim(0.5, 1.0)


def plot(ax):
    """Plot Figure 1C onto given axes."""
    df = pl.read_csv(PANELS / "fig1c.csv")
    plot_lineplot(ax, df, "auroc", "", CONSERVATION_TIERS,
                  xlabel="Conservation Tier (phyloP100way)")


def main():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot(ax)
    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
