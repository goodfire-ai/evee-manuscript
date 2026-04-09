#!/usr/bin/env python3
"""
Figure 1G — DMS generalization: Spearman |rho| barplot, 2x2 faceted by gene.

Spearman |rho| between predicted scores and continuous DMS functional readouts
for BRCA1, BRCA2, TP53, and LDLR. Error bars show 95% bootstrap CIs.

Input:  data/panels/fig1f.csv
Output: figures/figure1/panels/fig1g.{png,pdf}
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
    apply_theme, save_figure, DMS_METHOD_SPEC,
    FONT_SIZE_TITLE, FONT_SIZE_TICK, FONT_SIZE_LABEL,
)

PANELS = ROOT / "data" / "panels"
OUT_STEM = ROOT / "figures" / "figure1" / "panels" / "fig1g"

apply_theme()

# Method groups with gaps for visual separation (reversed order)
SHOW_METHODS = (
    "clinvar_covprobe64", "clinvar_gfc_emb",
    "evo2_loss",
    "alphamissense", "cadd_phred",
)

# Custom x-positions with gaps between groups
X_POSITIONS = (0, 1, 2,  3.3, 4.3)

GENE_ORDER = ("BRCA1", "BRCA2", "TP53", "LDLR")


def _load_and_filter(path: Path) -> pl.DataFrame:
    """Load benchmark CSV and filter to correct eval_set per method."""
    df = pl.read_csv(path)
    df = df.filter(
        pl.col("method").is_in(SHOW_METHODS)
        & pl.col("gene").is_in(GENE_ORDER)
    )

    expected_eval_set = (
        pl.when(pl.col("method").str.starts_with("dms_iid"))
        .then(pl.lit("test20"))
        .when(pl.col("method").str.starts_with("clinvar_"))
        .then(pl.lit("clinvar_zeroshot"))
        .otherwise(pl.lit("all_annotated"))
    )
    return df.filter(pl.col("eval_set") == expected_eval_set)


def plot_dms_barplot(axes_2x2, df: pl.DataFrame, metric: str, ylabel: str, ylim: tuple):
    """Shared barplot logic for DMS results.

    Parameters
    ----------
    axes_2x2 : array of 4 Axes (flattened 2x2 grid)
    df : filtered benchmark DataFrame
    metric : "spearman" or "auroc"
    ylabel : y-axis label
    ylim : (ymin, ymax) tuple
    """
    method_set = set(df["method"].to_list())
    methods = [m for m in SHOW_METHODS if m in method_set]
    x = np.array([X_POSITIONS[SHOW_METHODS.index(m)] for m in methods])
    colors = [DMS_METHOD_SPEC[m][1] for m in methods]
    labels = [DMS_METHOD_SPEC[m][0] for m in methods]

    ci_col_lo = f"{metric}_ci_lo"
    ci_col_hi = f"{metric}_ci_hi"
    has_ci = ci_col_lo in df.columns and ci_col_hi in df.columns
    use_abs = (metric == "spearman")

    for gi, (gene, ax) in enumerate(zip(GENE_ORDER, axes_2x2)):
        sub = df.filter(pl.col("gene") == gene)
        vals, ci_lo, ci_hi = [], [], []

        for method in methods:
            row = sub.filter(pl.col("method") == method)
            if len(row) == 1 and row[0, metric] is not None:
                v = float(row[0, metric])
                if use_abs:
                    v = abs(v)

                lo_raw = row[0, ci_col_lo] if has_ci else None
                hi_raw = row[0, ci_col_hi] if has_ci else None

                if lo_raw is not None and hi_raw is not None:
                    if use_abs:
                        lo_abs, hi_abs = abs(float(lo_raw)), abs(float(hi_raw))
                        ci_bottom, ci_top = min(lo_abs, hi_abs), max(lo_abs, hi_abs)
                    else:
                        ci_bottom, ci_top = float(lo_raw), float(hi_raw)
                    ci_lo.append(v - ci_bottom)
                    ci_hi.append(ci_top - v)
                else:
                    ci_lo.append(0)
                    ci_hi.append(0)
                vals.append(v)
            else:
                vals.append(np.nan)
                ci_lo.append(0)
                ci_hi.append(0)

        ax.bar(x, vals, color=colors, width=0.7, edgecolor="white", linewidth=0.3)
        ax.errorbar(x, vals, yerr=[ci_lo, ci_hi],
                    fmt="none", color="black", capsize=2, lw=0.6)

        ax.set_title(gene, fontsize=FONT_SIZE_TITLE + 1, fontweight="semibold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=FONT_SIZE_TICK + 1, rotation=45, ha="right")
        ax.set_ylim(*ylim)
        ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK + 1)
        ax.grid(axis="y", alpha=0.15)

        if gi == 0:
            ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL + 1, fontweight="semibold")


def plot(axes_2x2):
    """Plot Figure 1G onto a flat array of 4 axes."""
    df = _load_and_filter(PANELS / "fig1f.csv")
    plot_dms_barplot(axes_2x2, df, metric="spearman",
                     ylabel="Spearman |\u03c1|", ylim=(0.0, 0.8))


def main():
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.5), sharey=True)
    plot(axes.flat)
    fig.tight_layout(w_pad=1.5)
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
