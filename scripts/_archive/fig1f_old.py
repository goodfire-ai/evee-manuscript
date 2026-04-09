#!/usr/bin/env python3
"""
Figure 1F — DMS Spearman |ρ| barplot, 2x2 faceted by gene.

Three method groups with visual separation:
  [Baselines]  gap  [ClinVar probes]  gap  [DMS IID probes]

Source: dms/01_dms_benchmark.py (benchmark_results.csv)
Input:  data/figure1/fig1f/benchmark_results.csv
Output: figures/figure1/panels/fig1f.{png,pdf}
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
OUT_STEM = ROOT / "figures" / "figure1" / "panels" / "fig1f"

apply_theme()

# Method groups with gaps for visual separation
# Group 1: external baselines | Group 2: Evo2 loss | Group 3: ClinVar probes | Group 4: DMS probes
SHOW_METHODS = (
    "cadd_phred", "alphamissense",
    "evo2_loss",
    "clinvar_gfc_emb", "clinvar_covprobe64",
    "dms_iid_L27_w64", "dms_iid_covprobe64",
)

# Custom x-positions with gaps between groups
X_POSITIONS = (0, 1,  2.3,  3.6, 4.6,  6.0, 7.0)

GENE_ORDER = ("BRCA1", "BRCA2", "TP53", "LDLR")


def _load_and_filter(path: Path) -> pl.DataFrame:
    """Load benchmark CSV and filter to correct eval_set per method."""
    df = pl.read_csv(path)
    df = df.filter(
        pl.col("method").is_in(SHOW_METHODS)
        & pl.col("gene").is_in(GENE_ORDER)
    )

    # Select correct eval_set per method type
    expected_eval_set = (
        pl.when(pl.col("method").str.starts_with("dms_iid"))
        .then(pl.lit("test20"))
        .when(pl.col("method").str.starts_with("clinvar_"))
        .then(pl.lit("clinvar_zeroshot"))
        .otherwise(pl.lit("all_annotated"))
    )
    return df.filter(pl.col("eval_set") == expected_eval_set)


def plot_dms_barplot(axes_2x2, df: pl.DataFrame, metric: str, ylabel: str, ylim: tuple):
    """Shared barplot logic for DMS results (used by fig1f and supfig6).

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
    use_abs = (metric == "spearman")  # |ρ| for Spearman

    for gi, (gene, ax) in enumerate(zip(GENE_ORDER, axes_2x2)):
        sub = df.filter(pl.col("gene") == gene)
        vals, ci_lo, ci_hi = [], [], []

        for method in methods:
            row = sub.filter(pl.col("method") == method)
            if len(row) == 1 and row[0, metric] is not None:
                v = float(row[0, metric])
                if use_abs:
                    v = abs(v)

                # CI bounds
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

        ax.set_title(gene, fontsize=FONT_SIZE_TITLE, fontweight="semibold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=FONT_SIZE_TICK, rotation=45, ha="right")
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.15)

        if gi % 2 == 0:
            ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL, fontweight="semibold")


def plot(axes_2x2):
    """Plot Figure 1F onto a flat array of 4 axes."""
    df = _load_and_filter(PANELS / "fig1f.csv")
    plot_dms_barplot(axes_2x2, df, metric="spearman",
                     ylabel="Spearman |ρ|", ylim=(0.0, 0.8))


def main():
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharey=True)
    plot(axes.flat)
    fig.tight_layout(h_pad=2.0, w_pad=1.5)
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
