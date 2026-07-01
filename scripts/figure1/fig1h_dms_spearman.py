#!/usr/bin/env python3
"""
Fig 1H — DMS Spearman |rho|, 2x2 faceted by gene.

Spearman |rho| between predicted scores and continuous DMS functional readouts
for BRCA1, BRCA2, TP53, and LDLR. Error bars show 95% bootstrap CIs.

This is the 2x2 reformatted version (was fig1g in the previous layout).

Input:  artifacts/dms_benchmark.feather
Output: figures/figure1/fig1h_dms_spearman.{png,pdf}
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
from theme.theme import (
    apply_theme, save_figure, add_panel_label, DMS_METHOD_SPEC,
    FONT_SIZE_TITLE, FONT_SIZE_TICK, FONT_SIZE_LABEL,
    FIG_WIDTH_DOUBLE,
)

ARTIFACTS = ROOT / "artifacts"
OUT_STEM = ROOT / "figures" / "figure1" / "fig1h_dms_spearman"

apply_theme()

# Methods shown and their x-positions (with visual gap between groups)
SHOW_METHODS = (
    "clinvar_covprobe64", "clinvar_gfc_emb",
    "evo2_loss",
    "alphamissense", "cadd_phred",
)
X_POSITIONS = (0, 1, 2,  3.3, 4.3)

GENE_ORDER = ("BRCA1", "BRCA2", "TP53", "LDLR")


def _load_and_filter(path: Path) -> pl.DataFrame:
    df = pl.read_ipc(path)
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


def plot_gene(ax, sub: pl.DataFrame, gene: str, methods: list, x: np.ndarray,
              colors: list, labels: list, show_ylabel: bool):
    """Draw a single gene panel."""
    ci_col_lo = "spearman_ci_lo"
    ci_col_hi = "spearman_ci_hi"
    has_ci = ci_col_lo in sub.columns and ci_col_hi in sub.columns

    vals, ci_lo, ci_hi = [], [], []
    for method in methods:
        row = sub.filter(pl.col("method") == method)
        if len(row) == 1 and row[0, "spearman"] is not None:
            v = abs(float(row[0, "spearman"]))
            lo_raw = row[0, ci_col_lo] if has_ci else None
            hi_raw = row[0, ci_col_hi] if has_ci else None
            if lo_raw is not None and hi_raw is not None:
                lo_abs = abs(float(lo_raw))
                hi_abs = abs(float(hi_raw))
                ci_lo.append(v - min(lo_abs, hi_abs))
                ci_hi.append(max(lo_abs, hi_abs) - v)
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
    ax.set_ylim(0.0, 0.8)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK + 1)
    ax.grid(axis="y", alpha=0.15)

    if show_ylabel:
        ax.set_ylabel("Spearman |ρ|", fontsize=FONT_SIZE_LABEL + 1,
                       fontweight="semibold")


def main():
    df = _load_and_filter(ARTIFACTS / "dms_benchmark.feather")

    method_set = set(df["method"].to_list())
    methods = [m for m in SHOW_METHODS if m in method_set]
    x = np.array([X_POSITIONS[SHOW_METHODS.index(m)] for m in methods])
    colors = [DMS_METHOD_SPEC[m][1] for m in methods]
    labels = [DMS_METHOD_SPEC[m][0] for m in methods]

    # 2x2 grid — square-ish panels, double-column width
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH_DOUBLE, 6.5), sharey=True)

    for idx, (gene, ax) in enumerate(zip(GENE_ORDER, axes.flat)):
        row_idx = idx // 2
        show_ylabel = (idx % 2 == 0)   # left column only
        sub = df.filter(pl.col("gene") == gene)
        plot_gene(ax, sub, gene, methods, x, colors, labels, show_ylabel)

    fig.tight_layout(h_pad=2.5, w_pad=1.5)
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
