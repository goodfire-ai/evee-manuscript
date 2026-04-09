#!/usr/bin/env python3
"""
Supplementary Figure 2 — Full heatmap on all consequence types, naive and deconf.

Source: notebooks/2026-03-11-17-18_supplement.ipynb, cell 8
Input:  data/supplement/supfig2/benchmark_stratified_naive.csv
        data/supplement/supfig2/benchmark_stratified_deconfv3.csv
Output: figures/supplement/supfig2.{png,pdf}
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
    apply_theme, save_figure, HEATMAP_CMAP,
    FONT_SIZE_TITLE, FONT_SIZE_TICK, FONT_SIZE_CELL,
)

PANELS = ROOT / "data" / "panels"
OUT_STEM = ROOT / "figures" / "supplement" / "supfig2"

apply_theme()

CONSEQ_ORDER = (
    "Overall", "Missense",
    "Synonymous", "Nonsense", "Splice Donor", "Splice Acceptor",
    "Intronic", "Initiator Codon", "5' UTR", "3' UTR",
    "Non-coding Transcript", "Other",
)

# Unified method order — used for BOTH panels so x-axes match
METHOD_ORDER = (
    "Evo2 probe+", "Evo2 probe", "CADD v1.7",
    "AlphaMissense", "GPN-MSA", "Evo2 loss",
    "NTv3",
)


def _rename_methods(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("method").replace({
            "Covpool probe": "Evo2 probe+",
            "Evo2 probe": "Evo2 probe",
            "Evo2 mean probe": "Evo2 probe",
            "Evo2 cov probe": "Evo2 probe+",
            "NTv3 cosine dist": "NTv3",
        })
    )


def _blank_alphamissense_nonmissense(df: pl.DataFrame) -> pl.DataFrame:
    """Set AlphaMissense to NaN for non-missense rows; Overall = Missense."""
    # Get AlphaMissense Missense AUROC to use for Overall
    am_miss = df.filter(
        (pl.col("method") == "AlphaMissense") & (pl.col("consequence") == "Missense")
    )
    am_miss_auroc = am_miss[0, "auroc"] if len(am_miss) == 1 else None

    # Blank non-missense AlphaMissense rows, copy Missense value to Overall
    is_am = pl.col("method") == "AlphaMissense"
    return df.with_columns(
        pl.when(is_am & ~pl.col("consequence").is_in(("Overall", "Missense")))
        .then(pl.lit(None))
        .when(is_am & (pl.col("consequence") == "Overall"))
        .then(pl.lit(am_miss_auroc))
        .otherwise(pl.col("auroc"))
        .alias("auroc")
    )


def plot_heatmap_panel(ax, strat_df, metric, title, method_order):
    """Draw a heatmap on the given axes."""
    strat_df = _blank_alphamissense_nonmissense(strat_df)
    conseq_set = set(strat_df["consequence"].to_list())
    method_set = set(strat_df["method"].to_list())
    conseqs = [c for c in CONSEQ_ORDER if c in conseq_set]
    methods = [m for m in method_order if m in method_set]

    conseq_labels = []
    for c in conseqs:
        sub = strat_df.filter(pl.col("consequence") == c)
        n = int(sub[0, "n_total"]) if "n_total" in sub.columns else 0
        p = int(sub[0, "n_pathogenic"]) if "n_pathogenic" in sub.columns else 0
        conseq_labels.append(f"{c}\n(n={n:,}, {100*p/n:.0f}%P)" if n else c)

    matrix = np.full((len(conseqs), len(methods)), np.nan)
    pivot = strat_df.pivot(on="method", index="consequence", values=metric)
    for i, c in enumerate(conseqs):
        row = pivot.filter(pl.col("consequence") == c)
        if len(row) == 1:
            for j, m in enumerate(methods):
                if m in pivot.columns:
                    v = row[0, m]
                    if v is not None:
                        matrix[i, j] = float(v)

    valid_cols = ~np.all(np.isnan(matrix), axis=0)
    matrix = matrix[:, valid_cols]
    methods = [m for m, v in zip(methods, valid_cols) if v]

    im = ax.imshow(matrix, aspect="auto", cmap=HEATMAP_CMAP, vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(
        methods, fontsize=FONT_SIZE_TICK, fontweight="semibold",
        rotation=45, ha="right",
    )
    ax.set_yticks(range(len(conseqs)))
    ax.set_yticklabels(conseq_labels, fontsize=FONT_SIZE_TICK)
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight="semibold")

    for i in range(len(conseqs)):
        for j in range(len(methods)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=FONT_SIZE_CELL, fontweight="semibold",
                        color="#222222")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

    return im


def main():
    naive_df = _rename_methods(pl.read_csv(PANELS / "supfig2_naive.csv"))
    deconf_df = _rename_methods(pl.read_csv(PANELS / "supfig2_deconf.csv"))

    n_naive = int(naive_df.filter(pl.col("consequence") == "Overall")[0, "n_total"])
    n_deconf = int(deconf_df.filter(pl.col("consequence") == "Overall")[0, "n_total"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8),
                                    gridspec_kw={"width_ratios": [1.0, 1.0]})

    plot_heatmap_panel(
        ax1, naive_df, "auroc",
        f"ClinVar Star \u22651 (n={n_naive:,})",
        METHOD_ORDER,
    )
    plot_heatmap_panel(
        ax2, deconf_df, "auroc",
        f"CADD-Deconfounded ClinVar (n={n_deconf:,})",
        METHOD_ORDER,
    )

    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
