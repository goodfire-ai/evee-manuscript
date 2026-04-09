#!/usr/bin/env python3
"""
Figure 1A — Heatmap: naive ClinVar (star >= 1) P,LP / B,LB.

Source: phase13/06_plot.py → plot_consequence_heatmaps()
Input:  data/panels/fig1a.csv
Output: figures/figure1/panels/fig1a.{png,pdf}
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
    FONT_SIZE_TITLE, FONT_SIZE_TICK, FONT_SIZE_CELL, FONT_SIZE_LABEL,
)

PANELS = ROOT / "data" / "panels"
OUT_STEM = ROOT / "figures" / "figure1" / "panels" / "fig1a"

apply_theme()

# ── Consequence order (merged) ──────────────────────────────────
CONSEQ_ORDER = (
    "Overall", "Missense", "Synonymous", "Nonsense",
    "Splice", "UTR", "Intronic", "Other",
)

MERGE_GROUPS = {
    "Splice": ("Splice Donor", "Splice Acceptor"),
    "UTR": ("5' UTR", "3' UTR"),
}

DROP_CONSEQS = {
    "Missense (canonical)", "Missense (non-canonical)",
    "Splice Donor", "Splice Acceptor", "5' UTR", "3' UTR",
    "Initiator Codon", "Non-coding Transcript",
}
DROP_METHODS = {"REVEL", "PolyPhen2", "NTv3 L2 dist", "NTv3 subref probe"}

# Evo2 probes first, then baselines by overall AUROC descending
METHOD_PREFIX = ("Evo2 probe+", "Evo2 probe", "Evo2 loss")


def _merge_categories(strat_df: pl.DataFrame) -> pl.DataFrame:
    """Add merged Splice / UTR rows via n-weighted averaging."""
    merge_rows = []
    methods = strat_df["method"].unique().to_list()
    numeric_cols = [c for c in strat_df.columns if c not in ("method", "type", "consequence", "n_total", "n_pathogenic", "n_valid")]
    for merged_name, sub_conseqs in MERGE_GROUPS.items():
        for method in methods:
            sub = strat_df.filter(
                (pl.col("method") == method)
                & pl.col("consequence").is_in(list(sub_conseqs))
            )
            if len(sub) < len(sub_conseqs):
                continue
            total_n = sub["n_total"].sum()
            w = sub["n_total"].to_numpy() / total_n
            row = {
                "method": method,
                "type": sub[0, "type"],
                "consequence": merged_name,
                "n_total": int(total_n),
                "n_pathogenic": int(sub["n_pathogenic"].sum()),
                "n_valid": int(sub["n_valid"].sum()),
            }
            for col in numeric_cols:
                vals = sub[col].to_numpy()
                if np.all(np.isnan(vals)):
                    row[col] = None
                else:
                    row[col] = float((np.nan_to_num(vals) * w).sum())
            merge_rows.append(row)
    if merge_rows:
        strat_df = pl.concat([strat_df, pl.DataFrame(merge_rows)], how="diagonal_relaxed")
    return strat_df


def _prepare(strat_df: pl.DataFrame) -> pl.DataFrame:
    """Merge categories, filter, and rename methods."""
    strat_df = _merge_categories(strat_df)
    df = strat_df.filter(
        ~pl.col("consequence").is_in(list(DROP_CONSEQS))
        & ~pl.col("method").is_in(list(DROP_METHODS))
    )
    rename_map = {
        "Covpool probe": "Evo2 probe+",
        "Evo2 probe": "Evo2 probe",
        "Evo2 mean probe": "Evo2 probe",
        "Evo2 cov probe": "Evo2 probe+",
        "NTv3 cosine dist": "NTv3",
        "AlphaGenome composite": "AlphaGenome",
    }
    df = df.with_columns(pl.col("method").replace_strict(rename_map, default=pl.first()))

    # AlphaMissense: blank non-missense/non-overall, set Overall = Missense
    am_non_miss = (pl.col("method") == "AlphaMissense") & ~pl.col("consequence").is_in(["Overall", "Missense"])
    blank_cols = [c for c in ("auroc", "auprc", "auroc_lo", "auroc_hi") if c in df.columns]
    df = df.with_columns(
        [pl.when(am_non_miss).then(None).otherwise(pl.col(c)).alias(c) for c in blank_cols]
    )

    # Copy AlphaMissense Missense values to Overall
    am_miss = df.filter((pl.col("method") == "AlphaMissense") & (pl.col("consequence") == "Missense"))
    if len(am_miss) == 1:
        am_ov_mask = (pl.col("method") == "AlphaMissense") & (pl.col("consequence") == "Overall")
        copy_cols = [c for c in ("auroc", "auroc_lo", "auroc_hi", "auprc") if c in df.columns]
        df = df.with_columns(
            [pl.when(am_ov_mask).then(pl.lit(am_miss[0, c])).otherwise(pl.col(c)).alias(c)
             for c in copy_cols if am_miss[0, c] is not None]
        )
    return df


def plot_heatmap(ax, strat_df: pl.DataFrame, metric: str, title: str):
    """Draw heatmap on given axes: consequences (y) × methods (x)."""
    conseqs = [c for c in CONSEQ_ORDER if c in strat_df["consequence"].to_list()]

    # Sort methods: Evo2 probes first, then by Overall descending
    overall = strat_df.filter(pl.col("consequence") == "Overall").select("method", metric)
    rest = overall.sort(metric, descending=True)["method"].to_list()
    rest = [m for m in rest if m not in METHOD_PREFIX]
    methods = [m for m in METHOD_PREFIX if m in overall["method"].to_list()] + rest

    # Consequence labels: bold name, regular stats
    conseq_names = []
    conseq_stats = []
    for c in conseqs:
        sub = strat_df.filter(pl.col("consequence") == c).row(0, named=True)
        n = int(sub.get("n_total", 0))
        p = int(sub.get("n_pathogenic", 0))
        conseq_names.append(c)
        conseq_stats.append(f"(n={n:,}, {100*p/n:.0f}%P)" if n else "")

    # Build matrix via pivot
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

    # Remove all-NaN cols
    valid = ~np.all(np.isnan(matrix), axis=0)
    matrix = matrix[:, valid]
    methods = [m for m, v in zip(methods, valid) if v]

    im = ax.imshow(matrix, aspect="equal", cmap=HEATMAP_CMAP, vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(
        methods, fontsize=FONT_SIZE_TICK, fontweight="semibold",
        rotation=45, ha="right",
    )
    ax.set_yticks(range(len(conseqs)))
    ax.set_yticklabels([])  # Clear default labels
    # Manual two-part labels: bold name, regular stats
    for i, (name, stat) in enumerate(zip(conseq_names, conseq_stats)):
        ax.text(-0.02, i - 0.01, name, transform=ax.get_yaxis_transform(),
                ha="right", va="center",
                fontsize=FONT_SIZE_TICK, fontweight="semibold")
        if stat:
            ax.text(-0.02, i + 0.24, stat, transform=ax.get_yaxis_transform(),
                    ha="right", va="center",
                    fontsize=FONT_SIZE_TICK - 1, fontweight="normal", color="#666666")
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight="semibold", pad=6)

    # Cell annotations
    for i in range(len(conseqs)):
        for j in range(len(methods)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=FONT_SIZE_CELL, fontweight="semibold",
                        color="#222222")

    # Black border around heatmap
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

    return im


def plot(ax):
    """Plot Figure 1A onto given axes."""
    strat_df = pl.read_csv(PANELS / "fig1a.csv")
    df = _prepare(strat_df)
    ov = df.filter(pl.col("consequence") == "Overall").row(0, named=True)
    title = f"ClinVar Star \u22651 (n={int(ov['n_total']):,})"
    plot_heatmap(ax, df, "auroc", title)


def main():
    fig, ax = plt.subplots(figsize=(9, 7))
    plot(ax)
    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
