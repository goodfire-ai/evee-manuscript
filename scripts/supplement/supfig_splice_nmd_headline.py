#!/usr/bin/env python3
"""
Supplementary Figure — EVEE RNA-seq validation summary (splice + NMD).

Bar chart of validation rate by tier: Splice Hom/LOH, Splice Het, Splice All,
NMD All, All Combined.  Splice validation: PSI_mutant < 0.30 (PSI on mutant
allele, VAF-corrected).  NMD validation: expr_mutant < 0.50 (expression on
mutant allele relative to CCLE median, VAF-corrected).

Input:  artifacts/splice_nmd_validation.feather
Output: figures/supplement/supfig_splice_nmd_headline.{png,pdf}

Run:
    uv run python scripts/supplement/supfig_splice_nmd_headline.py
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
from theme.theme import apply_theme, save_figure

apply_theme()

ARTIFACT = ROOT / "artifacts" / "splice_nmd_validation.feather"
OUT_STEM  = ROOT / "figures" / "supplement" / "supfig_splice_nmd_headline"

# Per-variant best disruption metric (min dm across cell-line observations)
def per_var_best(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate to one row per variant: min dm_best, max vaf.  Zyg from max vaf."""
    return (
        df.group_by("variant_id").agg(
            pl.col("dm").min().alias("dm_best"),
            pl.col("vaf").max().alias("vaf_max"),
            pl.col("tier").first(),
        ).with_columns(
            pl.when(pl.col("vaf_max") >= 0.80).then(pl.lit("hom"))
              .when(pl.col("vaf_max") >= 0.40).then(pl.lit("het"))
              .otherwise(pl.lit("subcl")).alias("zyg")
        ).filter(pl.col("zyg").is_in(["hom", "het"]))
    )


def main():
    raw = pl.read_ipc(ARTIFACT)
    sp_raw = raw.filter(pl.col("tier") == "splice")
    nm_raw = raw.filter(pl.col("tier") == "nmd")

    sp = per_var_best(sp_raw).to_pandas()
    nm = per_var_best(nm_raw).to_pandas()

    import pandas as pd
    combined = pd.concat([
        sp.assign(thresh=0.30),
        nm.assign(thresh=0.50),
    ])

    groups = [
        ("Splice\nHom/LOH", sp[sp["zyg"] == "hom"],  0.30, "#a02c2c"),
        ("Splice\nHet",     sp[sp["zyg"] == "het"],   0.30, "#2966a3"),
        ("Splice\nAll",     sp,                       0.30, "#555555"),
        ("NMD\nAll",        nm,                       0.50, "#6b3aa1"),
        ("All\nCombined",   combined,                 None, "#333333"),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    xs = np.arange(len(groups))

    for xi, (label, sub_df, thresh, col) in enumerate(groups):
        d = len(sub_df)
        if thresh is None:
            n = int((sub_df["dm_best"] < sub_df["thresh"]).sum())
        else:
            n = int((sub_df["dm_best"] < thresh).sum())
        pct = n / d * 100 if d > 0 else 0
        ax.bar(xi, pct, color=col, alpha=0.85, width=0.6,
               edgecolor="white", lw=0.5)
        ax.text(xi, pct + 1.5, f"{n}/{d}\n({pct:.0f}%)",
                ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color=col)

    ax.axhline(100, color="#888888", lw=0.5, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.set_ylim(0, 120)
    ax.set_ylabel(
        "% variants with >70% mutant-allele disruption\n"
        "(splice: PSI_mut < 0.30; NMD: expr_mut < 0.50)",
        fontsize=9,
    )
    ax.set_title(
        "EVEE RNA-seq validation summary\n"
        "(EVEE path ≥ 0.99 · ≥ 5 carrier reads · ClinVar ★★+)",
        fontsize=10, fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"[wrote] {OUT_STEM}.png / .pdf")
    for label, sub_df, thresh, _ in groups:
        d = len(sub_df)
        if thresh is None:
            n = int((sub_df["dm_best"] < sub_df["thresh"]).sum())
        else:
            n = int((sub_df["dm_best"] < thresh).sum())
        print(f"  {label.replace(chr(10),' '):<20s} {n}/{d} = {n/d*100:.0f}%")


if __name__ == "__main__":
    main()
