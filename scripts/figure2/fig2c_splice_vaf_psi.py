#!/usr/bin/env python3
"""
Fig 2C — EVEE splice + branchpoint variant validation: VAF vs 1−PSI.

Scatter plot of VAF (allele fraction in carrier) vs 1 − PSI(canonical),
where 0 = no splice disruption and 1 = complete splice loss.
Each point is one (variant × carrier cell line) observation from Sanger CMP.

Adapted from scripts/supplement/supfig_splice_vaf_validation.py,
re-routed to figures/figure2/.

Input:  artifacts/splice_vaf_psi.feather
Output: figures/figure2/fig2c_splice_vaf_psi.{png,pdf}
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
from theme.theme import apply_theme, save_figure, FONT_SIZE_LABEL, FONT_SIZE_TICK

apply_theme()

ARTIFACT = ROOT / "artifacts" / "splice_vaf_psi.feather"
OUT_STEM  = ROOT / "figures" / "figure2" / "fig2c_splice_vaf_psi"


def main():
    df = pl.read_ipc(ARTIFACT).to_pandas()

    hom = df[df["zyg"] == "hom"]
    het = df[df["zyg"] == "het"]
    rho, pval = stats.spearmanr(df["vaf"], df["disruption"])

    fig, ax = plt.subplots(figsize=(5.0, 4.5))

    ax.scatter(
        hom["vaf"], hom["disruption"],
        marker="s", s=55, color="#a02c2c", alpha=0.85,
        edgecolors="white", lw=0.4, zorder=4,
        label=f"Hom/LOH (VAF ≥ 0.80)  n={len(hom)}",
    )
    ax.scatter(
        het["vaf"], het["disruption"],
        marker="o", s=45, color="#2966a3", alpha=0.75,
        edgecolors="white", lw=0.3, zorder=3,
        label=f"Het (VAF 0.40–0.80)  n={len(het)}",
    )

    xs = np.linspace(0.35, 1.05, 100)
    ax.plot(xs, xs, "--", color="#888888", lw=1.0, alpha=0.7,
            label="Expected (PSI_mutant = 0)")

    ax.text(0.03, 0.97,
            f"Spearman ρ = {rho:.2f}\np = {pval:.1e}  n = {len(df)}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=FONT_SIZE_TICK, color="#444444")

    ax.set_xlim(0.35, 1.05)
    ax.set_ylim(-0.04, 1.08)
    ax.set_xlabel("VAF (allele fraction in carrier)",
                  fontsize=FONT_SIZE_LABEL, fontweight="semibold")
    ax.set_ylabel("1 − PSI", fontsize=FONT_SIZE_LABEL, fontweight="semibold")
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.legend(fontsize=FONT_SIZE_TICK - 2, loc="lower right", framealpha=0.0)

    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")
    print(f"  n={len(df)} (hom={len(hom)}, het={len(het)})")
    print(f"  Spearman ρ={rho:.2f}  p={pval:.2e}")


if __name__ == "__main__":
    main()
