#!/usr/bin/env python3
"""
Supplementary Figure 3 — Dataset characterization: consequence/pathogenic
ratios, ClinVar Star>=1 vs CADD-deconfounded.

Source: notebooks/2026-03-11-17-18_supplement.ipynb, cell 14 (revised)
Input:  data/supplement/supfig3/dataset_characterization.csv
Output: figures/supplement/supfig3.{png,pdf}
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

PANELS = ROOT / "data" / "panels"
OUT_STEM = ROOT / "figures" / "supplement" / "supfig3"

apply_theme()

LABEL_NAIVE = "ClinVar Star\u22651 (834K)"
LABEL_DECONF = "Deconf (159K)"
COLOR_NAIVE = COLORS["steel"]      # Cool muted tone
COLOR_DECONF = COLORS["gf_brown"]  # Goodfire warm brown


def main():
    comp_df = pl.read_csv(PANELS / "supfig3.csv")

    # Rename "Naive (834K)" → "ClinVar Star≥1 (834K)"
    comp_df = comp_df.with_columns(
        pl.col("dataset").replace({"Naive (834K)": LABEL_NAIVE})
    )

    # Keep only consequences present in both datasets
    both = (
        comp_df.group_by("consequence")
        .agg(pl.col("dataset").n_unique().alias("n_datasets"))
        .filter(pl.col("n_datasets") == 2)["consequence"]
        .to_list()
    )
    comp_df = comp_df.filter(pl.col("consequence").is_in(both))

    # Order by naive count descending
    naive_sub = comp_df.filter(pl.col("dataset") == LABEL_NAIVE).sort("count", descending=True)
    ordered_conseqs = naive_sub["consequence"].to_list()

    x = np.arange(len(ordered_conseqs))
    width = 0.35

    def _get(dataset, col):
        sub = comp_df.filter(pl.col("dataset") == dataset)
        return [sub.filter(pl.col("consequence") == c)[0, col] for c in ordered_conseqs]

    counts_n = _get(LABEL_NAIVE, "count")
    counts_d = _get(LABEL_DECONF, "count")
    npath_n = _get(LABEL_NAIVE, "n_pathogenic")
    npath_d = _get(LABEL_DECONF, "n_pathogenic")
    prates_n = _get(LABEL_NAIVE, "pathogenic_rate")
    prates_d = _get(LABEL_DECONF, "pathogenic_rate")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Variant counts (log scale) with pathogenic/benign stacked
    ax1.barh(x - width / 2, counts_n, width, color=COLOR_NAIVE, alpha=0.30)
    ax1.barh(x + width / 2, counts_d, width, color=COLOR_DECONF, alpha=0.30)
    ax1.barh(x - width / 2, npath_n, width, label=LABEL_NAIVE, color=COLOR_NAIVE, alpha=0.85)
    ax1.barh(x + width / 2, npath_d, width, label=LABEL_DECONF, color=COLOR_DECONF, alpha=0.85)
    ax1.set_xscale("log")
    ax1.set_yticks(x)
    ax1.set_yticklabels(ordered_conseqs, fontsize=FONT_SIZE_TICK)
    ax1.set_xlabel("Variant Count", fontsize=FONT_SIZE_LABEL, fontweight="semibold")
    ax1.set_title("Variant Counts by Consequence", fontsize=FONT_SIZE_TITLE, fontweight="semibold")
    ax1.invert_yaxis()
    ax1.grid(axis="x", alpha=0.15)
    ax1.text(0.98, 0.02, "dark = pathogenic\nlight = total",
             transform=ax1.transAxes, fontsize=FONT_SIZE_TICK, ha="right", va="bottom",
             color="gray", style="italic")

    # Panel B: Pathogenic rate
    ax2.barh(x - width / 2, prates_n, width, color=COLOR_NAIVE, alpha=0.8)
    ax2.barh(x + width / 2, prates_d, width, color=COLOR_DECONF, alpha=0.8)
    ax2.set_yticks(x)
    ax2.set_yticklabels(ordered_conseqs, fontsize=FONT_SIZE_TICK)
    ax2.set_xlabel("Pathogenic Rate (%)", fontsize=FONT_SIZE_LABEL, fontweight="semibold")
    ax2.set_title("Pathogenic Rate by Consequence", fontsize=FONT_SIZE_TITLE, fontweight="semibold")
    ax2.invert_yaxis()
    ax2.grid(axis="x", alpha=0.15)
    ax2.set_xlim(0, 105)

    # Panel C: Composition (% of dataset)
    comp_n = [100 * c / sum(counts_n) for c in counts_n]
    comp_d = [100 * c / sum(counts_d) for c in counts_d]
    ax3.barh(x - width / 2, comp_n, width, color=COLOR_NAIVE, alpha=0.8)
    ax3.barh(x + width / 2, comp_d, width, color=COLOR_DECONF, alpha=0.8)
    ax3.set_yticks(x)
    ax3.set_yticklabels(ordered_conseqs, fontsize=FONT_SIZE_TICK)
    ax3.set_xlabel("% of Dataset", fontsize=FONT_SIZE_LABEL, fontweight="semibold")
    ax3.set_title("Consequence Composition", fontsize=FONT_SIZE_TITLE, fontweight="semibold")
    ax3.invert_yaxis()
    ax3.grid(axis="x", alpha=0.15)

    # Single shared legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=FONT_SIZE_LEGEND,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.10)
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
