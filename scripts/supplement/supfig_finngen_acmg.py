#!/usr/bin/env python3
"""
Supplementary Figure — FinnGen R12 ACMG/AMP evidence scores
for ClinVar resubmission candidates identified by the EVEE pipeline.

Input:  artifacts/finngen_resubmission.feather
Output: figures/supplement/supfig_finngen_acmg.{png,pdf}

Run:
    uv run python scripts/supplement/supfig_finngen_acmg.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from theme.theme import apply_theme, save_figure, COLORS

apply_theme()

ARTIFACT = ROOT / "artifacts" / "finngen_resubmission.feather"
OUT_STEM  = ROOT / "figures" / "supplement" / "supfig_finngen_acmg"

# ── criterion colors (from manuscript palette) ────────────────────────────────
CRIT_COLORS = {
    "PVS1": COLORS["crimson"],    # red  — null variant, strongest evidence
    "PS4":  COLORS["steel"],      # blue — case-control odds ratio
    "PP3":  COLORS["gf_orange"],  # orange — EVEE/Evo2 computational score
    "PM2":  COLORS["sage"],       # green — Finnish population frequency
}

CRIT_LABELS = {
    "PVS1": "PVS1 (null variant)",
    "PS4":  "PS4 (case-control OR)",
    "PP3":  "PP3 (EVEE/Evo2 score)",
    "PM2":  "PM2 (Finnish AF < 0.1%)",
}

THRESHOLDS = [
    (4,  "LP-lean", "#AAAAAA"),
    (6,  "LP",      "#666666"),
    (10, "P",       "#222222"),
]


def main():
    df = pl.read_ipc(ARTIFACT).to_pandas()

    labels   = df["display_label"].tolist()
    n        = len(df)
    x        = np.arange(n)
    bar_w    = 0.52

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    bottoms = np.zeros(n)
    for crit in ["PVS1", "PS4", "PP3", "PM2"]:
        heights = df[f"ACMG_{crit}_pts"].values.astype(float)
        ax.bar(x, heights, bar_w,
               bottom=bottoms,
               color=CRIT_COLORS[crit],
               label=CRIT_LABELS[crit],
               edgecolor="white",
               linewidth=0.4,
               zorder=3)
        bottoms += heights

    # Threshold lines
    for yval, tlabel, tcolor in THRESHOLDS:
        ax.axhline(y=yval, color=tcolor, linestyle="--",
                   linewidth=0.9, zorder=4, alpha=0.85)
        ax.text(n - 0.1, yval + 0.18, tlabel,
                ha="right", va="bottom",
                fontsize=7, color=tcolor, style="italic", zorder=5)

    # Axes
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, ha="center", linespacing=1.3)
    ax.set_ylabel("ACMG/AMP evidence points", fontsize=8, fontweight="semibold")
    ax.set_ylim(0, 13)
    ax.set_xlim(-0.6, n - 0.4)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.grid(axis="y", alpha=0.15, linewidth=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    patches = [
        mpatches.Patch(facecolor=CRIT_COLORS[k], edgecolor="none",
                       label=CRIT_LABELS[k])
        for k in ["PVS1", "PS4", "PP3", "PM2"]
    ]
    ax.legend(handles=patches,
              loc="upper right",
              fontsize=7,
              frameon=False,
              title="ACMG/AMP criterion",
              title_fontsize=7.5,
              handlelength=1.2,
              handletextpad=0.5,
              labelspacing=0.35)

    plt.tight_layout(pad=0.8)
    save_figure(fig, OUT_STEM)
    print(f"Saved → {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
