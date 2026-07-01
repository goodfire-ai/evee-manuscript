#!/usr/bin/env python3
"""
Figure 5b — LDLR severity violin: EVEE pathogenicity by clinical tier.

Per-tier violin (with jittered points) of EVEE pathogenicity across the 3-tier
FH-severity ordinal (clinical FH → suspected FH → presymptomatic carrier),
annotating per-tier medians.

Input:  artifacts/ldlr_severity_per_variant.csv
Output: figures/figure5/fig5b_ldlr_violin.{png,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from theme.theme import (
    apply_theme, save_figure,
    FONT_SIZE_LABEL, FONT_SIZE_TICK,
    FIG_WIDTH_DOUBLE,
)

ARTIFACTS = ROOT / "artifacts"
OUT_STEM = ROOT / "figures" / "figure5" / "fig5b_ldlr_violin"

apply_theme()

TIER_ORDER = ["clinical_FH", "suspected_FH", "presymptomatic_carrier"]
TIER_DISPLAY = {
    "clinical_FH":            "Clinical FH",
    "suspected_FH":           "Suspected FH",
    "presymptomatic_carrier": "Presymptomatic\nCarrier",
}
TIER_COLORS = {
    "clinical_FH":            "#a83232",
    "suspected_FH":           "#db8a48",
    "presymptomatic_carrier": "#7a9c8b",
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(ARTIFACTS / "ldlr_severity_per_variant.csv")
    return df[df["tier3"].isin(TIER_ORDER)].copy()


def _violin_panel(ax, joined: pd.DataFrame):
    rng = np.random.RandomState(0)
    n_per = {t: int((joined["tier3"] == t).sum()) for t in TIER_ORDER}

    for i, t in enumerate(TIER_ORDER):
        sub = joined[joined["tier3"] == t]
        vals = sub["evee_pathogenicity"].dropna().values
        if len(vals) >= 5:
            parts = ax.violinplot([vals], positions=[i], widths=0.55,
                                  showmeans=False, showmedians=False,
                                  showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(TIER_COLORS[t])
                pc.set_alpha(0.30)
                pc.set_edgecolor(TIER_COLORS[t])
        xs = i + (rng.rand(len(vals)) - 0.5) * 0.30
        ax.scatter(xs, vals, s=18, color=TIER_COLORS[t], alpha=0.80,
                   edgecolor="white", linewidth=0.4, zorder=3)
        if len(vals) >= 2:
            med = np.median(vals)
            ax.plot([i - 0.22, i + 0.22], [med, med],
                    color=TIER_COLORS[t], lw=2.4,
                    solid_capstyle="round", zorder=4)
            ax.text(i + 0.30, med, f"{med:.2f}",
                    ha="left", va="center", fontsize=FONT_SIZE_TICK + 2,
                    color="#222222", fontweight="semibold", zorder=5)

    ax.set_xticks(range(len(TIER_ORDER)))
    ax.set_xticklabels([])
    for i, t in enumerate(TIER_ORDER):
        name = TIER_DISPLAY[t]
        n_lines = name.count("\n") + 1
        ax.text(i, -0.04, name, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=FONT_SIZE_TICK + 2,
                fontweight="semibold")
        ax.text(i, -0.04 - 0.05 * n_lines, f"n = {n_per[t]}",
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=FONT_SIZE_TICK + 2)
    ax.set_ylabel("Predicted Pathogenicity",
                  fontsize=FONT_SIZE_LABEL + 2, fontweight="semibold")
    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK + 2)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="#cccccc", lw=0.6, ls="--", zorder=0)
    ax.grid(True, axis="y", lw=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main():
    df = load_data()
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    _violin_panel(ax, df)
    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
