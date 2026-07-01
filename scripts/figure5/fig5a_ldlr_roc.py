#!/usr/bin/env python3
"""
Figure 5a — LDLR severity ROC: EVEE vs AlphaMissense vs CADD.

ROC for discriminating any-FH (clinical + suspected) from presymptomatic
carriers in the LDLR clinical cohort. Positives = tier3 != presymptomatic_carrier.
Curves sorted by descending AUC; n+/n- annotated.

Input:  artifacts/ldlr_severity_per_variant.csv
Output: figures/figure5/fig5a_ldlr_roc.{png,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from theme.theme import (
    apply_theme, save_figure, COLORS,
    FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_LEGEND,
    FIG_WIDTH_DOUBLE,
)

ARTIFACTS = ROOT / "artifacts"
OUT_STEM = ROOT / "figures" / "figure5" / "fig5a_ldlr_roc"

apply_theme()

# 3-tier scale (positive = any FH; negative = presymptomatic carrier)
TIER_ORDER = ["clinical_FH", "suspected_FH", "presymptomatic_carrier"]

# Scores compared — colours locked to the paper-wide method palette
# (EVEE = gf_orange, AlphaMissense = sage, CADD = steel).
METHODS = [
    ("evee_pathogenicity", "EVEE",          COLORS["gf_orange"]),
    ("alphamissense",      "AlphaMissense", COLORS["sage"]),
    ("cadd",               "CADD",          COLORS["steel"]),
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(ARTIFACTS / "ldlr_severity_per_variant.csv")
    df = df[df["tier3"].isin(TIER_ORDER)].copy()
    df["is_any_FH"] = (df["tier3"] != "presymptomatic_carrier").astype(int)
    return df


def _roc_panel(ax, df, label_col):
    n_pos = int(df[label_col].sum())
    n_neg = len(df) - n_pos

    drawn = []
    for col, label, color in METHODS:
        sub = df.dropna(subset=[col])
        if len(sub) < 10 or sub[label_col].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(sub[label_col], sub[col])
        auc = roc_auc_score(sub[label_col], sub[col])
        drawn.append({"label": label, "color": color,
                      "auc": float(auc), "fpr": fpr, "tpr": tpr})
    drawn.sort(key=lambda d: -d["auc"])
    for d in drawn:
        ax.plot(d["fpr"], d["tpr"], color=d["color"], lw=2.2,
                label=f"{d['label']}   AUC = {d['auc']:.2f}")
    ax.plot([0, 1], [0, 1], color="#cccccc", lw=0.8, ls="--", label="chance")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate", fontsize=FONT_SIZE_LABEL + 2,
                  fontweight="semibold")
    ax.set_ylabel("True positive rate", fontsize=FONT_SIZE_LABEL + 2,
                  fontweight="semibold")
    ax.text(0.02, 0.97, f"n+ = {n_pos},  n− = {n_neg}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=FONT_SIZE_TICK + 2, color="#555555")
    ax.tick_params(labelsize=FONT_SIZE_TICK + 2)
    ax.legend(fontsize=FONT_SIZE_LEGEND + 2, frameon=False, loc="lower right")
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main():
    df = load_data()
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    _roc_panel(ax, df, "is_any_FH")
    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
