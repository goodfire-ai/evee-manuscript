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
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

N_BOOT = 2000
BOOT_SEED = 0

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


def _bootstrap_roc(y, s, n_boot=N_BOOT, seed=BOOT_SEED):
    """Stratified bootstrap (resample positives and negatives independently).

    Returns the AUC 95% CI plus a 95% TPR band interpolated on a common FPR grid.
    """
    y = np.asarray(y)
    s = np.asarray(s)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng = np.random.RandomState(seed)
    grid = np.linspace(0.0, 1.0, 101)
    aucs = np.empty(n_boot)
    tprs = np.empty((n_boot, grid.size))
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, pos.size, replace=True),
                              rng.choice(neg, neg.size, replace=True)])
        yb, sb = y[idx], s[idx]
        aucs[b] = roc_auc_score(yb, sb)
        fpr, tpr, _ = roc_curve(yb, sb)
        tprs[b] = np.interp(grid, fpr, tpr)
        tprs[b, 0] = 0.0
    auc_lo, auc_hi = np.percentile(aucs, [2.5, 97.5])
    tpr_lo = np.percentile(tprs, 2.5, axis=0)
    tpr_hi = np.percentile(tprs, 97.5, axis=0)
    return (float(auc_lo), float(auc_hi)), grid, tpr_lo, tpr_hi


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
        (auc_lo, auc_hi), grid, tpr_lo, tpr_hi = _bootstrap_roc(
            sub[label_col].values, sub[col].values)
        drawn.append({"label": label, "color": color, "auc": float(auc),
                      "auc_lo": auc_lo, "auc_hi": auc_hi, "fpr": fpr, "tpr": tpr,
                      "grid": grid, "tpr_lo": tpr_lo, "tpr_hi": tpr_hi})
    drawn.sort(key=lambda d: -d["auc"])
    # 95% bootstrap ROC bands (all methods), drawn beneath the curves
    for d in drawn:
        ax.fill_between(d["grid"], d["tpr_lo"], d["tpr_hi"],
                        color=d["color"], alpha=0.13, lw=0, zorder=1)
    for d in drawn:
        ax.plot(d["fpr"], d["tpr"], color=d["color"], lw=2.2, zorder=3)
    ax.plot([0, 1], [0, 1], color="#cccccc", lw=0.8, ls="--", zorder=2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=FONT_SIZE_LABEL + 2,
                  fontweight="semibold")
    ax.set_ylabel("True Positive Rate", fontsize=FONT_SIZE_LABEL + 2,
                  fontweight="semibold")
    ax.text(0.02, 0.97, f"n+ = {n_pos},  n− = {n_neg}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=FONT_SIZE_TICK + 2, color="#555555")
    ax.tick_params(labelsize=FONT_SIZE_TICK + 2)

    # Two-column legend: method names (with colored handle) on the left,
    # AUC [95% CI] aligned in their own column on the right.
    blank = plt.Line2D([0], [0], color="none")
    left_handles = [plt.Line2D([0], [0], color=d["color"], lw=2.2) for d in drawn]
    left_labels = [d["label"] for d in drawn]
    left_handles.append(plt.Line2D([0], [0], color="#cccccc", lw=0.8, ls="--"))
    left_labels.append("chance")
    right_handles = [blank] * (len(drawn) + 1)
    right_labels = [f"AUC = {d['auc']:.2f} [{d['auc_lo']:.2f}–{d['auc_hi']:.2f}]"
                    for d in drawn] + [""]
    ax.legend(left_handles + right_handles, left_labels + right_labels,
              ncol=2, fontsize=FONT_SIZE_LEGEND + 2, frameon=False,
              loc="lower right", columnspacing=0.6, handletextpad=0.6,
              handlelength=1.4, labelspacing=0.4)
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
