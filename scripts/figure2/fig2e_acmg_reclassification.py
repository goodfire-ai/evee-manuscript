#!/usr/bin/env python3
"""
Fig 2E — ACMG/AMP evidence reclassification of FinnGen R12 EVEE candidates.

Stacked bar chart of ACMG/AMP evidence points for 6 FinnGen R12 ClinVar
resubmission candidates, colored by criterion (PVS1, PS4, PP3, PM2).
Dashed threshold lines mark LP-lean (4), LP (6), and P (10) point thresholds.

Adapted from evee_resubmission_natgen/fig_acmg_stacked.py,
re-routed to figures/figure2/.

Input:  /mnt/home/ryo/finngen_r12_actionable.tsv
Output: figures/figure2/fig2e_acmg_reclassification.{png,pdf}
"""
import re
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

REPO   = Path(__file__).resolve().parents[2]
DATA   = Path("/mnt/home/ryo/finngen_r12_actionable.tsv")
OUT_STEM = REPO / "figures" / "figure2" / "fig2e_acmg_reclassification"

sys.path.insert(0, str(REPO))
from theme.theme import apply_theme, save_figure, COLORS

apply_theme()
OUT_STEM.parent.mkdir(parents=True, exist_ok=True)

# ── colors ────────────────────────────────────────────────────────────────────
CRIT_COLORS = {
    "PVS1": COLORS["crimson"],
    "PS4":  COLORS["steel"],
    "PP3":  COLORS["gf_orange"],
    "PM2":  COLORS["sage"],
}
CRIT_LABELS = {
    "PVS1": "PVS1 (null variant)",
    "PS4":  "PS4 (case-control OR)",
    "PP3":  "PP3 (EVEE/Evo2 score)",
    "PM2":  "PM2 (Finnish AF < 0.1%)",
}

THRESHOLDS = [
    (4,  "LP-lean", "#999999"),
    (6,  "LP",      "#555555"),
    (10, "P",       "#222222"),
]

RESUBMISSION_IDS = {
    "chr4:87845065:GGAAA:G",
    "chr11:46728029:C:G",
    "chr11:46726097:G:T",
    "chr8:54621480:T:G",
    "chr19:50276162:G:A",
    "chr2:178527204:A:T",
}


def parse_pts(val, col: str) -> float:
    if pd.isna(val):
        return 0.0
    v = str(val).strip()
    if not v or v in ("nan", "ND"):
        return 0.0
    try:
        return float(v)
    except ValueError:
        pass
    m = re.search(r"\((\d+(?:\.\d+)?)\)", v)
    if m:
        return float(m.group(1))
    if re.search(r"none|het_flag|blocked|not_applied", v, re.I):
        return 0.0
    if col == "ACMG_PS4":
        if re.search(r"strong",     v, re.I): return 4.0
        if re.search(r"moderate",   v, re.I): return 2.0
        if re.search(r"supporting", v, re.I): return 1.0
    if col == "ACMG_PP3" and re.search(r"moderate", v, re.I):
        return 1.0
    if col == "ACMG_PM2" and re.search(r"supporting", v, re.I):
        return 1.0
    if col == "ACMG_PVS1" and re.search(r"pvs1", v, re.I):
        m2 = re.search(r"(\d+)", v)
        return float(m2.group(1)) if m2 else 8.0
    return 0.0


def make_label(row) -> str:
    gene = row["gene"]
    hgvs = str(row["HGVS"])
    m = re.search(r"p\.[A-Za-z0-9*]+(?:Ter\d+|fs\w*)?", hgvs)
    if m:
        prot = m.group(0)
        if len(prot) > 16:
            prot = prot[:15] + "…"
    else:
        vid = row["variant_id"]
        parts = vid.split(":")
        prot = f":{parts[1]}"
    return f"{gene}\n{prot}"


def main():
    df = pd.read_csv(DATA, sep="\t")

    for col in ["ACMG_PVS1", "ACMG_PS4", "ACMG_PP3", "ACMG_PM2"]:
        df[f"{col}_pts"] = df[col].apply(lambda x, c=col: parse_pts(x, c))

    df["total_pts"] = df[["ACMG_PVS1_pts", "ACMG_PS4_pts",
                           "ACMG_PP3_pts",  "ACMG_PM2_pts"]].sum(axis=1)
    df["is_resubmit"] = df["variant_id"].isin(RESUBMISSION_IDS)
    df = df[df["is_resubmit"]].sort_values("total_pts", ascending=False)
    df = df.reset_index(drop=True)

    df["label"] = df.apply(make_label, axis=1)
    df.loc[df["variant_id"] == "chr11:46726097:G:T", "label"] = "F2\nchr11:46726097"

    n = len(df)
    x = np.arange(n)
    bar_width = 0.55

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    bottoms = np.zeros(n)
    for crit in ["PVS1", "PS4", "PP3", "PM2"]:
        heights = df[f"ACMG_{crit}_pts"].values.astype(float)
        ax.bar(x, heights, bar_width,
               bottom=bottoms,
               color=CRIT_COLORS[crit],
               label=CRIT_LABELS[crit],
               edgecolor="white",
               linewidth=0.4,
               zorder=3)
        bottoms += heights

    for yval, tlabel, tcolor in THRESHOLDS:
        ax.axhline(y=yval, color=tcolor, linestyle="--",
                   linewidth=0.9, zorder=4, alpha=0.8)
        ax.text(n - 0.1, yval + 0.18, tlabel,
                ha="right", va="bottom",
                fontsize=7, color=tcolor, style="italic", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], fontsize=7, ha="center", linespacing=1.3)
    ax.set_ylabel("ACMG/AMP evidence points", fontsize=8, fontweight="semibold")
    ax.set_ylim(0, 13)
    ax.set_xlim(-0.65, n - 0.35)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
    ax.grid(axis="y", alpha=0.15, linewidth=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_patches = [
        mpatches.Patch(facecolor=CRIT_COLORS[k], edgecolor="none",
                       label=CRIT_LABELS[k])
        for k in ["PVS1", "PS4", "PP3", "PM2"]
    ]
    ax.legend(handles=legend_patches,
              loc="upper right", fontsize=7, frameon=False,
              title="ACMG/AMP criterion", title_fontsize=7.5,
              handlelength=1.2, handletextpad=0.5, labelspacing=0.35)

    plt.tight_layout(pad=0.8)
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")
    print(f"  {n} resubmission candidates")


if __name__ == "__main__":
    main()
