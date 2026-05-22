#!/usr/bin/env python3
"""
fig_acmg_stacked.py
-------------------
Stacked bar chart of ACMG/AMP evidence points for FinnGen R12 EVEE
resubmission candidates.

Outputs:
    fig_acmg_stacked.png   (300 dpi, for submission)
    fig_acmg_stacked.svg   (vector, for editing)

Run from any directory:
    python3 /mnt/home/ryo/evee-manuscript/evee_resubmission_natgen/fig_acmg_stacked.py
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

# ── paths ────────────────────────────────────────────────────────────────────
REPO      = Path("/mnt/home/ryo/evee-manuscript")
OUTDIR    = REPO / "evee_resubmission_natgen"
DATA      = Path("/mnt/home/ryo/finngen_r12_actionable.tsv")

sys.path.insert(0, str(REPO))
from theme.theme import apply_theme, save_figure, COLORS

apply_theme()

# ── ACMG criterion colors ─────────────────────────────────────────────────────
CRIT_COLORS = {
    "PVS1": COLORS["crimson"],    # strong pathogenic — red
    "PS4":  COLORS["steel"],      # case-control — blue
    "PP3":  COLORS["gf_orange"],  # computational — Goodfire orange
    "PM2":  COLORS["sage"],       # population freq — green
}

CRIT_LABELS = {
    "PVS1": "PVS1 (null variant)",
    "PS4":  "PS4 (case-control OR)",
    "PP3":  "PP3 (EVEE/Evo2 score)",
    "PM2":  "PM2 (Finnish AF < 0.1%)",
}

# ── classification thresholds ─────────────────────────────────────────────────
THRESHOLDS = [
    (4,  "LP-lean", "#999999"),
    (6,  "LP",      "#555555"),
    (10, "P",       "#222222"),
]

# ── 6 resubmission candidates (variant_id) ────────────────────────────────────
RESUBMISSION_IDS = {
    "chr4:87845065:GGAAA:G",   # MEPE
    "chr11:46728029:C:G",       # F2 p.Leu389Val
    "chr11:46726097:G:T",       # F2 chr11 missense
    "chr8:54621480:T:G",        # RP1
    "chr19:50276162:G:A",       # MYH14
    "chr2:178527204:A:T",       # TTN
}

# ── parse ACMG numeric points from mixed text/numeric cells ───────────────────
def parse_pts(val, col: str) -> float:
    if pd.isna(val):
        return 0.0
    v = str(val).strip()
    if not v or v in ("nan", "ND"):
        return 0.0
    # Direct numeric
    try:
        return float(v)
    except ValueError:
        pass
    # Number in parentheses: PS4_strong(4) → 4, PS4_moderate(2) → 2
    m = re.search(r"\((\d+(?:\.\d+)?)\)", v)
    if m:
        return float(m.group(1))
    # Blocked / none → 0
    if re.search(r"none|het_flag|blocked|not_applied", v, re.I):
        return 0.0
    # Text-only fallbacks
    if col == "ACMG_PS4":
        if re.search(r"strong",    v, re.I): return 4.0
        if re.search(r"moderate",  v, re.I): return 2.0
        if re.search(r"supporting",v, re.I): return 1.0
    if col == "ACMG_PP3" and re.search(r"moderate", v, re.I):
        return 1.0
    if col == "ACMG_PM2" and re.search(r"supporting", v, re.I):
        return 1.0
    if col == "ACMG_PVS1" and re.search(r"pvs1", v, re.I):
        m2 = re.search(r"(\d+)", v)
        return float(m2.group(1)) if m2 else 8.0
    return 0.0


# ── short x-axis labels ───────────────────────────────────────────────────────
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
        prot = f":{parts[1]}"           # just the position
    return f"{gene}\n{prot}"


# ── load & process ────────────────────────────────────────────────────────────
df = pd.read_csv(DATA, sep="\t")

for col in ["ACMG_PVS1", "ACMG_PS4", "ACMG_PP3", "ACMG_PM2"]:
    df[f"{col}_pts"] = df[col].apply(lambda x: parse_pts(x, col))

df["total_pts"] = df[["ACMG_PVS1_pts", "ACMG_PS4_pts",
                        "ACMG_PP3_pts", "ACMG_PM2_pts"]].sum(axis=1)

# Sort: resubmission candidates first (by score), then remaining by score
df["is_resubmit"] = df["variant_id"].isin(RESUBMISSION_IDS)
df = df[df["is_resubmit"]].sort_values("total_pts", ascending=False)
df = df.reset_index(drop=True)

df["label"] = df.apply(make_label, axis=1)
# Override label for the F2 variant with unresolved HGVS
df.loc[df["variant_id"] == "chr11:46726097:G:T", "label"] = "F2\nchr11:46726097"

n = len(df)
x = np.arange(n)
bar_width = 0.55

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.2))

bottoms = np.zeros(n)
for crit in ["PVS1", "PS4", "PP3", "PM2"]:
    col = f"ACMG_{crit}_pts"
    heights = df[col].values.astype(float)
    ax.bar(x, heights, bar_width,
           bottom=bottoms,
           color=CRIT_COLORS[crit],
           label=CRIT_LABELS[crit],
           edgecolor="white",
           linewidth=0.4,
           zorder=3)
    bottoms += heights

# Threshold dashed lines
for yval, tlabel, tcolor in THRESHOLDS:
    ax.axhline(y=yval, color=tcolor, linestyle="--",
               linewidth=0.9, zorder=4, alpha=0.8)
    ax.text(n - 0.1, yval + 0.18, tlabel,
            ha="right", va="bottom",
            fontsize=7, color=tcolor, style="italic", zorder=5)

n_resub = len(df)

# Axes formatting
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
ax.spines["bottom"].set_linewidth(0.6)
ax.spines["left"].set_linewidth(0.6)

# Legend
legend_patches = [
    mpatches.Patch(facecolor=CRIT_COLORS[k], edgecolor="none",
                   label=CRIT_LABELS[k])
    for k in ["PVS1", "PS4", "PP3", "PM2"]
]
ax.legend(handles=legend_patches,
          loc="upper right",
          fontsize=7,
          frameon=False,
          title="ACMG/AMP criterion",
          title_fontsize=7.5,
          handlelength=1.2,
          handletextpad=0.5,
          labelspacing=0.35)

plt.tight_layout(pad=0.8)

# ── save ──────────────────────────────────────────────────────────────────────
stem = OUTDIR / "fig_acmg_stacked"
fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
plt.close(fig)

print(f"Saved → {stem}.png / .svg")
print(f"  {n_resub} resubmission candidates  |  {n - n_resub} additional actionable variants")
pts_check = df[["gene", "label", "ACMG_PVS1_pts", "ACMG_PS4_pts",
                 "ACMG_PP3_pts", "ACMG_PM2_pts", "total_pts",
                 "is_resubmit"]].to_string(index=False)
print("\nParsed ACMG points:\n" + pts_check)
