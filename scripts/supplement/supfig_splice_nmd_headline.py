#!/usr/bin/env python3
"""
Supplementary Figure — EVEE RNA-seq validation summary.

Bar chart of validation rate by mechanism class (canonical splice, intronic
splice, NMD) plus zygosity stratification (Hom/LOH vs Het) across all tiers.

Source: evee_resubmission_natgen/supp_table_validation_per_variant.csv
        (one row per unique variant; "Validated" boolean per row).

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
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from theme.theme import apply_theme, save_figure, CONSEQ_COLORS, COLORS

apply_theme()

CSV     = ROOT / "evee_resubmission_natgen" / "supp_table_validation_per_variant.csv"
OUT_STEM = ROOT / "figures" / "supplement" / "supfig_splice_nmd_headline"


def rate(sub: pd.DataFrame) -> tuple[int, int, float]:
    n = int(sub["Validated"].sum())
    d = len(sub)
    return n, d, (n / d * 100 if d else 0)


def main():
    df = pd.read_csv(CSV)
    df["Validated"] = df["Validated"].astype(bool)

    canon    = df[df["tier"] == "Tier 1"]
    intronic = df[df["tier"] == "Tier 2"]
    nmd      = df[df["tier"] == "Tier 3"]
    hom      = df[df["Zygosity"] == "Hom/LOH"]
    het      = df[df["Zygosity"] == "Het"]

    groups = [
        ("Canonical\nsplice", canon,    CONSEQ_COLORS["Splice"],    None),
        ("Intronic",          intronic, CONSEQ_COLORS["Intronic"],  "all Hom/LOH"),
        ("NMD",               nmd,      CONSEQ_COLORS["Nonsense"],  None),
        ("Hom/LOH",           hom,      COLORS["gf_orange"],        None),
        ("Het",               het,      COLORS["steel"],            None),
    ]

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    xs = np.arange(len(groups))

    for xi, (label, sub, col, note) in enumerate(groups):
        n, d, pct = rate(sub)
        ax.bar(xi, pct, color=col, alpha=0.9, width=0.65,
               edgecolor="white", lw=0.5)
        annotation = f"{n}/{d}\n({pct:.0f}%)"
        if note:
            annotation = f"{n}/{d}\n({pct:.0f}%)\n{note}"
        ax.text(xi, pct + 2, annotation,
                ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=col)

    # Separator between mechanism-class group and zygosity group
    ax.axvline(2.5, color="#cccccc", lw=0.5, ls=":")
    ax.text(1.0, 128, "by mechanism", ha="center", fontsize=7,
            color="#666666", style="italic")
    ax.text(3.5, 128, "by zygosity", ha="center", fontsize=7,
            color="#666666", style="italic")

    ax.axhline(100, color="#bbbbbb", lw=0.5, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax.set_ylim(0, 135)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Validated (%)", fontsize=9)
    ax.set_title("EVEE RNA-seq validation", fontsize=10, fontweight="semibold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"[wrote] {OUT_STEM}.png / .pdf")
    for label, sub, _, _ in groups:
        n, d, pct = rate(sub)
        print(f"  {label.replace(chr(10),' '):<20s} {n}/{d} = {pct:.0f}%")


if __name__ == "__main__":
    main()
