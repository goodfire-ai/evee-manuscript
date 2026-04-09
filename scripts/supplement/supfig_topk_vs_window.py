#!/usr/bin/env python3
"""
Supplementary Figure — Top-K divergent vs contiguous window AUROC.

Horizontal grouped bar chart comparing per-consequence AUROC between
sparse top-K position selection (256 positions) and a dense contiguous
window (1001 positions). 95% bootstrap CIs shown as error bars.

Input:  artifacts/topk_vs_window.feather
Output: figures/supplement/panels/supfig_topk_vs_window.{png,pdf}
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
from theme.mayo_theme import apply_theme, save_figure, COLORS

ARTIFACTS = ROOT / "artifacts"
OUT_STEM = ROOT / "figures" / "supplement" / "panels" / "supfig_topk_vs_window"

CONSEQUENCE_ORDER = (
    "Overall", "Missense", "Intronic", "Synonymous", "Other",
    "Splice Donor", "Splice Acceptor", "5p UTR", "3p UTR", "Nonsense",
)

MODE_STYLE = {
    "topk":   {"color": COLORS["gf_orange"], "label": "Top-K (256 positions)"},
    "window": {"color": COLORS["steel"],     "label": "Contiguous window (1001 positions)"},
}

apply_theme()


def main():
    df = pl.read_ipc(ARTIFACTS / "topk_vs_window.feather")

    consequences = [c for c in CONSEQUENCE_ORDER if c in df["consequence"].to_list()]
    y = np.arange(len(consequences))
    bar_height = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, mode in enumerate(("topk", "window")):
        style = MODE_STYLE[mode]
        sub = df.filter(pl.col("mode") == mode)

        # Align to consequence order
        lookup = dict(zip(sub["consequence"].to_list(), range(sub.height)))
        vals = np.array([float(sub[lookup[c], "auroc"]) for c in consequences])
        lo = np.array([float(sub[lookup[c], "auroc_lo"]) for c in consequences])
        hi = np.array([float(sub[lookup[c], "auroc_hi"]) for c in consequences])

        offset = (i - 0.5) * bar_height
        ax.barh(y + offset, vals, height=bar_height,
                color=style["color"], label=style["label"],
                edgecolor="white", linewidth=0.3)
        ax.errorbar(vals, y + offset, xerr=[vals - lo, hi - vals],
                    fmt="none", color="black", capsize=2, lw=0.6)

    # Consequence labels with sample size
    n_lookup = dict(zip(
        df.filter(pl.col("mode") == "topk")["consequence"].to_list(),
        df.filter(pl.col("mode") == "topk")["n"].to_list(),
    ))
    labels = [f"{c}  (n={n_lookup[c]:,})" for c in consequences]

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("AUROC (gene-holdout test)")
    ax.axhline(0.5, color=COLORS["gray"], linewidth=0.3, alpha=0.5)  # separator after Overall
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.15)
    ax.legend(fontsize=7, frameon=False, loc="lower right")

    fig.tight_layout()
    OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
