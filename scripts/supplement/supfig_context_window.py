#!/usr/bin/env python3
"""
Supplementary Figure — Context window sweep.

Panel A: Overall (gray) + per-consequence AUROC vs context window size.
Panel B: Gene-clamped vs genomic overall AUROC.
Panel C: AUROC gain from genomic context (genomic − clamped) per consequence at large windows.

Input:  artifacts/context_window_per_consequence.feather
        artifacts/context_window_clamped_vs_genomic.feather
        artifacts/context_window_auroc_diff.feather
Output: figures/supplement/panels/supfig_context_window.{png,pdf}
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
from theme.mayo_theme import apply_theme, save_figure, add_panel_label, COLORS, CONSEQ_COLORS

ARTIFACTS = ROOT / "artifacts"
OUT_STEM = ROOT / "figures" / "supplement" / "panels" / "supfig_context_window"

CSQ_COLORS = {
    "missense":   CONSEQ_COLORS["Missense"],
    "nonsense":   CONSEQ_COLORS["Nonsense"],
    "splice":     CONSEQ_COLORS["Splice"],
    "synonymous": CONSEQ_COLORS["Synonymous"],
    "intron":     CONSEQ_COLORS["Intronic"],
}

CSQ_ORDER = ["missense", "nonsense", "splice", "synonymous", "intron"]

apply_theme()


def _format_xticks(ax, windows):
    ax.set_xscale("log", base=2)
    ax.set_xticks(windows)
    ax.set_xticklabels([f"{w // 1000}k" if w >= 1024 else str(w) for w in windows], rotation=45, ha="right")
    ax.set_xlabel("Context window (bp)")


def plot_per_consequence(ax, df):
    """Panel A: Overall (gray) + per-consequence AUROC."""
    windows = df["window"].to_list()
    ax.plot(windows, df["overall"], marker="o", markersize=5, linewidth=2.5, color=COLORS["gray"], label="Overall", zorder=10)
    for csq in CSQ_ORDER:
        ax.plot(windows, df[csq], marker="o", markersize=3, linewidth=1.2, color=CSQ_COLORS[csq], label=csq.capitalize(), alpha=0.8)
    _format_xticks(ax, windows)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=6, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))


def plot_clamped_vs_genomic(ax, df):
    """Panel B: Gene-clamped (beige) vs genomic (orange)."""
    windows = df["window"].to_list()
    ax.plot(windows, df["gene_clamped"], marker="o", markersize=5, linewidth=2, color=COLORS["gf_beige"], label="Gene-clamped")
    ax.plot(windows, df["genomic"], marker="o", markersize=5, linewidth=2, color=COLORS["gf_orange"], label="Genomic")
    _format_xticks(ax, windows)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=7, frameon=False, loc="lower right")


def plot_auroc_diff(ax, diff):
    """Panel B: AUROC gain from genomic context per consequence at large windows."""
    windows = diff["window"].to_list()
    x = np.arange(len(windows))
    all_cols = ["overall"] + CSQ_ORDER
    all_colors = {**CSQ_COLORS, "overall": COLORS["gray"]}
    all_labels = {**{c: c.capitalize() for c in CSQ_ORDER}, "overall": "Overall"}
    n = len(all_cols)
    width = 0.8 / n

    for j, col in enumerate(all_cols):
        ax.bar(x + j * width - 0.4 + width / 2, diff[col].to_list(), width,
               color=all_colors[col], label=all_labels[col])

    ax.axhline(0, color=COLORS["gray"], linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{w // 1000}k" for w in windows])
    ax.set_xlabel("Context window (bp)")
    ax.set_ylabel("ΔAUROC (genomic − clamped)")

    # Symmetric y-axis with 0 centered
    ymax = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1])) * 1.15
    ax.set_ylim(-ymax, ymax)

    ax.legend(fontsize=6, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))


def main():
    per_csq = pl.read_ipc(ARTIFACTS / "context_window_per_consequence.feather")
    clamped = pl.read_ipc(ARTIFACTS / "context_window_clamped_vs_genomic.feather")
    diff = pl.read_ipc(ARTIFACTS / "context_window_auroc_diff.feather")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    plot_per_consequence(ax1, per_csq)
    add_panel_label(ax1, "A")

    plot_auroc_diff(ax2, diff)
    add_panel_label(ax2, "B")

    fig.tight_layout(w_pad=3)
    OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
