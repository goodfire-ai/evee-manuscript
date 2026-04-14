#!/usr/bin/env python3
"""
Supplementary Figure — Context window sweep.

A. Per-consequence AUROC vs context window (genomic where available,
   falls back to gene-clamped below 8k where deltas are negligible).
B. AUROC gain from genomic context (genomic − clamped) per consequence.

Input:  artifacts/context_window_sweep.feather
Output: figures/supplement/supfig2_context_window.{png,pdf}
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
from theme.theme import apply_theme, save_figure, COLORS, CONSEQ_COLORS

ARTIFACTS = ROOT / "artifacts"
OUT_STEM = ROOT / "figures" / "supplement" / "supfig2_context_window"

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
    ax.set_xticklabels([f"{w // 1000}k" if w >= 1024 else str(w) for w in windows],
                       rotation=45, ha="right")
    ax.set_xlabel("Context window (bp)")


def plot_per_consequence(ax, df):
    """A. Overall (gray) + per-consequence AUROC curves."""
    windows = df.filter(pl.col("consequence") == "overall")["window"].to_list()

    # Use genomic where available, fall back to clamped
    auroc = pl.col("auroc_genomic").fill_null(pl.col("auroc_clamped"))

    for csq, style in [("overall", dict(marker="o", markersize=5, linewidth=2.5,
                                         color=COLORS["gray"], label="Overall", zorder=10)),
                        *[(c, dict(marker="o", markersize=3, linewidth=1.2,
                                   color=CSQ_COLORS[c], label=c.capitalize(), alpha=0.8))
                          for c in CSQ_ORDER]]:
        vals = df.filter(pl.col("consequence") == csq).with_columns(auroc.alias("y"))["y"].to_list()
        ax.plot(windows, vals, **style)

    _format_xticks(ax, windows)
    ax.set_title("A. Context size sweep (unclamped)", fontweight="semibold")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=6, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 0.99))


def plot_auroc_diff(ax, df):
    """B. AUROC gain from genomic context per consequence."""
    # Only windows where ALL consequences have genomic data
    has_genomic = (
        df.group_by("window")
        .agg(pl.col("auroc_genomic").null_count().alias("nulls"))
        .filter(pl.col("nulls") == 0)["window"].to_list()
    )
    diff = (
        df.filter(pl.col("window").is_in(has_genomic))
        .with_columns((pl.col("auroc_genomic") - pl.col("auroc_clamped")).alias("delta"))
    )
    windows = diff.filter(pl.col("consequence") == "overall").sort("window")["window"].to_list()
    x = np.arange(len(windows))

    all_csqs = ["overall"] + CSQ_ORDER
    all_colors = {**CSQ_COLORS, "overall": COLORS["gray"]}
    all_labels = {**{c: c.capitalize() for c in CSQ_ORDER}, "overall": "Overall"}
    width = 0.8 / len(all_csqs)

    for j, csq in enumerate(all_csqs):
        vals = diff.filter(pl.col("consequence") == csq)["delta"].to_list()
        ax.bar(x + j * width - 0.4 + width / 2, vals, width,
               color=all_colors[csq], label=all_labels[csq])

    ax.axhline(0, color=COLORS["gray"], linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{w // 1000}k" for w in windows])
    ax.set_xlabel("Context window (bp)")
    ax.set_ylabel("\u0394AUROC (genomic \u2212 clamped)")

    ymax = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1])) * 1.15
    ax.set_ylim(-ymax, ymax)
    ax.set_title("B. Gene clamping versus unclamped", fontweight="semibold")
    ax.legend(fontsize=6, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 0.99))


def main():
    df = pl.read_ipc(ARTIFACTS / "context_window_sweep.feather")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    plot_per_consequence(ax1, df)
    plot_auroc_diff(ax2, df)

    fig.tight_layout(w_pad=3)
    OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
