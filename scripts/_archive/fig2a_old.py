#!/usr/bin/env python3
"""
Figure 2A — K-Sparse Probes: performance vs number of SAE features.

Shows AUROC, F1, MCC, and Accuracy for the best SAE configuration
on the ClinVar labeled (211K) dataset, with embedding baseline.

Input:  data/panels/fig2a.csv
Output: figures/figure2/panels/fig2a.{png,pdf}
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
from theme.mayo_theme import apply_theme, save_figure, COLORS, FONT_SIZE_TICK

PANELS = ROOT / "data" / "panels"
OUT_STEM = ROOT / "figures" / "figure2" / "panels" / "fig2a"

apply_theme()

# Metric colors from theme palette
C_AUROC = COLORS["gf_orange"]   # Warm orange — primary metric
C_F1    = COLORS["light_gray"]  # Light gray
C_MCC   = COLORS["gray"]        # Medium gray
C_ACC   = COLORS["gf_brown"]    # Warm taupe

METRIC_STYLE = {
    "auc_roc":  {"color": C_AUROC, "marker": "o", "ls": "-",  "label": "AUC-ROC", "zorder": 4},
    "f1":       {"color": C_F1,    "marker": "o", "ls": "-",  "label": "F1",       "zorder": 3},
    "mcc":      {"color": C_MCC,   "marker": "o", "ls": "-",  "label": "MCC",      "zorder": 2},
    "accuracy": {"color": C_ACC,   "marker": "o", "ls": "-",  "label": "Accuracy", "zorder": 1},
}


def plot_ksparse(df_probe, baseline_auc, title="K-Sparse Probes"):
    """
    Create k-sparse probe figure for one dataset.

    Parameters
    ----------
    df_probe : pl.DataFrame
        Rows with columns k, auc_roc, f1, accuracy, mcc.
    baseline_auc : float
        All-features embedding probe AUROC.
    title : str

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    ks = df_probe["k"].to_numpy()

    # Embedding baseline band
    ax.axhspan(baseline_auc - 0.005, baseline_auc + 0.005,
               color=COLORS["gf_beige"], alpha=0.5, zorder=0)
    ax.axhline(baseline_auc, color=COLORS["gray"], ls="--", lw=0.8, zorder=0)
    ax.text(ks[0], baseline_auc + 0.012,
            f"Evo2 probe (AUC={baseline_auc:.3f})",
            fontsize=FONT_SIZE_TICK - 1, color="#666666", va="bottom")

    # Plot each metric
    for metric, style in METRIC_STYLE.items():
        vals = df_probe[metric].to_numpy()
        ax.plot(ks, vals, marker=style["marker"], ls=style["ls"],
                color=style["color"], label=style["label"],
                markersize=5, lw=1.4, zorder=style["zorder"])

    # Annotate AUC values just above each data point
    auroc_vals = df_probe["auc_roc"].to_numpy()
    annot_fs = FONT_SIZE_TICK - 2
    for k, auc in zip(ks, auroc_vals):
        if auc > baseline_auc - 0.025:
            ax.annotate(f"{auc:.3f}", xy=(k, auc), xytext=(0, -8),
                        textcoords="offset points", fontsize=annot_fs,
                        color=C_AUROC, ha="center", va="top")
        else:
            ax.annotate(f"{auc:.3f}", xy=(k, auc), xytext=(0, 5),
                        textcoords="offset points", fontsize=annot_fs,
                        color=C_AUROC, ha="center", va="bottom")

    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Number of SAE features (k)")
    ax.set_ylabel("Score")
    ax.set_title(title, fontweight="bold")

    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.15, lw=0.5)

    # Y-axis range
    ax.set_ylim(0.5, 1.0)

    fig.tight_layout()
    return fig


def main():
    df = pl.read_csv(PANELS / "fig2a.csv")

    # Use deconfounded (170K) dataset — best SAE
    df_deconf = df.filter(
        (pl.col("dataset") == "deconfounded") & (pl.col("sae_name") != "all_features")
    )
    baseline_deconf = df.filter(
        (pl.col("dataset") == "deconfounded") & (pl.col("sae_name") == "all_features")
    )[0, "auc_roc"]

    fig = plot_ksparse(df_deconf, baseline_deconf, title="K-Sparse Probes")
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
