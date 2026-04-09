#!/usr/bin/env python3
"""
Supplementary Figure — K-Sparse Probes: performance vs number of SAE features.

Shows AUROC, F1, MCC, and Accuracy for the best SAE configuration
on the ClinVar deconfounded dataset, with embedding baseline.
(Demoted from main Figure 2A in the figure overhaul.)

Input:  artifacts/fig2a.feather
Output: figures/supplement/supfig_ksparse.{png,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from theme.mayo_theme import apply_theme, save_figure, COLORS, FONT_SIZE_TICK

ARTIFACTS = ROOT / "artifacts"
OUT_STEM = ROOT / "figures" / "supplement" / "supfig_ksparse"

apply_theme()

C_AUROC = COLORS["gf_orange"]
C_F1    = COLORS["light_gray"]
C_MCC   = COLORS["gray"]
C_ACC   = COLORS["gf_brown"]

METRIC_STYLE = {
    "auc_roc":  {"color": C_AUROC, "marker": "o", "ls": "-",  "label": "AUC-ROC", "zorder": 4},
    "f1":       {"color": C_F1,    "marker": "o", "ls": "-",  "label": "F1",       "zorder": 3},
    "mcc":      {"color": C_MCC,   "marker": "o", "ls": "-",  "label": "MCC",      "zorder": 2},
    "accuracy": {"color": C_ACC,   "marker": "o", "ls": "-",  "label": "Accuracy", "zorder": 1},
}


def main():
    df = pl.read_ipc(ARTIFACTS / "fig2a.feather")

    df_deconf = df.filter(
        (pl.col("dataset") == "deconfounded") & (pl.col("sae_name") != "all_features")
    )
    baseline_deconf = df.filter(
        (pl.col("dataset") == "deconfounded") & (pl.col("sae_name") == "all_features")
    )[0, "auc_roc"]

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ks = df_deconf["k"].to_numpy()

    ax.axhspan(baseline_deconf - 0.005, baseline_deconf + 0.005,
               color=COLORS["gf_beige"], alpha=0.5, zorder=0)
    ax.axhline(baseline_deconf, color=COLORS["gray"], ls="--", lw=0.8, zorder=0)
    ax.text(ks[0], baseline_deconf + 0.012,
            f"Evo2 probe (AUC={baseline_deconf:.3f})",
            fontsize=FONT_SIZE_TICK - 1, color="#666666", va="bottom")

    for metric, style in METRIC_STYLE.items():
        vals = df_deconf[metric].to_numpy()
        ax.plot(ks, vals, marker=style["marker"], ls=style["ls"],
                color=style["color"], label=style["label"],
                markersize=5, lw=1.4, zorder=style["zorder"])

    auroc_vals = df_deconf["auc_roc"].to_numpy()
    annot_fs = FONT_SIZE_TICK - 2
    for k, auc in zip(ks, auroc_vals):
        if auc > baseline_deconf - 0.025:
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
    ax.set_title("K-Sparse Probes", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.15, lw=0.5)
    ax.set_ylim(0.5, 1.0)

    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
