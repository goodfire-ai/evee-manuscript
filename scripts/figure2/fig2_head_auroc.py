#!/usr/bin/env python3
"""
Figure 2 panel — Token probe AUROC by annotation category.

Horizontal boxplot with jittered scatter of per-head AUROC values,
grouped by annotation category from artifacts/heads.feather.
Binary heads only (236 heads); categorical heads lack AUROC.

Input:  website_probes/v5/token/eval.json + artifacts/heads.feather
Output: figures/figure2/panels/fig2_head_auroc.{png,pdf}
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from theme.mayo_theme import (
    apply_theme, save_figure,
    FONT_SIZE_TITLE, FONT_SIZE_LABEL, FONT_SIZE_TICK,
)

EVAL_JSON = ROOT / "artifacts" / "token_eval.json"
HEADS_FEATHER = ROOT / "artifacts" / "heads.feather"
OUT_STEM = ROOT / "figures" / "figure2" / "panels" / "fig2_head_auroc"

# Merge fine-grained groups into 7 display categories
MERGE_MAP = {
    "Region":         "Sequence & Region",
    "Conservation":   "Sequence & Region",
    "Splicing":       "Splicing",
    "ChIP-seq":       "Regulatory",
    "ATAC-seq":       "Regulatory",
    "Regulatory":     "Regulatory",
    "InterPro":       "InterPro",
    "Protein":        "Protein & Structure",
    "Structure":      "Protein & Structure",
    "PTM":            "Protein & Structure",
    "ELM Motifs":     "Substitution & Motif",
    "Variant Effect":  "Substitution & Motif",
}

GROUP_COLORS = {
    "Sequence & Region":    "#C44E52",
    "Splicing":             "#DA8BC3",
    "Regulatory":           "#4C72B0",
    "InterPro":             "#DD8452",
    "Protein & Structure":  "#55A868",
    "Substitution & Motif": "#8172B3",
    "Other":                "#A0A0A0",
}


def plot(ax):
    eval_data = json.loads(EVAL_JSON.read_text())
    heads_df = pl.read_ipc(HEADS_FEATHER)
    head_groups = dict(zip(heads_df["head"].to_list(), heads_df["group"].to_list()))

    by_group: dict[str, list[float]] = defaultdict(list)
    for h, m in eval_data.items():
        if m.get("kind") != "binary":
            continue
        auc = m.get("auc")
        if auc is None:
            continue
        raw_group = head_groups.get(h, "Other")
        if raw_group == "Pathogenicity":
            continue
        group = MERGE_MAP.get(raw_group, "Other")
        by_group[group].append(auc)

    group_order = sorted(by_group, key=lambda g: np.median(by_group[g]))

    data, labels, counts, colors = [], [], [], []
    for g in group_order:
        vals = by_group[g]
        data.append(vals)
        labels.append(g)
        counts.append(len(vals))
        colors.append(GROUP_COLORS.get(g, "#888888"))

    positions = list(range(len(labels)))
    bp = ax.boxplot(
        data, positions=positions, vert=False, widths=0.55,
        patch_artist=True, showfliers=False,
        boxprops=dict(linewidth=0.8),
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color + "55")
        patch.set_edgecolor(color)

    rng = np.random.default_rng(42)
    for i, (vals, color) in enumerate(zip(data, colors)):
        jitter = rng.uniform(-0.22, 0.22, size=len(vals))
        ax.scatter(vals, np.array([i] * len(vals)) + jitter, s=10, alpha=0.45,
                   color=color, edgecolors="none", zorder=3)

    ax.set_yticks(positions)
    ax.set_yticklabels([])
    for i, (label, count) in enumerate(zip(labels, counts)):
        ax.text(-0.02, positions[i] - 0.01, label, transform=ax.get_yaxis_transform(),
                ha="right", va="center",
                fontsize=FONT_SIZE_TICK, fontweight="semibold")
        ax.text(-0.02, positions[i] - 0.21, f"(n={count})", transform=ax.get_yaxis_transform(),
                ha="right", va="center",
                fontsize=FONT_SIZE_TICK - 1, fontweight="normal", color="#444444")

    ax.set_xlabel("AUROC", fontsize=FONT_SIZE_LABEL)
    ax.axvline(0.5, color="grey", linestyle="--", alpha=0.3, linewidth=0.6)
    ax.set_xlim(0.48, 1.02)
    ax.grid(axis="x", alpha=0.2)
    ax.set_title("Probe AUROC by Annotation Category", fontsize=FONT_SIZE_TITLE)


def main():
    apply_theme()
    fig, ax = plt.subplots(figsize=(6, 5))
    plot(ax)
    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
