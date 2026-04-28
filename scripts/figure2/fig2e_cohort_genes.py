#!/usr/bin/env python3
"""
Figure 2e — gene-level distribution of EVEE pathogenicity scores in the Mayo RA cohort.

Vertical (portrait) dot plot: pathogenicity score on y-axis, genes as columns.
Named pathway genes get individual columns; everything else collapses into a
single "Other" column. High-priority variants (≥0.75) drawn as saturated dots
with white edges; below-threshold variants as translucent dots.

Filter (matches the Mayo abstract numbers exactly):
  - source: handoff_final.parquet (299 rare variants, gnomAD AF ≤1%, 588 patients)
  - candidate_bin == "uncertain_conflicting_or_unannotated" (n=225)
  - EVEE pathogenicity ≥ 0.75 → 18 variants

Pathway groupings:
  - Immune regulation:           IFIH1, NOD2, IL10RA
  - Folate / methotrexate path:  ATIC, MTHFR, DHFR, TYMS
        DHFR is the direct MTX target; ATIC is inhibited by MTX-polyglutamates
        in the RA anti-inflammatory mechanism; MTHFR is a well-established
        MTX pharmacogene; TYMS is folate-dependent thymidylate synthesis
        studied in MTX pharmacogenomics.
  - Other:                       everything else (STAT1 etc.)

Input:  /mnt/data/artifacts/ryo/goodfire_handoff/handoff_final.parquet
Output: figures/figure2/fig2e_cohort_genes.{png,pdf}
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
from theme.theme import (
    apply_theme, save_figure,
    COLORS, FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_LEGEND,
)

DATA = Path("/mnt/data/artifacts/ryo/goodfire_handoff/handoff_final.parquet")
OUT_STEM = ROOT / "figures" / "figure2" / "fig2e_cohort_genes"

THRESHOLD = 0.75
UNCERTAIN_BIN = "uncertain_conflicting_or_unannotated"

INNATE_IMMUNE = ["IFIH1", "NOD2", "IL10RA"]
FOLATE_PURINE = ["ATIC", "MTHFR", "DHFR", "TYMS"]

COLOR_INNATE = "#6B5B95"      # muted purple
COLOR_MTX    = "#2A9D8F"      # muted teal
COLOR_OTHER  = COLORS["gray"]

GROUP_COLOR = {"innate": COLOR_INNATE, "folate": COLOR_MTX, "other": COLOR_OTHER}
GROUP_LABEL_LONG = {
    "innate": "Immune regulation",
    "folate": "Folate / methotrexate pathway",
    "other":  "Other",
}


def group_for(gene: str) -> str:
    if gene in INNATE_IMMUNE: return "innate"
    if gene in FOLATE_PURINE: return "folate"
    return "other"


def load_uncertain() -> pl.DataFrame:
    df = pl.read_parquet(DATA, columns=["gene", "pathogenicity", "candidate_bin"])
    df = df.filter(pl.col("candidate_bin") == UNCERTAIN_BIN)
    df = df.with_columns(pl.col("gene")
                         .map_elements(group_for, return_dtype=pl.Utf8)
                         .alias("group"))
    return df.drop("candidate_bin")


def draw(ax, uncertain: pl.DataFrame):
    cols: list[dict] = []
    for grp_key, grp_label in (("innate", "Immune\nregulation"),
                               ("folate", "Folate /\nmethotrexate")):
        sub = uncertain.filter(pl.col("group") == grp_key)
        gene_stats = (sub.filter(pl.col("pathogenicity") >= THRESHOLD)
                      .group_by("gene")
                      .agg(n_hi=pl.len(),
                           mx=pl.col("pathogenicity").max())
                      .sort(["n_hi", "mx"], descending=[True, True]))
        for g in gene_stats["gene"].to_list():
            cols.append(dict(label=g, group=grp_key, italic=True,
                             group_label=grp_label))
    n_other_total = uncertain.filter(pl.col("group") == "other").height
    cols.append(dict(label="Other", group="other", italic=False,
                     group_label=f"Other\n(n={n_other_total})"))

    n = len(cols)
    xs = list(range(n))
    rng = np.random.default_rng(11)

    for x, c in zip(xs, cols):
        if c["group"] == "other":
            sub = uncertain.filter(pl.col("group") == "other")
        else:
            sub = uncertain.filter(pl.col("gene") == c["label"])
        scores = sub["pathogenicity"].to_numpy()
        is_hi = scores >= THRESHOLD
        jitter = rng.uniform(-0.22, 0.22, size=len(scores))
        ax.scatter((x + jitter)[~is_hi], scores[~is_hi], s=10,
                   color=GROUP_COLOR[c["group"]], alpha=0.30, edgecolors="none",
                   zorder=2)
        ax.scatter((x + jitter)[is_hi], scores[is_hi], s=28,
                   color=GROUP_COLOR[c["group"]], alpha=0.95,
                   edgecolors="white", linewidths=0.5, zorder=4)

    ax.axhline(THRESHOLD, color="#444444", linestyle="--", linewidth=0.6,
               alpha=0.5, zorder=1)

    ax.set_xticks(xs)
    ax.set_xticklabels([c["label"] for c in cols], fontsize=FONT_SIZE_TICK,
                       rotation=45, ha="right", rotation_mode="anchor")
    for tick, c in zip(ax.get_xticklabels(), cols):
        tick.set_color(GROUP_COLOR[c["group"]])
        if c["italic"]:
            tick.set_fontstyle("italic")
        if c["label"] == "IFIH1":
            tick.set_fontweight("bold")

    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(-0.04, 1.06)
    ax.set_yticks([0, 0.25, 0.5, THRESHOLD, 1.0])
    ax.set_ylabel("EVEE pathogenicity score", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK)
    ax.tick_params(axis="x", labelsize=FONT_SIZE_TICK)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.15, linewidth=0.4)
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.6)

    # Per-column tally above the plot area: n_hi/n_total
    for x, c in zip(xs, cols):
        if c["group"] == "other":
            sub = uncertain.filter(pl.col("group") == "other")
        else:
            sub = uncertain.filter(pl.col("gene") == c["label"])
        n_hi = int((sub["pathogenicity"] >= THRESHOLD).sum())
        n_total = len(sub)
        ax.text(x, 1.07, f"{n_hi}/{n_total}",
                ha="center", va="bottom",
                fontsize=FONT_SIZE_LEGEND - 0.5, color="#666666")

    # Pathway brackets above the named-pathway columns
    bracket_y = 1.15
    label_y = 1.18
    seen: set[str] = set()
    for x, c in zip(xs, cols):
        if c["group"] in seen or c["group"] == "other":
            continue
        seen.add(c["group"])
        members = [xx for xx, cc in zip(xs, cols) if cc["group"] == c["group"]]
        x_lo, x_hi = min(members) - 0.35, max(members) + 0.35
        ax.plot([x_lo, x_hi], [bracket_y, bracket_y],
                color=GROUP_COLOR[c["group"]], linewidth=1.0, clip_on=False,
                solid_capstyle="butt")
        ax.text((x_lo + x_hi) / 2, label_y, c["group_label"],
                ha="center", va="bottom",
                fontsize=FONT_SIZE_LEGEND, color=GROUP_COLOR[c["group"]],
                fontweight="semibold", clip_on=False, linespacing=1.05)


def main():
    apply_theme()
    uncertain = load_uncertain()
    fig, ax = plt.subplots(figsize=(3.8, 4.6))
    draw(ax, uncertain)
    fig.subplots_adjust(left=0.18, right=0.97, top=0.83, bottom=0.16)
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png/.pdf")


if __name__ == "__main__":
    main()
