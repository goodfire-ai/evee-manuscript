#!/usr/bin/env python3
"""
Figure 4a,b — BRCA1 rs80358099 splice-acceptor prediction validated by RNA-seq.

Multi-track browser figure: EVEE per-position disruption tracks (top) + gene model
(middle) + RNA-seq sashimi + synthetic coverage (bottom) for JHOS-4 (carrier,
VAF ≈ 1.0) and two wild-type controls.

Input:  artifacts/brca1_disruption_profile.feather
        artifacts/brca1_e22_data.json
Output: figures/figure4/fig4ab_brca1_rna.{png,pdf}

Run:
    uv run python scripts/figure4/fig4ab_brca1_rna.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from theme.theme import apply_theme, save_figure

apply_theme()

ARTIFACT_PROFILE = ROOT / "artifacts" / "brca1_disruption_profile.feather"
ARTIFACT_DATA    = ROOT / "artifacts" / "brca1_e22_data.json"
OUT_STEM         = ROOT / "figures" / "figure4" / "fig4ab_brca1_rna"

# Compressed-intron splice diagram axis (introns shrunk, exons 1:1)
SEGMENTS = [
    ("exon23",   43_049_121, 43_049_194,    0,  55),
    ("intron22", 43_049_195, 43_051_061,   55, 115),
    ("exon22",   43_051_062, 43_051_116,  115, 185),
    ("intron21", 43_051_117, 43_057_050,  185, 245),
    ("exon21",   43_057_051, 43_057_134,  245, 315),
]
DISPLAY_HI = SEGMENTS[-1][4]


def to_disp(pos):
    for _name, rlo, rhi, dlo, dhi in SEGMENTS:
        if rlo <= pos <= rhi:
            return dlo + (pos - rlo) * (dhi - dlo) / (rhi - rlo)
    return SEGMENTS[0][3] if pos < SEGMENTS[0][1] else SEGMENTS[-1][4]


TRACK_HEADS = [
    ("region_intron",           "Intron region"),
    ("is_splice_acceptor",      "Splice acceptor"),
    ("is_branchpoint_region",   "Branchpoint region"),
    ("is_polypyrimidine_tract", "Polypyrimidine tract"),
    ("is_exon_to_intron",       "Exon→intron boundary"),
    ("is_splice_donor",         "Splice donor"),
]

VARIANT_ID = "chr17:43051117:C:T"

EXONS = {
    "exon21": (43_057_051, 43_057_134),
    "exon22": (43_051_062, 43_051_116),
    "exon23": (43_049_121, 43_049_194),
}
NOTCH_LO, NOTCH_HI = 43_051_109, 43_051_116


def head_delta_per_position(full: pl.DataFrame, head: str) -> list[tuple[int, float]]:
    sub = full.select(["direction", "genomic_pos", f"ref_{head}", f"var_{head}"])
    sub = sub.with_columns((pl.col(f"var_{head}") - pl.col(f"ref_{head}")).alias("delta"))
    sub = sub.with_columns(pl.col("delta").abs().alias("abs_delta"))
    idx = sub.group_by("genomic_pos").agg(pl.col("abs_delta").max().alias("max_abs"))
    merged = (sub.join(idx, on="genomic_pos")
                 .filter(pl.col("abs_delta") == pl.col("max_abs"))
                 .sort(["genomic_pos", "direction"])
                 .unique(subset=["genomic_pos"], keep="first"))
    rows = merged.sort("genomic_pos").select(["genomic_pos", "delta"]).to_numpy()
    return [(int(p), float(d)) for p, d in rows]


def make_coverage(jr: dict) -> dict:
    cov = {ex: np.zeros(hi - lo + 1) for ex, (lo, hi) in EXONS.items()}
    canon = jr.get("canonical_e21e22", 0)
    cryptic = jr.get("cryptic_plus8", 0)
    skip = jr.get("full_skip", 0)
    e22e23 = jr.get("canonical_e22e23", 0)

    cov["exon21"][:] = canon + cryptic + skip

    rlo22 = EXONS["exon22"][0]
    for i, g in enumerate(range(rlo22, EXONS["exon22"][1] + 1)):
        cov["exon22"][i] = canon if NOTCH_LO <= g <= NOTCH_HI else canon + cryptic

    cov["exon23"][:] = e22e23 + skip
    return cov


def main(out_stem=None, fig_width=14):
    if out_stem is not None:
        global OUT_STEM
        OUT_STEM = out_stem
    with open(ARTIFACT_DATA) as f:
        data = json.load(f)

    full = pl.read_ipc(ARTIFACT_PROFILE).filter(pl.col("variant_id") == VARIANT_ID)
    print(f"  rows for {VARIANT_ID}: {full.height}")

    track_data = {h: head_delta_per_position(full, h) for h, _ in TRACK_HEADS}

    VARIANT_POS  = data["variant"]["pos"]
    VARIANT_DISP = to_disp(VARIANT_POS)
    HGVSC        = data["variant"]["hgvsc"]
    PATH         = float(data["variant"]["pathogenicity"])

    DISP_LO_POS = VARIANT_POS - 256
    DISP_HI_POS = VARIANT_POS + 256

    def bp_to_disp_width(p): return abs(to_disp(p + 1) - to_disp(p))

    SAMPLES = [
        {"label": "JHOS-4", "sub": "carrier (VAF≈1)", "color": "#a02c2c",
         "junctions": {"canonical_e21e22": 0, "cryptic_plus8": 25,
                       "full_skip": 24, "canonical_e22e23": 34}},
        {"label": "JHOS-2", "sub": "sister WT",       "color": "#2966a3",
         "junctions": {"canonical_e21e22": 185, "cryptic_plus8": 0,
                       "full_skip": 2, "canonical_e22e23": 190}},
        {"label": "A2780",  "sub": "WT control",       "color": "#444444",
         "junctions": {"canonical_e21e22": 248, "cryptic_plus8": 0,
                       "full_skip": 0, "canonical_e22e23": 294}},
    ]
    for s in SAMPLES:
        s["_cov"] = make_coverage(s["junctions"])
        s["_max_cov"] = max((a.max() for a in s["_cov"].values()), default=1) or 1

    JUNC_ENDS = {
        "canonical_e21e22": (to_disp(43_051_116), to_disp(43_057_051)),
        "cryptic_plus8":    (to_disp(43_051_108), to_disp(43_057_051)),
        "full_skip":        (to_disp(43_049_194), to_disp(43_057_051)),
        "canonical_e22e23": (to_disp(43_049_194), to_disp(43_051_062)),
    }
    JUNC_COLORS  = {"canonical_e21e22": "#666", "cryptic_plus8": "#d04646",
                    "full_skip": "#e89a2c", "canonical_e22e23": "#666"}
    JUNC_APEX    = {"canonical_e21e22": 0.30, "cryptic_plus8": 0.08,
                    "full_skip": 0.45, "canonical_e22e23": 0.20}

    N_TRACKS = len(TRACK_HEADS)
    H_TRACK  = 0.65
    H_GENE   = 1.05
    H_SPACER = 0.45
    H_RNA    = 1.4
    fig_h    = N_TRACKS * H_TRACK + H_GENE + H_SPACER + len(SAMPLES) * H_RNA + 1.2

    fig = plt.figure(figsize=(fig_width, fig_h))
    hr = [H_TRACK] * N_TRACKS + [H_GENE, H_SPACER] + [H_RNA] * len(SAMPLES)
    gs = fig.add_gridspec(N_TRACKS + 2 + len(SAMPLES), 1, height_ratios=hr, hspace=0.06)


    LABEL_X = -22

    # ── EVEE disruption tracks ──────────────────────────────────────────────
    for ti, (head, label) in enumerate(TRACK_HEADS):
        ax = fig.add_subplot(gs[ti])
        ax.set_xlim(LABEL_X - 5, DISPLAY_HI + 55)
        pts = track_data[head]

        track_max = max((abs(d) for _, d in pts), default=0.1) * 1.10 if pts else 1.0
        track_max = max(track_max, 0.1)
        ax.set_ylim(-track_max, track_max)

        positions, deltas, widths = [], [], []
        for x_pos, delta in pts:
            if abs(delta) < 0.02 or not (DISP_LO_POS <= x_pos <= DISP_HI_POS):
                continue
            positions.append(to_disp(x_pos))
            deltas.append(delta)
            widths.append(max(bp_to_disp_width(x_pos), 1.1))
        if positions:
            colors = ["#d04646" if d > 0 else "#2966a3" for d in deltas]
            ax.bar(positions, deltas, width=widths, color=colors, edgecolor="none",
                   zorder=3, alpha=0.95, align="center")

        ax.text(LABEL_X, 0, label, fontsize=16, ha="right", va="center", color="#222")
        if pts:
            max_d = max(pts, key=lambda x: abs(x[1]))[1]
            ax.text(DISPLAY_HI + 14, 0, f"Δ {max_d:+.2f}",
                    fontsize=14, ha="left", va="center",
                    color="#d04646" if max_d > 0 else "#2966a3", fontweight="bold")

        ax.plot([0, DISPLAY_HI], [0, 0], color="#888", lw=0.5, zorder=1)
        ax.set_yticks([]); ax.set_xticks([])
        for s in ax.spines.values(): s.set_visible(False)
        ax.axvline(VARIANT_DISP, color="#a02c2c", lw=0.7, ls="--", alpha=0.45, zorder=10)

    # ── Gene model ──────────────────────────────────────────────────────────
    ax_mid = fig.add_subplot(gs[N_TRACKS])
    ax_mid.set_xlim(LABEL_X - 5, DISPLAY_HI + 55)
    ax_mid.set_ylim(-1.1, 1.3)
    EXON_Y = 0; EXON_H = 0.55

    for name, rlo, rhi, dlo, dhi in SEGMENTS:
        if "intron" in name:
            ax_mid.plot([dlo, dhi], [0, 0], color="#aaa", lw=1.2)
            for x in np.linspace(dlo, dhi, 6)[1:-1]:
                ax_mid.text(x, 0, "◀", fontsize=9, ha="center", va="center",
                            color="#bbb", zorder=4)

    for name, rlo, rhi, dlo, dhi in SEGMENTS:
        if "exon" in name:
            ax_mid.add_patch(Rectangle((dlo, EXON_Y - EXON_H / 2), dhi - dlo, EXON_H,
                                       facecolor="#3a7ca5", edgecolor="black", lw=0.7, zorder=3))
            ax_mid.text((dlo + dhi) / 2, EXON_Y + EXON_H / 2 + 0.16,
                        name.replace("exon", "Exon "), fontsize=14, ha="center", va="bottom",
                        color="#1b4d72", fontweight="bold")

    trunc_lo = to_disp(NOTCH_LO); trunc_hi = to_disp(NOTCH_HI)
    ax_mid.add_patch(Rectangle((trunc_lo, EXON_Y - EXON_H / 2), trunc_hi - trunc_lo, EXON_H,
                                facecolor="none", edgecolor="#d04646", hatch="///", lw=1.0, zorder=4))
    ax_mid.axvline(VARIANT_DISP, color="#a02c2c", lw=0.7, ls="--", alpha=0.55, zorder=10)
    ax_mid.annotate("rs80358099",
                    xy=(VARIANT_DISP, EXON_Y - EXON_H / 2),
                    xytext=(VARIANT_DISP, EXON_Y - 0.82),
                    fontsize=13, ha="center", va="top", color="#a02c2c", fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="#a02c2c", lw=0.6, alpha=0.7))
    ax_mid.annotate("8 bp truncation",
                    xy=((trunc_lo + trunc_hi) / 2, EXON_Y + EXON_H / 2),
                    xytext=((trunc_lo + trunc_hi) / 2 + 38, EXON_Y + 0.95),
                    fontsize=13, ha="center", va="center", color="#d04646", fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="#d04646", lw=0.6, alpha=0.8,
                                   connectionstyle="arc3,rad=0.15"))
    ax_mid.text(DISPLAY_HI + 15, 0, "5′", fontsize=13, ha="left", va="center",
                color="#555", fontweight="bold")
    ax_mid.text(LABEL_X, 0, "3′", fontsize=13, ha="right", va="center",
                color="#555", fontweight="bold")
    ax_mid.set_yticks([]); ax_mid.set_xticks([])
    for s in ax_mid.spines.values(): s.set_visible(False)

    # ── RNA-seq lanes ───────────────────────────────────────────────────────
    MAX_READS = max(v for s in SAMPLES for v in s["junctions"].values())

    for si, sample in enumerate(SAMPLES):
        ax_rna = fig.add_subplot(gs[N_TRACKS + 2 + si])
        ax_rna.set_xlim(LABEL_X - 5, DISPLAY_HI + 55)
        ax_rna.set_ylim(0, 2.2)
        ax_rna.set_yticks([]); ax_rna.set_xticks([])
        for s in ax_rna.spines.values(): s.set_visible(False)
        ax_rna.axvline(VARIANT_DISP, color="#a02c2c", lw=0.7, ls="--", alpha=0.35, zorder=10)

        ax_rna.text(LABEL_X, 1.14, sample["label"], fontsize=17, ha="right", va="center",
                    color=sample["color"], fontweight="bold")
        ax_rna.text(LABEL_X, 0.70, sample["sub"], fontsize=13, ha="right", va="center",
                    color=sample["color"], style="italic")

        COV_BASE = 0.05; COV_MAX_Y = 0.95
        max_cov = sample["_max_cov"]
        for exon_name, arr in sample["_cov"].items():
            if arr.max() == 0: continue
            rlo, rhi = EXONS[exon_name]
            xs_disp = np.array([to_disp(x) for x in range(rlo, rhi + 1)])
            ys = COV_BASE + (arr / max_cov) * (COV_MAX_Y - COV_BASE)
            ax_rna.fill_between(xs_disp, COV_BASE, ys, color=sample["color"],
                                alpha=0.60, lw=0, zorder=3)
            ax_rna.plot(xs_disp, ys, color=sample["color"], lw=0.7, alpha=0.9, zorder=4)

        ax_rna.plot([LABEL_X + 5, DISPLAY_HI + 15], [COV_BASE, COV_BASE],
                    color="#ccc", lw=0.4, zorder=1)
        ax_rna.text(DISPLAY_HI + 5, COV_MAX_Y, f"max {int(max_cov)}",
                    fontsize=13, ha="left", va="center", color=sample["color"], fontweight="bold")

        SASHI_BASE = 1.05; SASHI_TOP = 2.05; SASHI_RANGE = SASHI_TOP - SASHI_BASE
        for junc_key, reads in sample["junctions"].items():
            x0, x1 = JUNC_ENDS[junc_key]
            color = JUNC_COLORS[junc_key]
            apex = SASHI_BASE + JUNC_APEX[junc_key] * SASHI_RANGE + 0.5 * SASHI_RANGE
            if reads == 0:
                lw = 1.2; alpha = 0.6; ls = (0, (1, 1.8))
            else:
                lw = 0.8 + 3.5 * (np.log1p(reads) / np.log1p(MAX_READS))
                alpha = 0.85; ls = "-"
            mid = (x0 + x1) / 2; span = abs(x1 - x0) / 2
            if span < 1: continue
            xs = np.linspace(min(x0, x1), max(x0, x1), 200)
            ys = SASHI_BASE + (apex - SASHI_BASE) * (1 - ((xs - mid) / span) ** 2)
            ax_rna.plot(xs, ys, color=color, lw=lw, alpha=alpha, ls=ls, zorder=3)
            if reads > 0:
                ax_rna.text(mid, apex + 0.03, f"{reads}", fontsize=13,
                            ha="center", va="bottom", color=color,
                            fontweight="bold")


    bot_legend = [
        plt.Line2D([0], [0], color="#d04646", lw=3.5, label="Cryptic +8 acceptor"),
        plt.Line2D([0], [0], color="#e89a2c", lw=3.5, label="Full exon-22 skip"),
        plt.Line2D([0], [0], color="#666",    lw=2.5, label="Canonical junction"),
        plt.Line2D([0], [0], color="#888",    lw=1.2, ls=(0, (1, 1.8)), label="0 reads (suppressed)"),
    ]
    fig.legend(handles=bot_legend, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), fontsize=13, framealpha=0.95,
               title="RNA-seq junctions", title_fontsize=13.5, ncol=4)

    plt.subplots_adjust(left=0.11, right=0.92, top=0.93, bottom=0.10)
    save_figure(fig, OUT_STEM)
    print(f"[wrote] {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
