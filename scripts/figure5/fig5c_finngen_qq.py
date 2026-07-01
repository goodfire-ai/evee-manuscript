#!/usr/bin/env python3
"""
Fig 5c -- FinnGen R12 QQ plot: EVEE-score dose-response, AF-matched.

Per-variant min(p) QQ plot (Bonferroni-corrected across FinnGen R12 phenotypes)
showing that the predicted (EVEE) pathogenicity score tracks disease-association
enrichment across its range, validated against a dataset EVEE never trained on:

  EVEE >= 0.95               -- high-confidence predictions
  EVEE >= 0.80               -- inclusive predicted-pathogenic set (hi95 nested)
  EVEE <  0.80 (AF-matched)  -- null reference, 5:1 AF-matched to the >=0.80 set

Input:  artifacts/finngen_r12_qq_pathogenicity.feather
Output: figures/figure5/fig5c_finngen_qq.{png,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from theme.theme import (
    apply_theme, save_figure,
    FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_LEGEND,
)

ARTIFACT = ROOT / "artifacts" / "finngen_r12_qq_pathogenicity.feather"
OUT_STEM = ROOT / "figures" / "figure5" / "fig5c_finngen_qq"

apply_theme()

SYMLOG_THRESH = 30.0
LINEAR_TICKS  = [0, 5, 10, 20, 30]
LOG_TICKS     = [50, 100, 200, 500, 1000, 2000, 5000]

COLOR_HI95    = "#C0392B"   # crimson  -- EVEE >= 0.95
COLOR_HI80    = "#E67E22"   # amber    -- EVEE >= 0.80
COLOR_CONTROL = "#8A8A8A"   # grey     -- EVEE < 0.80 (AF-matched null)


def _apply_log_above(ax, ymax):
    if ymax > SYMLOG_THRESH:
        ax.set_yscale("symlog", linthresh=SYMLOG_THRESH, linscale=1.0)
        ticks = [t for t in LINEAR_TICKS + LOG_TICKS if t <= ymax * 1.05]
    else:
        ticks = [t for t in LINEAR_TICKS if t <= ymax * 1.05]
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())


def _apply_qq_axes(ax, all_expected, all_observed):
    xmax = float(np.nanmax(all_expected)) * 1.15
    ymax = float(np.nanmax(all_observed))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax * 1.02)
    ax.plot([0, max(xmax, ymax)], [0, max(xmax, ymax)], "--", color="#aaaaaa", lw=0.8)
    _apply_log_above(ax, ymax)


def _get_group(df: pl.DataFrame, group: str):
    sub = df.filter(pl.col("group") == group)
    return (sub["expected"].to_numpy(),
            sub["observed"].to_numpy(),
            float(sub["lambda_gc"][0]),
            int(sub["n_variants"][0]))


def main():
    if not ARTIFACT.exists():
        raise FileNotFoundError(
            f"Artifact not found: {ARTIFACT}\n"
            "Generate with: scripts/prepare/prepare_finngen_r12_qq_pathogenicity.py"
        )

    df = pl.read_ipc(ARTIFACT)

    x95, y95, l95, n95 = _get_group(df, "hi95")
    x80, y80, l80, n80 = _get_group(df, "hi80")
    xc,  yc,  lc,  nc  = _get_group(df, "control")

    fig, ax = plt.subplots(figsize=(5.4, 4.4))

    DOT_SIZE = 20
    DOT_EDGE = dict(edgecolor="white", linewidth=0.2)
    # control first (background), then nested positive curves on top
    h_control = ax.scatter(xc, yc, s=DOT_SIZE, alpha=0.7, color=COLOR_CONTROL,
               label=f"EVEE < 0.80, AF-matched   $\\lambda$={lc:.2f}",
               rasterized=True, zorder=1, **DOT_EDGE)
    h_80 = ax.scatter(x80, y80, s=DOT_SIZE, alpha=0.85, color=COLOR_HI80,
               label=f"EVEE $\\geq$ 0.80   $\\lambda$={l80:.2f}",
               rasterized=True, zorder=2, **DOT_EDGE)
    h_95 = ax.scatter(x95, y95, s=DOT_SIZE, alpha=0.8, color=COLOR_HI95,
               label=f"EVEE $\\geq$ 0.95   $\\lambda$={l95:.2f}",
               rasterized=True, zorder=3, **DOT_EDGE)

    _apply_qq_axes(ax,
                   np.concatenate([x95, x80, xc]),
                   np.concatenate([y95, y80, yc]))

    FS_LABEL  = FONT_SIZE_LABEL  + 2
    FS_TICK   = FONT_SIZE_TICK   + 2
    FS_LEGEND = FONT_SIZE_LEGEND + 2

    # NB: use literal "log" inside \mathbf -- mathtext's \log macro does not bold
    ax.set_xlabel(r"Expected $\mathbf{-log_{10}(p)}$", fontsize=FS_LABEL, fontweight="semibold")
    ax.set_ylabel(r"Observed $\mathbf{-log_{10}P}$ FinnGen", fontsize=FS_LABEL, fontweight="semibold")
    ax.tick_params(labelsize=FS_TICK)
    legend = ax.legend(handles=[h_95, h_80, h_control],
                       loc="upper left", fontsize=FS_LEGEND, frameon=False)
    for handle in legend.legend_handles:
        handle.set_sizes([42])
        handle.set_alpha(1.0)
    ax.grid(alpha=0.12)

    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")
    print(f"  n: >=0.95={n95:,}  >=0.80={n80:,}  control={nc:,}")
    print(f"  lambda: >=0.95={l95:.3f}  >=0.80={l80:.3f}  control={lc:.3f}")


if __name__ == "__main__":
    main()
