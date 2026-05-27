#!/usr/bin/env python3
"""
Fig 1I — FinnGen R12 QQ plot: ClinVar-labeled variants, AF-matched.

Per-variant min(p) QQ plot (Bonferroni-corrected across FinnGen R12 phenotypes)
comparing ClinVar pathogenic/LP variants vs AF-matched benign/LB comparators.

Adapted from evee_resubmission_natgen/fig_finngen_r12_qq_clinvar_af_matched.py
with manuscript theme applied and output routed to figures/figure1/.

Input:  artifacts/finngen_r12_qq_clinvar.feather
Output: figures/figure1/fig1i_finngen_qq.{png,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
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

ARTIFACT = ROOT / "artifacts" / "finngen_r12_qq_clinvar.feather"
OUT_STEM = ROOT / "figures" / "figure1" / "fig1i_finngen_qq"

apply_theme()

SYMLOG_THRESH = 30.0
LINEAR_TICKS  = [0, 5, 10, 20, 30]
LOG_TICKS     = [50, 100, 200, 500, 1000, 2000, 5000]

COLOR_BENIGN_STRICT = cm.viridis(0.05)
COLOR_COMPLEMENT    = cm.viridis(0.55)
COLOR_PATHOGENIC    = cm.viridis(0.95)


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
    xmax = float(np.nanmax(all_expected)) * 1.15   # expanded x margin
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
            float(sub["lambda_gc_full"][0]),
            int(sub["n_variants"][0]))


def main():
    if not ARTIFACT.exists():
        raise FileNotFoundError(
            f"Artifact not found: {ARTIFACT}\n"
            "Generate with: scripts/prepare/prepare_finngen_r12_qq_clinvar.py"
        )

    df = pl.read_ipc(ARTIFACT)
    K  = int(df["K"][0])

    xa, ya, lp, _,   np_ = _get_group(df, "pathogenic")
    xc, yc, lc, lcf, nc  = _get_group(df, "benign_comp")
    xs, ys, ls, lsf, ns  = _get_group(df, "benign_strict")

    fig, ax = plt.subplots(figsize=(7.2, 5.4))   # 4.5:6 height:width

    DOT_SIZE = 12
    ax.scatter(xs, ys, s=DOT_SIZE, alpha=0.6, color=COLOR_BENIGN_STRICT,
               label=f"Control benign (strict)  λ={ls:.3f}",
               rasterized=True, zorder=1)
    ax.scatter(xc, yc, s=DOT_SIZE, alpha=0.6, color=COLOR_COMPLEMENT,
               label=f"Control benign  λ={lc:.3f}",
               rasterized=True, zorder=2)
    ax.scatter(xa, ya, s=DOT_SIZE, alpha=0.6, color=COLOR_PATHOGENIC,
               label=f"EVEE predicted pathogenic  λ={lp:.3f}",
               rasterized=True, zorder=3)

    _apply_qq_axes(ax,
                   np.concatenate([xa, xc, xs]),
                   np.concatenate([ya, yc, ys]))

    FS_LABEL  = FONT_SIZE_LABEL  + 2
    FS_TICK   = FONT_SIZE_TICK   + 2
    FS_LEGEND = FONT_SIZE_LEGEND + 2

    ax.set_xlabel("Expected −log₁₀(p)", fontsize=FS_LABEL, fontweight="semibold")
    ax.set_ylabel("Observed −log₁₀ P FinnGen", fontsize=FS_LABEL, fontweight="semibold")
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(loc="upper left", fontsize=FS_LEGEND, frameon=False)
    ax.grid(alpha=0.12)

    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")
    print(f"  N_path={np_:,}  N_benign_comp={nc:,}  N_benign_strict={ns:,}")


if __name__ == "__main__":
    main()
