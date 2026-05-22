#!/usr/bin/env python3
"""QQ plot: ClinVar-labeled FinnGen R12 variants, AF-matched.

Reads pre-computed QQ curve data from artifacts/finngen_r12_qq_clinvar.feather.
Generate the artifact first:
    /mnt/home/ryo/variant-viewer/.venv/bin/python \\
        scripts/prepare/prepare_finngen_r12_qq_clinvar.py

Run:
    uv run python evee_resubmission_natgen/fig_finngen_r12_qq_clinvar_af_matched.py
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl

ROOT     = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "finngen_r12_qq_clinvar.feather"
OUT      = Path(__file__).with_name("fig_finngen_r12_qq_clinvar_af_matched.png")
OUT_SVG  = Path(__file__).with_name("fig_finngen_r12_qq_clinvar_af_matched.svg")

SYMLOG_THRESH = 30.0
LINEAR_TICKS  = [0, 5, 10, 20, 30]
LOG_TICKS     = [50, 100, 200, 500, 1000, 2000, 5000]

COLOR_BENIGN_STRICT = cm.viridis(0.05)
COLOR_COMPLEMENT    = cm.viridis(0.55)
COLOR_PATHOGENIC    = cm.viridis(0.95)


def apply_log_above(ax, ymax):
    if ymax > SYMLOG_THRESH:
        ax.set_yscale("symlog", linthresh=SYMLOG_THRESH, linscale=1.0)
        ticks = [t for t in LINEAR_TICKS + LOG_TICKS if t <= ymax * 1.05]
    else:
        ticks = [t for t in LINEAR_TICKS if t <= ymax * 1.05]
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())


def apply_qq_axes(ax, all_expected, all_observed):
    xmax = float(np.nanmax(all_expected)) * 1.02
    ymax = float(np.nanmax(all_observed))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax * 1.02)
    ax.plot([0, max(xmax, ymax)], [0, max(xmax, ymax)], "--", color="gray", lw=1)
    apply_log_above(ax, ymax)


def get_group(df: pl.DataFrame, group: str):
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
            "Run: scripts/prepare/prepare_finngen_r12_qq_clinvar.py"
        )

    df = pl.read_ipc(ARTIFACT)
    K  = int(df["K"][0])

    xa, ya, lp, _,   np_ = get_group(df, "pathogenic")
    xc, yc, lc, lcf, nc  = get_group(df, "benign_comp")
    xs, ys, ls, lsf, ns  = get_group(df, "benign_strict")

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

    ax.scatter(xs, ys, s=4, alpha=0.6, color=COLOR_BENIGN_STRICT,
               label=f"Benign strict AF-matched  λ={ls:.3f}  (full λ={lsf:.3f})",
               rasterized=True, zorder=1)
    ax.scatter(xc, yc, s=4, alpha=0.6, color=COLOR_COMPLEMENT,
               label=f"Benign/LB AF-matched  λ={lc:.3f}  (full λ={lcf:.3f})",
               rasterized=True, zorder=2)
    ax.scatter(xa, ya, s=4, alpha=0.6, color=COLOR_PATHOGENIC,
               label=f"Pathogenic/LP  λ={lp:.3f}",
               rasterized=True, zorder=3)

    apply_qq_axes(ax,
                  np.concatenate([xa, xc, xs]),
                  np.concatenate([ya, yc, ys]))
    ax.set_xlabel("Expected −log₁₀(p) under uniform null")
    ax.set_ylabel(f"Observed −log₁₀(1 − (1 − min p)^K),  K={K:,}")
    ax.set_title(
        f"Per-variant min(p) QQ, Bonferroni-corrected\n"
        f"N_path={np_:,}  N_comp={nc:,}  N_strict={ns:,}"
    )
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "EVEE pathogenicity vs FinnGen R12:\n"
        "ClinVar-labeled variants, AF-matched benign comparators",
        fontsize=11,
    )
    fig.savefig(OUT,     dpi=150, bbox_inches="tight")
    fig.savefig(OUT_SVG, bbox_inches="tight")
    print(f"Saved {OUT}")
    print(f"Saved {OUT_SVG}")


if __name__ == "__main__":
    main()
