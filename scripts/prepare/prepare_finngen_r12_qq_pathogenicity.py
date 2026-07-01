#!/usr/bin/env python3
"""Prepare artifact for the FinnGen R12 dose-response QQ plot.

Grouping is defined by the predicted (EVEE) pathogenicity score, providing an
independent validation against FinnGen R12 (a dataset EVEE never trained on):

  hi95    : EVEE score >= 0.95            (high-confidence predictions)
  hi80    : EVEE score >= 0.80            (inclusive predicted-pathogenic set;
                                           hi95 is a nested subset)
  control : EVEE score <  0.80, allele-frequency-matched 5:1 to hi80
            (null reference; the <0.80 complement of the 0.80 threshold)

The 0.80 threshold partitions the overlap into hi80 (positive) and <0.80
(control), so the control is matched to hi80. 5:1 matching (5 controls per
hi80 variant, pooled) stabilizes the null lambda against sampling noise.

A de-circularized reference -- EVEE >= 0.95 restricted to variants with no
established ClinVar classification (not P/LP/B/LB) -- is computed and printed
for the caption; it is not plotted.

Input:  /mnt/data/shared/life-sciences/EVEE/finngen-validation/
            eevee_regenie_overlap/all_endpoints.parquet
Output: artifacts/finngen_r12_qq_pathogenicity.feather

Run:
    uv run python scripts/prepare/prepare_finngen_r12_qq_pathogenicity.py
"""
from __future__ import annotations
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

PARQUET = Path("/mnt/data/shared/life-sciences/EVEE/finngen-validation/"
               "eevee_regenie_overlap/all_endpoints.parquet")
ROOT    = Path(__file__).resolve().parents[2]
OUT     = ROOT / "artifacts" / "finngen_r12_qq_pathogenicity.feather"

SEED        = 42
N_BINS      = 30
MAF_MIN     = 1e-5
MAF_MAX     = 0.5
THRESH_HI   = 0.95
THRESH_LO   = 0.80
MATCH_RATIO = 5            # controls per hi80 variant (5:1)

WHERE_HI95 = f"eevee_pathogenicity >= {THRESH_HI}"
WHERE_HI80 = f"eevee_pathogenicity >= {THRESH_LO}"
WHERE_LOW  = f"eevee_pathogenicity <  {THRESH_LO}"
# de-circularized reference (printed for caption, not plotted)
WHERE_UNL  = (f"eevee_label NOT IN ('pathogenic','likely_pathogenic',"
              f"'benign','likely_benign') AND eevee_pathogenicity >= {THRESH_HI}")


def af_matched_indices(maf_target, maf_pool, rng, ratio=1):
    edges = np.logspace(np.log10(MAF_MIN), np.log10(MAF_MAX), N_BINS + 1)
    edges[0] = 0.0; edges[-1] = 1.0
    target_hist, _ = np.histogram(maf_target, bins=edges)
    pool_bin = np.clip(np.digitize(maf_pool, edges) - 1, 0, N_BINS - 1)
    out = []
    for b in range(N_BINS):
        need = int(target_hist[b]) * ratio
        if need == 0:
            continue
        avail = np.where(pool_bin == b)[0]
        if avail.size == 0:
            continue
        out.append(rng.choice(avail, size=need, replace=avail.size < need))
    return np.concatenate(out) if out else np.array([], dtype=int)


def lambda_gc(pvals):
    from scipy.special import ndtri
    p = pvals[(pvals > 0) & (pvals < 1)]
    if p.size == 0:
        return float("nan")
    chi2_obs = ndtri(np.clip(1.0 - p / 2.0, 1e-300, 1.0)) ** 2
    return float(np.median(chi2_obs) / 0.4549364231195724)


def qq_xy(pvals, n_keep=5000):
    pvals = np.sort(pvals)
    n = pvals.size
    expected = -np.log10((np.arange(1, n + 1) - 0.5) / n)
    observed = -np.log10(pvals)
    if n > n_keep:
        keep = np.unique(np.round(np.logspace(0, np.log10(n), n_keep)).astype(int) - 1)
        keep = np.clip(keep, 0, n - 1)
        return expected[keep], observed[keep]
    return expected, observed


def bonferroni_correct_minp(pvals, k):
    return -np.expm1(k * np.log1p(-np.asarray(pvals, dtype=np.float64)))


def main():
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='16GB'")

    K = con.execute(
        f"SELECT COUNT(DISTINCT phenocode) FROM read_parquet('{PARQUET}')"
    ).fetchone()[0]
    print(f"K = {K:,} phenotypes  (hi={THRESH_HI}, lo={THRESH_LO}, match={MATCH_RATIO}:1)",
          flush=True)

    BASE = (f"FROM read_parquet('{PARQUET}') "
            f"WHERE pval > 0 AND pval IS NOT NULL "
            f"AND af_alt > 0 AND af_alt IS NOT NULL")

    def fetch_per_variant(where):
        sql = f"""
            SELECT MIN(pval) AS min_p,
                   LEAST(AVG(af_alt), 1.0 - AVG(af_alt)) AS maf
            {BASE} AND ({where})
            GROUP BY chrom, pos, ref, alt
            ORDER BY chrom, pos, ref, alt
        """
        arr = con.execute(sql).fetchnumpy()
        return (np.asarray(arr["min_p"], dtype=np.float64),
                np.asarray(arr["maf"],   dtype=np.float64))

    vp_95,  vm_95  = fetch_per_variant(WHERE_HI95)
    vp_80,  vm_80  = fetch_per_variant(WHERE_HI80)
    vp_low, vm_low = fetch_per_variant(WHERE_LOW)
    vp_unl, _      = fetch_per_variant(WHERE_UNL)

    rng = np.random.default_rng(SEED)
    vidx_ctrl = af_matched_indices(vm_80, vm_low, rng, ratio=MATCH_RATIO)  # matched to hi80
    vp_ctrl   = vp_low[vidx_ctrl]

    vp_95_c   = bonferroni_correct_minp(vp_95,  K)
    vp_80_c   = bonferroni_correct_minp(vp_80,  K)
    vp_ctrl_c = bonferroni_correct_minp(vp_ctrl, K)
    vp_unl_c  = bonferroni_correct_minp(vp_unl, K)

    lam = {
        "hi95":    (lambda_gc(vp_95_c),   vp_95.size),
        "hi80":    (lambda_gc(vp_80_c),   vp_80.size),
        "control": (lambda_gc(vp_ctrl_c), vp_ctrl.size),
    }
    for g, (lm, n) in lam.items():
        print(f"  {g}: lambda={lm:.3f}  n={n:,}", flush=True)
    print(f"  [caption ref] EVEE>=0.95 unclassified (not P/LP/B/LB): "
          f"lambda={lambda_gc(vp_unl_c):.3f}  n={vp_unl.size:,}", flush=True)

    # Control lambda is computed on the full 5:1 matched set (stable null), but
    # the plotted control curve is drawn from a subset sized to the >=0.80 group
    # so all three curves span a comparable expected-quantile range (avoids the
    # 5x-larger control extending further right and looking like the top hits).
    ctrl_plot = rng.choice(vp_ctrl_c, size=min(vp_80_c.size, vp_ctrl_c.size),
                           replace=False)
    qq_pvals = {"hi95": vp_95_c, "hi80": vp_80_c, "control": ctrl_plot}

    rows = []
    for group in ["hi95", "hi80", "control"]:
        exp, obs = qq_xy(qq_pvals[group])
        lm, n = lam[group]          # lambda and n_variants from the full sets
        for e, o in zip(exp.tolist(), obs.tolist()):
            rows.append({"group": group, "expected": e, "observed": o,
                         "lambda_gc": lm, "n_variants": n, "K": K})

    df = pl.DataFrame(rows, schema={
        "group":      pl.Utf8,
        "expected":   pl.Float64,
        "observed":   pl.Float64,
        "lambda_gc":  pl.Float64,
        "n_variants": pl.Int64,
        "K":          pl.Int64,
    })
    df.write_ipc(OUT)
    print(f"\nWritten -> {OUT}  ({len(df):,} rows)", flush=True)


if __name__ == "__main__":
    main()
