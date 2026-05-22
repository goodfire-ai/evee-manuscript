#!/usr/bin/env python3
"""Prepare artifact for ClinVar-labeled FinnGen R12 AF-matched QQ plot.

Input:  /mnt/data/shared/life-sciences/EVEE/finngen-validation/
            eevee_regenie_overlap/all_endpoints.parquet  (~178k labeled variants)
Output: artifacts/finngen_r12_qq_clinvar.feather

For each variant, aggregate to min(p) across all FinnGen R12 phenotypes,
apply Bonferroni correction, then AF-match benign to pathogenic distribution.

Run:
    /mnt/home/ryo/variant-viewer/.venv/bin/python \\
        scripts/prepare/prepare_finngen_r12_qq_clinvar.py
"""
from __future__ import annotations
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

PARQUET = Path("/mnt/data/shared/life-sciences/EVEE/finngen-validation/"
               "eevee_regenie_overlap/all_endpoints.parquet")
ROOT    = Path(__file__).resolve().parents[2]
OUT     = ROOT / "artifacts" / "finngen_r12_qq_clinvar.feather"

SEED    = 42
N_BINS  = 30
MAF_MIN = 1e-5
MAF_MAX = 0.5

WHERE_PATH   = "eevee_label IN ('pathogenic','likely_pathogenic')"
WHERE_COMP   = "eevee_label IN ('benign','likely_benign')"
WHERE_STRICT = "eevee_label = 'benign'"


def af_matched_indices(maf_target: np.ndarray, maf_pool: np.ndarray, rng) -> np.ndarray:
    edges = np.logspace(np.log10(MAF_MIN), np.log10(MAF_MAX), N_BINS + 1)
    edges[0] = 0.0; edges[-1] = 1.0
    target_hist, _ = np.histogram(maf_target, bins=edges)
    pool_bin = np.clip(np.digitize(maf_pool, edges) - 1, 0, N_BINS - 1)
    out = []
    for b in range(N_BINS):
        need = int(target_hist[b])
        if need == 0:
            continue
        avail = np.where(pool_bin == b)[0]
        if avail.size == 0:
            continue
        out.append(rng.choice(avail, size=need, replace=avail.size < need))
    return np.concatenate(out) if out else np.array([], dtype=int)


def lambda_gc(pvals: np.ndarray) -> float:
    from scipy.special import ndtri
    p = pvals[(pvals > 0) & (pvals < 1)]
    if p.size == 0:
        return float("nan")
    chi2_obs = ndtri(np.clip(1.0 - p / 2.0, 1e-300, 1.0)) ** 2
    return float(np.median(chi2_obs) / 0.4549364231195724)


def qq_xy(pvals: np.ndarray, n_keep: int = 5000):
    pvals = np.sort(pvals)
    n = pvals.size
    expected = -np.log10((np.arange(1, n + 1) - 0.5) / n)
    observed = -np.log10(pvals)
    if n > n_keep:
        keep = np.unique(
            np.round(np.logspace(0, np.log10(n), n_keep)).astype(int) - 1
        )
        keep = np.clip(keep, 0, n - 1)
        return expected[keep], observed[keep]
    return expected, observed


def bonferroni_correct_minp(pvals: np.ndarray, k: int) -> np.ndarray:
    return -np.expm1(k * np.log1p(-np.asarray(pvals, dtype=np.float64)))


def main():
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='16GB'")

    print("Counting phenotypes ...", flush=True)
    K = con.execute(
        f"SELECT COUNT(DISTINCT phenocode) FROM read_parquet('{PARQUET}')"
    ).fetchone()[0]
    print(f"  K = {K:,} phenotypes", flush=True)

    BASE = (f"FROM read_parquet('{PARQUET}') "
            f"WHERE pval > 0 AND pval IS NOT NULL "
            f"AND af_alt > 0 AND af_alt IS NOT NULL")

    def fetch_per_variant(where):
        sql = f"""
            SELECT MIN(pval) AS min_p,
                   LEAST(AVG(af_alt), 1.0 - AVG(af_alt)) AS maf
            {BASE} AND ({where})
            GROUP BY chrom, pos, ref, alt
        """
        arr = con.execute(sql).fetchnumpy()
        return (np.asarray(arr["min_p"], dtype=np.float64),
                np.asarray(arr["maf"],   dtype=np.float64))

    print("Fetching per-variant min(p) ...", flush=True)
    vp_path, vm_path = fetch_per_variant(WHERE_PATH)
    vp_comp, vm_comp = fetch_per_variant(WHERE_COMP)
    vp_str,  vm_str  = fetch_per_variant(WHERE_STRICT)
    print(f"  path={vp_path.size:,}  comp={vp_comp.size:,}  strict={vp_str.size:,}", flush=True)

    rng = np.random.default_rng(SEED)
    vidx_comp = af_matched_indices(vm_path, vm_comp, rng)
    vidx_str  = af_matched_indices(vm_path, vm_str,  rng)
    vp_comp_m = vp_comp[vidx_comp]
    vp_str_m  = vp_str[vidx_str]

    vp_path_c   = bonferroni_correct_minp(vp_path,   K)
    vp_comp_m_c = bonferroni_correct_minp(vp_comp_m, K)
    vp_str_m_c  = bonferroni_correct_minp(vp_str_m,  K)
    vp_comp_c   = bonferroni_correct_minp(vp_comp,   K)
    vp_str_c    = bonferroni_correct_minp(vp_str,    K)

    lam = {
        "pathogenic":    (lambda_gc(vp_path_c),   float("nan"),        vp_path.size),
        "benign_comp":   (lambda_gc(vp_comp_m_c), lambda_gc(vp_comp_c), vp_comp.size),
        "benign_strict": (lambda_gc(vp_str_m_c),  lambda_gc(vp_str_c),  vp_str.size),
    }
    for g, (lm, lf, n) in lam.items():
        print(f"  {g}: λ_matched={lm:.3f}  λ_full={lf:.3f}  n={n:,}", flush=True)

    rows = []
    for group, pvals in [
        ("pathogenic",    vp_path_c),
        ("benign_comp",   vp_comp_m_c),
        ("benign_strict", vp_str_m_c),
    ]:
        exp, obs = qq_xy(pvals)
        lm, lf, n = lam[group]
        for e, o in zip(exp.tolist(), obs.tolist()):
            rows.append({
                "group":          group,
                "expected":       e,
                "observed":       o,
                "lambda_gc":      lm,
                "lambda_gc_full": lf,
                "n_variants":     n,
                "K":              K,
            })

    df = pl.DataFrame(rows, schema={
        "group":          pl.Utf8,
        "expected":       pl.Float64,
        "observed":       pl.Float64,
        "lambda_gc":      pl.Float64,
        "lambda_gc_full": pl.Float64,
        "n_variants":     pl.Int64,
        "K":              pl.Int64,
    })

    df.write_ipc(OUT)
    print(f"\nWritten → {OUT}  ({len(df):,} rows)", flush=True)


if __name__ == "__main__":
    main()
