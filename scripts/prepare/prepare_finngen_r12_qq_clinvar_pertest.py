#!/usr/bin/env python3
"""Prepare artifact for per-test (no aggregation) ClinVar FinnGen R12 QQ plot.

Input:  /mnt/data/shared/life-sciences/EVEE/finngen-validation/
            eevee_regenie_overlap/all_endpoints.parquet
Output: artifacts/finngen_r12_qq_clinvar_pertest.feather

Pathogenic/LP group: all ~1.5 M variant-phenotype tests fetched in full.
Benign groups: reservoir-sampled to 5 M rows each (3x pathogenic count),
sufficient for AF-matched QQ and stable lambda_gc.

Run:
    /mnt/home/ryo/variant-viewer/.venv/bin/python \\
        scripts/prepare/prepare_finngen_r12_qq_clinvar_pertest.py
"""
from __future__ import annotations
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

PARQUET = Path("/mnt/data/shared/life-sciences/EVEE/finngen-validation/"
               "eevee_regenie_overlap/all_endpoints.parquet")
ROOT    = Path(__file__).resolve().parents[2]
OUT     = ROOT / "artifacts" / "finngen_r12_qq_clinvar_pertest.feather"

SEED          = 42
N_BINS        = 30
MAF_MIN       = 1e-5
MAF_MAX       = 0.5
BENIGN_SAMPLE = 5_000_000

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

    def fetch(where, sample: int | None = None):
        inner = f"SELECT pval, LEAST(af_alt, 1.0 - af_alt) AS maf {BASE} AND ({where})"
        if sample:
            sql = (f"SELECT pval, maf FROM ({inner}) "
                   f"USING SAMPLE {sample} ROWS (reservoir, {SEED})")
        else:
            sql = inner
        arr = con.execute(sql).fetchnumpy()
        return (np.asarray(arr["pval"], dtype=np.float64),
                np.asarray(arr["maf"],  dtype=np.float64))

    print("Fetching per-test p-values ...", flush=True)
    p_path, maf_path = fetch(WHERE_PATH)
    p_comp, maf_comp = fetch(WHERE_COMP,   sample=BENIGN_SAMPLE)
    p_str,  maf_str  = fetch(WHERE_STRICT, sample=BENIGN_SAMPLE)
    print(f"  path={p_path.size:,}  comp={p_comp.size:,}  strict={p_str.size:,}", flush=True)

    rng = np.random.default_rng(SEED)
    idx_comp = af_matched_indices(maf_path, maf_comp, rng)
    idx_str  = af_matched_indices(maf_path, maf_str,  rng)
    p_comp_m = p_comp[idx_comp]
    p_str_m  = p_str[idx_str]

    lam = {
        "pathogenic":    (lambda_gc(p_path),   float("nan"),       p_path.size),
        "benign_comp":   (lambda_gc(p_comp_m), lambda_gc(p_comp),  p_comp.size),
        "benign_strict": (lambda_gc(p_str_m),  lambda_gc(p_str),   p_str.size),
    }
    for g, (lm, lf, n) in lam.items():
        print(f"  {g}: λ_matched={lm:.3f}  λ_full={lf:.3f}  n={n:,}", flush=True)

    rows = []
    for group, pvals in [
        ("pathogenic",    p_path),
        ("benign_comp",   p_comp_m),
        ("benign_strict", p_str_m),
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
