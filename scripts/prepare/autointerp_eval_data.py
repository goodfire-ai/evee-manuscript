#!/usr/bin/env python3
"""
Prepare Figure 2E panel data — autointerp ablation composite scores.

Merges the original ablation study with corrected Sonnet 4.6 results,
then computes per-(model, config) composite SCV judge scores.

Input:  eval_probe_autointerp/results/ablation_study.csv
        eval_probe_autointerp/results/ablation_sonnet46.csv
Output: artifacts/autointerp_eval.feather
"""
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "autointerp_eval.feather"

DATA = ROOT / "data"
RESULTS_DIR = DATA / "autointerp"            # symlink to autointerp results
ABLATION_CSV = RESULTS_DIR / "ablation_study.feather"
SONNET46_CSV = RESULTS_DIR / "ablation_sonnet46.feather"

JUDGE_COLS = ["mechanism_coverage", "biological_accuracy", "specificity", "novel_insights"]


def main():
    orig = pd.read_csv(ABLATION_CSV)
    sonnet46 = pd.read_csv(SONNET46_CSV)
    df = pd.concat([orig[orig["model"] != "sonnet"], sonnet46], ignore_index=True)

    for c in JUDGE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    rows = []
    for (model, config), sub in df.groupby(["model", "config"]):
        composite = sub[JUDGE_COLS].mean(axis=1)
        rows.append({
            "model": model,
            "config": config,
            "composite_mean": round(composite.mean(), 4),
            "composite_sem": round(composite.sem(), 4),
            "n": len(composite),
        })

    out = pl.DataFrame(rows).sort("model", "config")
    out.write_ipc(OUT)
    print(f"Saved {len(out)} rows to {OUT}")


if __name__ == "__main__":
    main()
