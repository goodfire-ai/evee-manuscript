#!/usr/bin/env python3
"""
Prepare context ablation evaluation data for manuscript figures.

Reads mechanism judge and bio judge CSVs from the variant-viewer eval pipeline,
merges them, filters to the 5 manuscript configs, applies memorization filter,
and writes a tidy feather artifact with per-(variant, config, model) scores.

Input:  variant-viewer/eval/results/{mechanism,bio}_judge_expert_final.csv
Output: artifacts/context_ablation_eval.feather
"""
from pathlib import Path

import pandas as pd
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "context_ablation_eval.feather"

EVAL_RESULTS = Path(
    "/mnt/polished-lake/home/michaelpearce/projects/mayo/variant-viewer/eval/results"
)

CONFIGS = [
    "coord_only",
    "coord_gene",
    "coord_gene_context",
    "coord_gene_context_hgvsp",
    "coord_gene_context_hgvsp_evo2",
]

MERGE_KEYS = ["variant_id", "config", "model", "label"]


def main():
    # Load judge results
    mech = pd.read_csv(EVAL_RESULTS / "mechanism_judge_expert_final.csv")
    mech["mechanism_score"] = pd.to_numeric(mech["mechanism_score"], errors="coerce")

    bio = pd.read_csv(EVAL_RESULTS / "bio_judge_expert_final.csv")
    bio["biological_accuracy"] = pd.to_numeric(bio["biological_accuracy"], errors="coerce")
    bio["specificity"] = pd.to_numeric(bio["specificity"], errors="coerce")

    # Merge and compute composite
    df = mech[MERGE_KEYS + ["mechanism_score"]].merge(
        bio[MERGE_KEYS + ["biological_accuracy", "specificity"]],
        on=MERGE_KEYS, how="outer",
    )
    df["composite"] = df[["mechanism_score", "biological_accuracy", "specificity"]].mean(axis=1)

    # Filter to 5 configs
    df = df[df["config"].isin(CONFIGS)].copy()

    # Memorization filter: score 5 on coord_gene_context_hgvsp
    memorized = df[
        (df["config"] == "coord_gene_context_hgvsp") & (df["mechanism_score"] == 5)
    ]["variant_id"].unique()
    df = df[~df["variant_id"].isin(memorized)].copy()

    print(f"Variants: {df['variant_id'].nunique()} ({len(memorized)} memorized removed)")
    print(f"Rows: {len(df)}")

    # Write as polars feather
    out = pl.from_pandas(df)
    out.write_ipc(OUT)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
