#!/usr/bin/env python3
"""
Prepare disruption AUROC data — per-annotation pathogenicity prediction.

For each annotation head, computes AUROC of the disruption delta (var - ref)
for predicting ClinVar pathogenicity, filtering to variants where the
reference annotation is present (ref > 0.5).

Input:  website_probes/v4/token/scores.feather
        clinvar/evo2-7b/labeled/split.feather (pathogenicity labels)
        variant-viewer/heads.json (browser groups)
Output: data/panels/fig2_disrupt_auroc.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as pf
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "fig2_disrupt_auroc.feather"

DATA = ROOT / "data"
PROBE_DIR = DATA / "probes" / "token"        # symlink to probe checkpoint dir
SCORES_FEATHER = PROBE_DIR / "scores.feather"
LABELS_FEATHER = DATA / "clinvar" / "split.feather"  # symlink to labeled split
HEADS_JSON = DATA / "viewer" / "heads.json"  # symlink to variant-viewer heads

MIN_VARIANTS = 50  # minimum variants per head after filtering


def _browser_group(head: str, viewer_map: dict) -> str:
    """Assign browser group to a head."""
    if head in viewer_map:
        return viewer_map[head]
    if head.startswith("chipseq_"):
        return "ChIP-seq"
    if head.startswith("chromhmm_") or head.startswith("fstack_") or head.startswith("ccre_"):
        return "Chromatin"
    if head.startswith("atacseq_"):
        return "ATAC-seq"
    if head.startswith("interpro_"):
        return "InterPro"
    if head.startswith("in_"):
        return "Structure"
    if head.startswith("is_"):
        return "Structure"
    if head.startswith("elm_"):
        return "ELM Motif"
    if head.startswith("ptm_"):
        return "PTM"
    if head.startswith("amino_acid_"):
        return "Substitution"
    if head.startswith("secondary_structure_"):
        return "Protein"
    if head.startswith("region_"):
        return "Region"
    if head.startswith("spliceai_"):
        return "Splice"
    return "Other"


def main():
    # 1. Load browser groups
    with open(HEADS_JSON) as f:
        viewer_map = {h: cfg["group"] for h, cfg in json.load(f)["heads"].items()}

    # 2. Load pathogenicity labels
    print("Loading labels...")
    labels = pf.read_table(LABELS_FEATHER, columns=["variant_id", "pathogenic"]).to_pandas()
    label_map = dict(zip(labels["variant_id"], labels["pathogenic"]))
    print(f"  {len(label_map)} labeled variants")

    # 3. Load v4 scores
    print("Loading scores...")
    schema = pf.read_table(SCORES_FEATHER, columns=[]).schema
    all_cols = [f.name for f in schema]
    var_cols = sorted([c for c in all_cols if c.startswith("var_")])
    ref_cols = sorted([c for c in all_cols if c.startswith("ref_")])
    heads = [c[4:] for c in var_cols]

    table = pf.read_table(SCORES_FEATHER, columns=["variant_id"] + var_cols + ref_cols)
    scores = table.to_pandas()
    del table
    print(f"  {len(scores)} variants, {len(heads)} heads")

    # 4. Join with labels
    scores["pathogenic"] = scores["variant_id"].map(label_map)
    scores = scores.dropna(subset=["pathogenic"])
    scores["pathogenic"] = scores["pathogenic"].astype(int)
    print(f"  {len(scores)} variants with labels (pathogenic rate: {scores['pathogenic'].mean():.3f})")

    # 5. Compute AUROC per head
    print("Computing per-head AUROC...")
    rows = []
    for head in heads:
        ref_col = f"ref_{head}"
        var_col = f"var_{head}"

        # Filter to variants where reference annotation is present
        mask = scores[ref_col] > 0.5
        sub = scores.loc[mask, ["pathogenic", ref_col, var_col]]
        n = len(sub)
        if n < MIN_VARIANTS:
            continue

        n_path = sub["pathogenic"].sum()
        n_benign = n - n_path
        if n_path < 5 or n_benign < 5:
            continue

        y = sub["pathogenic"].values
        disruption = sub[var_col].values - sub[ref_col].values

        # AUROC: negative disruption predicts pathogenic
        try:
            auroc_neg = roc_auc_score(y, -disruption)
        except ValueError:
            auroc_neg = np.nan

        # AUROC: absolute disruption predicts pathogenic
        try:
            auroc_abs = roc_auc_score(y, np.abs(disruption))
        except ValueError:
            auroc_abs = np.nan

        auroc_best = max(
            auroc_neg if not np.isnan(auroc_neg) else 0,
            auroc_abs if not np.isnan(auroc_abs) else 0,
        )
        best_dir = "neg" if (auroc_neg >= auroc_abs or np.isnan(auroc_abs)) else "abs"

        rows.append({
            "head": head,
            "browser_group": _browser_group(head, viewer_map),
            "n_variants": n,
            "n_pathogenic": n_path,
            "auroc_neg": round(auroc_neg, 4) if not np.isnan(auroc_neg) else None,
            "auroc_abs": round(auroc_abs, 4) if not np.isnan(auroc_abs) else None,
            "auroc_best": round(auroc_best, 4),
            "best_direction": best_dir,
        })

    # 6. Write CSV
    import polars as pl
    df = pl.DataFrame(rows).sort("browser_group", "head")
    df.write_ipc(OUT)
    print(f"\nSaved {len(df)} rows to {OUT}")
    print(df.group_by("browser_group").agg(
        pl.len().alias("n_heads"),
        pl.col("auroc_best").median().alias("median_auroc"),
    ).sort("median_auroc", descending=True))


if __name__ == "__main__":
    main()
