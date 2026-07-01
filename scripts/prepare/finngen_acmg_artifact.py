#!/usr/bin/env python3
"""
Prepare artifact for FinnGen R12 ACMG figure.

Input:  /mnt/home/ryo/finngen_r12_actionable.tsv  (canonical source)
Output: artifacts/finngen_acmg.feather

Extracts the 6 candidate ClinVar variants, parses mixed text/numeric
ACMG criteria columns, and writes a clean feather file for the figure script.

Run:
    uv run python scripts/prepare/finngen_acmg_artifact.py
"""
import re
import sys
from pathlib import Path

import polars as pl
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC  = Path("/mnt/home/ryo/finngen_r12_actionable.tsv")
OUT  = ROOT / "artifacts" / "finngen_acmg.feather"

CANDIDATE_IDS = {
    "chr4:87845065:GGAAA:G",   # MEPE
    "chr11:46728029:C:G",       # F2 p.Leu389Val
    "chr11:46726097:G:T",       # F2 chr11 missense
    "chr8:54621480:T:G",        # RP1
    "chr19:50276162:G:A",       # MYH14
    "chr2:178527204:A:T",       # TTN
}

DISPLAY_LABELS = {
    "chr4:87845065:GGAAA:G":  "MEPE\np.Lys70IlefsTer26",
    "chr11:46728029:C:G":     "F2\np.Leu389Val",
    "chr11:46726097:G:T":     "F2\nchr11:46726097",
    "chr8:54621480:T:G":      "RP1\np.Leu172Arg",
    "chr19:50276162:G:A":     "MYH14\np.Glu1214Lys",
    "chr2:178527204:A:T":     "TTN\np.Val35195Glu",
}

BATCH = {
    "chr4:87845065:GGAAA:G":  1,
    "chr11:46728029:C:G":     1,
    "chr8:54621480:T:G":      1,
    "chr19:50276162:G:A":     1,
    "chr11:46726097:G:T":     2,
    "chr2:178527204:A:T":     2,
}


def parse_pts(val, col: str) -> float:
    if pd.isna(val):
        return 0.0
    v = str(val).strip()
    if not v or v in ("nan", "ND"):
        return 0.0
    try:
        return float(v)
    except ValueError:
        pass
    m = re.search(r"\((\d+(?:\.\d+)?)\)", v)
    if m:
        return float(m.group(1))
    if re.search(r"none|het_flag|blocked", v, re.I):
        return 0.0
    if col == "ACMG_PS4":
        if re.search(r"strong",     v, re.I): return 4.0
        if re.search(r"moderate",   v, re.I): return 2.0
        if re.search(r"supporting", v, re.I): return 1.0
    if col == "ACMG_PP3" and re.search(r"moderate", v, re.I):
        return 1.0
    if col == "ACMG_PM2" and re.search(r"supporting", v, re.I):
        return 1.0
    if col == "ACMG_PVS1" and re.search(r"pvs1", v, re.I):
        m2 = re.search(r"(\d+)", v)
        return float(m2.group(1)) if m2 else 8.0
    return 0.0


def main():
    df = pd.read_csv(SRC, sep="\t")
    df = df[df["variant_id"].isin(CANDIDATE_IDS)].copy()

    for col in ["ACMG_PVS1", "ACMG_PS4", "ACMG_PP3", "ACMG_PM2"]:
        df[f"{col}_pts"] = df[col].apply(lambda x: parse_pts(x, col))

    df["total_pts"] = (df["ACMG_PVS1_pts"] + df["ACMG_PS4_pts"]
                       + df["ACMG_PP3_pts"] + df["ACMG_PM2_pts"])
    df["display_label"] = df["variant_id"].map(DISPLAY_LABELS)
    df["submission_batch"] = df["variant_id"].map(BATCH)
    df = df.sort_values("total_pts", ascending=False).reset_index(drop=True)

    keep = ["gene", "HGVS", "variant_id", "display_label",
            "ACMG_PVS1_pts", "ACMG_PS4_pts", "ACMG_PP3_pts", "ACMG_PM2_pts",
            "total_pts", "ACMG_computed", "submission_batch",
            "FinnGen_OR", "FinnGen_pval", "EVEE_pathogenicity",
            "ClinVar_stars", "ClinVar_accession"]
    df = df[keep]

    pl.from_pandas(df).write_ipc(OUT)
    print(f"Written → {OUT}  ({len(df)} variants)")
    print(df[["gene", "display_label", "ACMG_PVS1_pts",
              "ACMG_PS4_pts", "ACMG_PP3_pts", "ACMG_PM2_pts",
              "total_pts"]].to_string(index=False))


if __name__ == "__main__":
    main()
