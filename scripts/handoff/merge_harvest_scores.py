#!/usr/bin/env python3
"""Stage 3: join harvest probe scores into evee_lookup.parquet.

After the harvest → embed pipeline finishes, it writes scores.feather with
columns [variant_id, score] under $HARVEST_OUT/clinvar-evo2-probe-v1/. This
script left-joins those onto the lookup parquet, filling in pathogenicity for
rows where evee_found=False.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

DEFAULT_LOOKUP = Path("/mnt/data/artifacts/ryo/goodfire_handoff/evee_lookup.parquet")
DEFAULT_SCORES = Path("/mnt/data/artifacts/ryo/goodfire_handoff/harvest/probe_v3/scores.feather")
DEFAULT_OUT = Path("/mnt/data/artifacts/ryo/goodfire_handoff/handoff_final.parquet")


def run(lookup: Path, scores: Path, out: Path) -> None:
    lookup_df = pl.read_parquet(lookup)
    scores_df = pl.read_ipc(scores).rename({"score": "harvest_pathogenicity", "variant_id": "evee_variant_id"})

    merged = lookup_df.join(scores_df, on="evee_variant_id", how="left")
    merged = merged.with_columns(
        pl.coalesce([pl.col("pathogenicity"), pl.col("harvest_pathogenicity")]).alias("pathogenicity_final"),
        pl.when(pl.col("evee_found")).then(pl.lit("evee_api"))
          .when(pl.col("harvest_pathogenicity").is_not_null()).then(pl.lit("local_harvest"))
          .otherwise(pl.lit("missing")).alias("score_source"),
    )

    merged.write_parquet(out)
    print(f"[merge] wrote {out}", flush=True)
    print(merged.group_by("score_source").len().sort("score_source"), flush=True)
    missing_final = merged.filter(pl.col("score_source") == "missing")
    if missing_final.height:
        print(f"[merge] WARNING: {missing_final.height} rows still missing:", flush=True)
        print(missing_final.select(["gene", "evee_variant_id"]), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lookup", type=Path, default=DEFAULT_LOOKUP)
    p.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    run(args.lookup, args.scores, args.out)


if __name__ == "__main__":
    main()
