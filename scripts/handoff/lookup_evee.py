#!/usr/bin/env python3
"""Stage 1: look up goodfire handoff variants against the public EVEE API.

For each row in the input CSV:
  - Build variant_id = chr{c}:{pos-1}:{ref}:{alt} (EVEE uses 0-based positions).
  - GET /variants/{id}. On 200 keep pathogenicity + interpretation + full raw JSON.
  - If the variant is found but no stored interpretation, trigger /analysis and poll.
  - On 404 mark missing; writes a harvest-ready manifest of the missing set.

All output lands under $OUT_DIR (default /mnt/data/artifacts/ryo/goodfire_handoff):
  - evee_lookup.parquet        — one row per input variant
  - missing_variants_manifest.csv — harvest pipeline input for not-found variants
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

import httpx
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "handoff"))

from evee_client import (
    evee_url,
    extract_interpretation,
    get_variant,
    interpretation_from_analysis,
    make_client,
    wait_for_analysis,
)

DEFAULT_INPUT = Path("/mnt/home/ryo/goodfire_handoff_all_rare_variants.csv")
DEFAULT_OUT = Path("/mnt/data/artifacts/ryo/goodfire_handoff")

ALIGNED_SCORE_KEYS = [
    "eff_pathogenic",
    "eff_alphamissense_c",
    "eff_cadd_c",
    "eff_revel_c",
    "eff_sift_c",
    "eff_polyphen_c",
    "eff_spliceai_max_c",
    "eff_clinpred_c",
    "eff_bayesdel_c",
]


def build_variant_id(chrom: str, position: int, ref: str, alt: str) -> str:
    # EVEE is 0-based; handoff CSV positions are 1-based (VCF/HGVS).
    return f"{chrom}:{position - 1}:{ref}:{alt}"


def _lookup_one(client: httpx.Client, variant_id: str, wait_interp: bool, interp_timeout: float) -> dict:
    """Hit /variants/{id}, optionally polling /analysis. Returns a result row."""
    t0 = time.monotonic()
    status, data = get_variant(client, variant_id)
    row = {
        "evee_variant_id": variant_id,
        "evee_http_status": status,
        "evee_found": status == 200,
        "evee_url": evee_url(variant_id),
        "pathogenicity": None,
        "interpretation_source": None,
        "interpretation_summary": None,
        "interpretation_mechanism": None,
        "interpretation_key_evidence": None,
        "interpretation_confidence": None,
        "evee_response_json": None,
        "lookup_seconds": None,
    }
    for k in ALIGNED_SCORE_KEYS:
        row[k] = None

    if data is None:
        row["lookup_seconds"] = round(time.monotonic() - t0, 2)
        return row

    row["pathogenicity"] = data.get("pathogenicity")
    for k in ALIGNED_SCORE_KEYS:
        row[k] = data.get(k)

    interp = extract_interpretation(data)
    if interp:
        row["interpretation_source"] = "stored"
    elif wait_interp:
        analysis = wait_for_analysis(client, variant_id, overall_timeout=interp_timeout)
        interp = interpretation_from_analysis(analysis)
        if interp:
            row["interpretation_source"] = "on_demand"
        else:
            row["interpretation_source"] = analysis.get("status", "unavailable")

    if interp:
        row["interpretation_summary"] = interp.get("summary")
        row["interpretation_mechanism"] = interp.get("mechanism")
        ke = interp.get("key_evidence")
        row["interpretation_key_evidence"] = json.dumps(ke) if ke is not None else None
        row["interpretation_confidence"] = interp.get("confidence")

    row["evee_response_json"] = json.dumps(data)
    row["lookup_seconds"] = round(time.monotonic() - t0, 2)
    return row


def run(
    input_csv: Path,
    out_dir: Path,
    limit: int | None,
    workers: int,
    wait_interp: bool,
    interp_timeout: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    input_df = pl.read_csv(input_csv, schema_overrides={"position": pl.Int64})
    if limit is not None:
        input_df = input_df.head(limit)

    variant_ids = [
        build_variant_id(c, p, r, a)
        for c, p, r, a in zip(
            input_df["chromosome"], input_df["position"], input_df["ref"], input_df["alt"], strict=True
        )
    ]

    print(f"[lookup] {len(variant_ids)} variants → EVEE API, workers={workers}, wait_interp={wait_interp}", flush=True)

    # One client per thread — httpx.Client isn't threadsafe for concurrent requests on one instance.
    def _worker(vid: str) -> dict:
        with make_client() as c:
            return _lookup_one(c, vid, wait_interp, interp_timeout)

    results: list[dict] = [None] * len(variant_ids)  # type: ignore
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_worker, vid): i for i, vid in enumerate(variant_ids)}
        done = 0
        for fut in cf.as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                print(f"[lookup] variant {variant_ids[i]} errored: {e!r}", flush=True)
                results[i] = {
                    "evee_variant_id": variant_ids[i],
                    "evee_http_status": -1,
                    "evee_found": False,
                    "evee_url": evee_url(variant_ids[i]),
                    "pathogenicity": None,
                    "interpretation_source": f"error:{type(e).__name__}",
                    "interpretation_summary": None,
                    "interpretation_mechanism": None,
                    "interpretation_key_evidence": None,
                    "interpretation_confidence": None,
                    "evee_response_json": None,
                    "lookup_seconds": None,
                }
                for k in ALIGNED_SCORE_KEYS:
                    results[i][k] = None
            done += 1
            if done % 20 == 0 or done == len(variant_ids):
                found = sum(1 for r in results if r and r.get("evee_found"))
                print(f"[lookup] {done}/{len(variant_ids)} done, {found} found", flush=True)

    result_df = pl.DataFrame(results)
    merged = pl.concat([input_df, result_df], how="horizontal")

    out_parquet = out_dir / "evee_lookup.parquet"
    merged.write_parquet(out_parquet)
    print(f"[lookup] wrote {out_parquet}", flush=True)

    missing = merged.filter(~pl.col("evee_found"))
    manifest_rows = [
        {
            "variant_id": vid,
            "chrom": c.removeprefix("chr"),
            "pos": p - 1,
            "ref": r,
            "alt": a,
            "label": "other",
        }
        for vid, c, p, r, a in zip(
            missing["evee_variant_id"],
            missing["chromosome"],
            missing["position"],
            missing["ref"],
            missing["alt"],
            strict=True,
        )
    ]
    manifest_path = out_dir / "missing_variants_manifest.csv"
    if manifest_rows:
        pl.DataFrame(manifest_rows).write_csv(manifest_path)
    else:
        # Still emit an empty header-only file for pipeline symmetry.
        pl.DataFrame(schema={"variant_id": pl.Utf8, "chrom": pl.Utf8, "pos": pl.Int64, "ref": pl.Utf8, "alt": pl.Utf8, "label": pl.Utf8}).write_csv(manifest_path)
    print(f"[lookup] wrote {manifest_path} ({len(manifest_rows)} missing)", flush=True)

    # Terse summary
    total = merged.height
    n_found = int(merged["evee_found"].sum())
    n_interp_stored = int((merged["interpretation_source"] == "stored").sum())
    n_interp_on_demand = int((merged["interpretation_source"] == "on_demand").sum())
    print(
        f"[lookup] summary: {n_found}/{total} found, "
        f"{n_interp_stored} stored interpretations, {n_interp_on_demand} on-demand",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--limit", type=int, default=None, help="process only first N rows")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--wait-interp", action="store_true", help="poll /analysis for variants without stored interp")
    p.add_argument("--interp-timeout", type=float, default=300.0)
    args = p.parse_args()
    run(args.input, args.out_dir, args.limit, args.workers, args.wait_interp, args.interp_timeout)


if __name__ == "__main__":
    main()
