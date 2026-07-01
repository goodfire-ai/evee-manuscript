"""Split builds/v5/clean.parquet into ~equal-sized shards by chromosome.

Output: one parquet file per shard, plus a manifest.json describing the mapping.
Use the shards as the upload artifact for Zenodo (single-PUT 7–9 GB is reliable;
35 GB is not).

Consumers can read all shards as one logical table:
    import polars as pl
    df = pl.scan_parquet("clean_shard_*.parquet")

Usage:
    uv run python scripts/shard_for_zenodo.py \\
        --input  /mnt/data/artifacts/public/variant-viewer/builds/v5/clean.parquet \\
        --output /mnt/data/artifacts/ryo/zenodo_shards \\
        --n-shards 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import polars as pl
from rich.console import Console

CONSOLE = Console()


def plan_shards(counts: dict[str, int], n_shards: int) -> list[list[str]]:
    """Greedy bin-pack chromosomes into `n_shards` balanced groups by row count."""
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    bins: list[list[str]] = [[] for _ in range(n_shards)]
    loads = [0] * n_shards
    for chrom, n in ordered:
        i = min(range(n_shards), key=lambda j: loads[j])
        bins[i].append(chrom)
        loads[i] += n
    return bins


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="Output directory (will be created).")
    p.add_argument("--n-shards", type=int, default=5)
    p.add_argument("--chrom-col", default="chrom")
    args = p.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"not a file: {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)

    CONSOLE.rule("Scan + count by chromosome")
    t0 = time.time()
    counts_df = (
        pl.scan_parquet(args.input)
        .group_by(args.chrom_col)
        .len()
        .collect(engine="streaming")
        .sort(args.chrom_col)
    )
    counts = dict(zip(counts_df[args.chrom_col].to_list(), counts_df["len"].to_list()))
    CONSOLE.print(f"chromosomes: {len(counts)}, total rows: {sum(counts.values()):,} "
                  f"({time.time() - t0:.1f} s)")

    CONSOLE.rule("Plan shards")
    bins = plan_shards(counts, args.n_shards)
    manifest = {"shards": [], "n_shards": args.n_shards, "input": str(args.input),
                "chrom_col": args.chrom_col}
    for i, bin_chroms in enumerate(bins):
        n = sum(counts[c] for c in bin_chroms)
        CONSOLE.print(f"  shard {i}: {len(bin_chroms)} chroms, {n:,} rows — {sorted(bin_chroms)}")
        manifest["shards"].append({
            "index": i,
            "filename": f"clean_shard_{i}.parquet",
            "chromosomes": sorted(bin_chroms),
            "row_count": n,
        })

    CONSOLE.rule("Write shards (streaming)")
    for shard in manifest["shards"]:
        out_path = args.output / shard["filename"]
        if out_path.is_file():
            CONSOLE.print(f"  [yellow]skip {out_path} (exists)[/yellow]")
            continue
        t0 = time.time()
        CONSOLE.print(f"  → {out_path.name} ({shard['row_count']:,} rows) ...")
        (
            pl.scan_parquet(args.input)
            .filter(pl.col(args.chrom_col).is_in(shard["chromosomes"]))
            .sink_parquet(out_path, compression="zstd")
        )
        size_gb = out_path.stat().st_size / 1e9
        shard["size_bytes"] = out_path.stat().st_size
        CONSOLE.print(f"    done in {time.time() - t0:.0f} s, {size_gb:.2f} GB on disk")

    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    CONSOLE.rule("Done")
    CONSOLE.print(f"Manifest: {manifest_path}")
    total_gb = sum(s.get("size_bytes", 0) for s in manifest["shards"]) / 1e9
    CONSOLE.print(f"Total shard size: {total_gb:.2f} GB (input was "
                  f"{args.input.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
