#!/usr/bin/env python3
"""
Prepare UMAP coordinates for SNV covariance-probe embeddings (Figure 1E).

Loads 30K test-set embeddings from covariance64_pool, computes PCA(100) + UMAP,
saves coords and metadata for downstream plotting.

Input:  data/clinvar/evo2-7b/deconfounded/covariance64_pool/
Output: artifacts/umap_snv.feather  (coords, pathogenic)
        artifacts/umap_snv_meta.feather  (consequence strings)

Usage:
    python scripts/prepare/umap_snv.py [--force]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import safetensors.torch
import torch
from goodfire_core.storage import ActivationDataset, FilesystemStorage
from sklearn.decomposition import PCA
from umap import UMAP

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
DATASET = ROOT / "data" / "clinvar" / "evo2-7b" / "deconfounded"
EMBED_DIR = DATASET / "covariance64_pool"
OUT_DIR = ROOT / "artifacts"

CONSEQ_MAP = {
    "missense_variant": "Missense",
    "synonymous_variant": "Synonymous",
    "nonsense": "Nonsense",
    "stop_gained": "Nonsense",
    "splice_donor_variant": "Splice",
    "splice_acceptor_variant": "Splice",
    "splice_region_variant": "Splice",
    "5_prime_UTR_variant": "UTR",
    "3_prime_UTR_variant": "UTR",
    "intron_variant": "Intronic",
    "initiator_codon_variant": "Other",
    "non-coding_transcript_variant": "Other",
    "stop_lost": "Other",
    "genic_upstream_transcript_variant": "Other",
    "genic_downstream_transcript_variant": "Other",
}

N_SAMPLE = 30_000


def main():
    parser = argparse.ArgumentParser(description="Prepare SNV UMAP coordinates")
    parser.add_argument("--force", action="store_true", help="Recompute even if output exists")
    args = parser.parse_args()

    out_tensors = OUT_DIR / "umap_snv.safetensors"
    out_meta = OUT_DIR / "umap_snv_meta.feather"

    if out_tensors.exists() and out_meta.exists() and not args.force:
        print(f"Output already exists: {out_tensors}")
        print("Use --force to recompute.")
        return

    # Load test-set IDs and metadata from local feathers
    split = pl.read_ipc(ARTIFACTS / "split_deconfounded.feather")
    test_ids = set(split.filter(pl.col("split") == "test")["variant_id"].to_list())
    meta = pl.read_ipc(ARTIFACTS / "metadata_deconfounded.feather").with_columns(
        (pl.col("label") == "pathogenic").cast(pl.Int32).alias("pathogenic"),
    )

    # Random subsample for UMAP
    rng = np.random.RandomState(42)
    test_list = list(test_ids)
    rng.shuffle(test_list)
    sample_ids = test_list[:min(N_SAMPLE, len(test_list))]
    print(f"Sampled {len(sample_ids):,} from {len(test_ids):,} test variants")

    # Load embeddings via training_iterator with sequence_ids filter
    storage = FilesystemStorage(EMBED_DIR.parent)
    ds = ActivationDataset(storage, EMBED_DIR.name, batch_size=4096, include_provenance=True)

    embeddings, ids = [], []
    for batch in ds.training_iterator(device="cpu", sequence_ids=sample_ids,
                                       shuffle=False, drop_last=False):
        embeddings.append(batch.acts.flatten(1).float().numpy())
        ids.extend(batch.sequence_ids)

    X = np.concatenate(embeddings)
    print(f"Loaded {X.shape[0]:,} embeddings, shape {X.shape}")

    # Left join to preserve embedding order
    df = pl.DataFrame({"variant_id": ids}).join(
        meta.select("variant_id", "pathogenic", "consequence"), on="variant_id", how="left",
    )
    y = df["pathogenic"].to_numpy()
    conseq = np.array([CONSEQ_MAP.get(c, "Other") for c in df["consequence"].to_list()])

    # PCA -> 100, UMAP with correlation metric
    print(f"PCA 4096 -> 100 on {len(X)} variants...")
    pca = PCA(n_components=100, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    print("Computing UMAP (n=40, d=0.03, correlation)...")
    reducer = UMAP(n_components=2, n_neighbors=40, min_dist=0.03,
                   metric="correlation", random_state=42)
    coords = reducer.fit_transform(X_pca)

    # Save: tensors in safetensors, strings in feather
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file({
        "coords": torch.from_numpy(coords),
        "pathogenic": torch.from_numpy(y),
    }, str(out_tensors))
    pl.DataFrame({"consequence": conseq}).write_ipc(out_meta)
    print(f"Saved UMAP to {OUT_DIR}")


if __name__ == "__main__":
    main()
