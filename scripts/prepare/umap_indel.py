#!/usr/bin/env python3
"""
Prepare UMAP coordinates for indel covariance-probe embeddings (Suppl. Fig 5).

Loads up to 30K indel embeddings from covariance64_pool, computes PCA(100) + UMAP,
saves coords and metadata for downstream plotting.

Input:  data/clinvar/evo2-7b/indels/covariance64_pool/
Output: data/embeddings/umap_indel.safetensors  (coords, pathogenic)
        data/embeddings/umap_indel_meta.feather  (csq_type strings)

Usage:
    python scripts/prepare/umap_indel.py [--force]
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
PANELS = ROOT / "artifacts"
EMBED_DIR = ROOT / "data" / "clinvar" / "evo2-7b" / "indels" / "covariance64_pool"
OUT_DIR = ROOT / "artifacts"

CONSEQ_MAP = {
    "frameshift_variant": "Frameshift",
    "inframe_deletion": "In-frame del",
    "inframe_insertion": "In-frame ins",
    "intron_variant": "Intronic",
    "nonsense": "Nonsense",
    "splice_donor_variant": "Splice",
    "splice_acceptor_variant": "Splice",
    "3_prime_UTR_variant": "UTR",
    "5_prime_UTR_variant": "UTR",
    "non-coding_transcript_variant": "Other",
}

N_SAMPLE = 30_000


def main():
    parser = argparse.ArgumentParser(description="Prepare indel UMAP coordinates")
    parser.add_argument("--force", action="store_true", help="Recompute even if output exists")
    args = parser.parse_args()

    out_tensors = OUT_DIR / "umap_indel.safetensors"
    out_meta = OUT_DIR / "umap_indel_meta.feather"

    if out_tensors.exists() and out_meta.exists() and not args.force:
        print(f"Output already exists: {out_tensors}")
        print("Use --force to recompute.")
        return

    # Load metadata from local feather
    meta = pl.read_ipc(PANELS / "metadata_labeled_indels.feather").with_columns(
        (pl.col("label") == "pathogenic").cast(pl.Int32).alias("pathogenic"),
        pl.col("consequence").replace(CONSEQ_MAP).alias("csq_type"),
    )

    # Subsample for UMAP
    rng = np.random.RandomState(42)
    all_ids = meta["variant_id"].to_list()
    rng.shuffle(all_ids)
    sample_ids = all_ids[:min(N_SAMPLE, len(all_ids))]

    # Load embeddings
    storage = FilesystemStorage(EMBED_DIR.parent)
    ds = ActivationDataset(storage, EMBED_DIR.name, batch_size=4096, include_provenance=True)

    embeddings, ids = [], []
    for batch in ds.training_iterator(device="cpu", sequence_ids=sample_ids,
                                       shuffle=False, drop_last=False):
        embeddings.append(batch.acts.flatten(1).float().numpy())
        ids.extend(batch.sequence_ids)

    X = np.concatenate(embeddings)
    print(f"Loaded {X.shape[0]:,} embeddings, shape {X.shape}")

    # Join metadata in embedding order
    umap_df = pl.DataFrame({"variant_id": ids}).join(
        meta.select("variant_id", "pathogenic", "csq_type"),
        on="variant_id", how="left",
    )

    # PCA -> UMAP
    print(f"PCA 100 on {len(X)} embeddings...")
    pca = PCA(n_components=100, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"  PCA 100: {pca.explained_variance_ratio_.sum():.1%} variance")

    print("Computing UMAP (n=40, d=0.03, correlation)...")
    reducer = UMAP(n_components=2, n_neighbors=40, min_dist=0.03,
                   metric="correlation", random_state=42)
    coords = reducer.fit_transform(X_pca)

    # Save: tensors in safetensors, strings in feather
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file({
        "coords": torch.from_numpy(coords),
        "pathogenic": torch.from_numpy(umap_df["pathogenic"].to_numpy()),
    }, str(out_tensors))
    umap_df.select("csq_type").write_ipc(out_meta)
    print(f"Saved UMAP to {OUT_DIR}")


if __name__ == "__main__":
    main()
