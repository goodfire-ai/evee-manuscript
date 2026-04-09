#!/usr/bin/env python3
"""
Prepare UMAP coordinates for combined SNV + indel + VUS covariance embeddings.

Loads 10K each of labeled SNVs (test set), indels, and VUS. L2-normalizes,
computes PCA(100) + UMAP, saves coords and metadata.

Input:  data/clinvar/evo2-7b/{labeled,indels,vus}/covariance64_pool/
Output: artifacts/umap_combined.feather  (coords, pathogenic)
        artifacts/umap_combined_meta.feather  (csq, variant_type)

Usage:
    python scripts/prepare/umap_combined.py [--force]
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
MAYO = ROOT / "data" / "clinvar" / "evo2-7b"
LABELED = MAYO / "labeled"
INDELS = MAYO / "indels"
VUS = MAYO / "vus"
OUT_DIR = ROOT / "artifacts"

CONSEQ_MAP = {
    "missense_variant": "Missense",
    "synonymous_variant": "Synonymous",
    "nonsense": "Nonsense",
    "stop_gained": "Nonsense",
    "splice_donor_variant": "Splice",
    "splice_acceptor_variant": "Splice",
    "intron_variant": "Intronic",
    "5_prime_UTR_variant": "UTR",
    "3_prime_UTR_variant": "UTR",
    "non-coding_transcript_variant": "Other",
    "initiator_codon_variant": "Other",
    "stop_lost": "Other",
    "frameshift_variant": "Frameshift",
    "inframe_deletion": "In-frame",
    "inframe_insertion": "In-frame",
}

N_PER = 10_000


def _load_embeddings(embed_dir: Path, sample_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """Load and flatten covariance embeddings for given IDs."""
    storage = FilesystemStorage(embed_dir.parent)
    ds = ActivationDataset(storage, embed_dir.name, batch_size=4096, include_provenance=True)

    embeddings, ids = [], []
    for batch in ds.training_iterator(device="cpu", sequence_ids=sample_ids,
                                       shuffle=False, drop_last=False):
        embeddings.append(batch.acts.flatten(1).float().numpy())
        ids.extend(batch.sequence_ids)
    return np.concatenate(embeddings), ids


def main():
    parser = argparse.ArgumentParser(description="Prepare combined UMAP coordinates")
    parser.add_argument("--force", action="store_true", help="Recompute even if output exists")
    args = parser.parse_args()

    out_tensors = OUT_DIR / "umap_combined.safetensors"
    out_meta = OUT_DIR / "umap_combined_meta.feather"

    if out_tensors.exists() and out_meta.exists() and not args.force:
        print(f"Output already exists: {out_tensors}")
        print("Use --force to recompute.")
        return

    rng = np.random.RandomState(42)

    # --- Labeled SNVs (test set only) ---
    split = pl.read_ipc(ARTIFACTS / "split_deconfounded.feather")
    test_ids = split.filter(pl.col("split") == "test")["variant_id"].to_list()
    rng.shuffle(test_ids)
    labeled_sample = test_ids[:N_PER]

    labeled_meta = pl.read_ipc(ARTIFACTS / "metadata_deconfounded.feather").with_columns(
        (pl.col("label") == "pathogenic").cast(pl.Int32).alias("pathogenic"),
    )

    X_labeled, ids_labeled = _load_embeddings(LABELED / "covariance64_pool", labeled_sample)
    labeled_df = pl.DataFrame({"variant_id": ids_labeled}).join(
        labeled_meta.select("variant_id", "pathogenic", "consequence"), on="variant_id", how="left",
    ).with_columns(pl.lit("SNV").alias("variant_type"))

    # --- Indels ---
    indel_meta = pl.read_ipc(ARTIFACTS / "metadata_labeled_indels.feather").with_columns(
        (pl.col("label") == "pathogenic").cast(pl.Int32).alias("pathogenic"),
    )
    indel_all = indel_meta["variant_id"].to_list()
    rng.shuffle(indel_all)
    indel_sample = indel_all[:min(N_PER, len(indel_all))]

    X_indel, ids_indel = _load_embeddings(INDELS / "covariance64_pool", indel_sample)
    indel_df = pl.DataFrame({"variant_id": ids_indel}).join(
        indel_meta.select("variant_id", "pathogenic", "consequence"), on="variant_id", how="left",
    ).with_columns(pl.lit("Indel").alias("variant_type"))

    # --- VUS ---
    vus_meta = pl.read_ipc(ARTIFACTS / "metadata_vus.feather")
    vus_all = vus_meta["variant_id"].to_list()
    rng.shuffle(vus_all)
    vus_sample = vus_all[:min(N_PER, len(vus_all))]

    X_vus, ids_vus = _load_embeddings(VUS / "covariance64_pool", vus_sample)
    vus_df = pl.DataFrame({"variant_id": ids_vus}).join(
        vus_meta.select("variant_id", "consequence"), on="variant_id", how="left",
    ).with_columns(
        pl.lit(-1).cast(pl.Int32).alias("pathogenic"),
        pl.lit("VUS").alias("variant_type"),
    ).select("variant_id", "pathogenic", "consequence", "variant_type")

    # --- Combine and compute UMAP ---
    joint_df = pl.concat([labeled_df, indel_df, vus_df]).with_columns(
        pl.col("consequence").replace(CONSEQ_MAP).fill_null("Other").alias("csq"),
    )
    X = np.concatenate([X_labeled, X_indel, X_vus])

    # L2-normalize (VUS have different norms)
    X_normed = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    print(f"PCA on {X_normed.shape[0]:,} embeddings...")
    pca = PCA(n_components=100, random_state=42)
    X_pca = pca.fit_transform(X_normed)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    print("Computing UMAP...")
    reducer = UMAP(n_components=2, n_neighbors=40, min_dist=0.03,
                   metric="correlation", random_state=42)
    coords = reducer.fit_transform(X_pca)

    # Save: tensors in safetensors, strings in feather
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file({
        "coords": torch.from_numpy(coords),
        "pathogenic": torch.from_numpy(joint_df["pathogenic"].to_numpy()),
    }, str(out_tensors))
    joint_df.select("csq", "variant_type").write_ipc(out_meta)
    print(f"Saved UMAP to {OUT_DIR}")


if __name__ == "__main__":
    main()
