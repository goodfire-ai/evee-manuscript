#!/usr/bin/env python3
"""
Prepare UMAP coordinates for subsampled combined SNV + indel + VUS embeddings.

Loads raw covariance64_pool embeddings directly from chunked safetensors
(no goodfire_core dependency). Subsamples to specified counts per group,
L2-normalizes, runs PCA(100) + UMAP, saves coords and metadata.

Subsample targets:
  - 10K benign SNV, 10K pathogenic SNV
  - 5K VUS
  - 1K benign indel, 1K pathogenic indel

Input:  data/clinvar/evo2-7b/{labeled,indels,vus}/covariance64_pool/
Output: artifacts/umap_combined.feather  (coords, pathogenic)
        artifacts/umap_combined_meta.feather  (csq, variant_type)

Usage:
    python scripts/prepare/umap_combined_subsample.py [--force]
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import polars as pl
import safetensors.numpy as sfnp
from sklearn.decomposition import PCA
from umap import UMAP

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
MAYO = ROOT / "data" / "clinvar" / "evo2-7b"
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


def _bf16_to_f32(raw: bytes, shape: tuple) -> np.ndarray:
    """Convert raw bfloat16 bytes to float32 numpy array."""
    # bfloat16 is the upper 16 bits of float32, so pad with 2 zero bytes
    u16 = np.frombuffer(raw, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32).reshape(shape)


def _load_embeddings_direct(embed_dir: Path, variant_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """Load covariance embeddings from chunked safetensors via SQLite index."""
    db_path = embed_dir / "index.sqlite"
    meta_path = embed_dir / "metadata.json"
    chunks_dir = embed_dir / "chunks"

    conn = sqlite3.connect(str(db_path))

    # Query locations for requested IDs
    placeholders = ",".join("?" * len(variant_ids))
    rows = conn.execute(
        f"SELECT sequence_id, chunk_id, offset FROM sequence_locations "
        f"WHERE sequence_id IN ({placeholders})",
        variant_ids,
    ).fetchall()
    conn.close()

    # Group by chunk for efficient loading
    by_chunk = {}
    for seq_id, chunk_id, offset in rows:
        by_chunk.setdefault(chunk_id, []).append((seq_id, offset))

    embeddings = []
    found_ids = []
    for chunk_id in sorted(by_chunk):
        chunk_path = chunks_dir / f"chunk_{chunk_id:06d}" / "activations.safetensors"

        # Read raw bytes for bfloat16 tensor
        from safetensors import safe_open
        with safe_open(str(chunk_path), framework="numpy") as f:
            # Get the raw tensor bytes and shape
            chunk_meta = json.loads(f.metadata().get("num_items", "0")) if f.metadata() else 0

        # Read raw bfloat16 bytes directly from safetensors file
        with open(chunk_path, "rb") as fh:
            header_size = int.from_bytes(fh.read(8), "little")
            header = json.loads(fh.read(header_size))
            data_start = 8 + header_size

            act_info = header["activations"]
            act_offset = act_info["data_offsets"]
            act_shape = act_info["shape"]  # [N, 64, 64]
            item_elems = act_shape[1] * act_shape[2]  # 64 * 64 = 4096
            item_bytes = item_elems * 2  # bf16 = 2 bytes each

            for seq_id, offset in by_chunk[chunk_id]:
                fh.seek(data_start + act_offset[0] + offset * item_bytes)
                raw = fh.read(item_bytes)
                if len(raw) < item_bytes:
                    continue  # skip if at end of chunk
                emb = _bf16_to_f32(raw, (item_elems,))
                embeddings.append(emb)
                found_ids.append(seq_id)

    return np.stack(embeddings), found_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_tensors = OUT_DIR / "umap_combined.safetensors"
    out_meta = OUT_DIR / "umap_combined_meta.feather"

    if out_tensors.exists() and out_meta.exists() and not args.force:
        print(f"Output already exists: {out_tensors}")
        print("Use --force to recompute.")
        return

    rng = np.random.RandomState(42)

    # --- Sample labeled SNVs ---
    labeled_meta = pl.read_ipc(ARTIFACTS / "metadata_deconfounded.feather")
    benign_ids = labeled_meta.filter(pl.col("label") == "benign")["variant_id"].to_list()
    path_ids = labeled_meta.filter(pl.col("label") == "pathogenic")["variant_id"].to_list()

    rng.shuffle(benign_ids)
    rng.shuffle(path_ids)
    snv_benign = benign_ids[:10_000]
    snv_path = path_ids[:10_000]
    snv_sample = snv_benign + snv_path
    print(f"SNV: {len(snv_benign)} benign + {len(snv_path)} pathogenic = {len(snv_sample)}")

    X_snv, ids_snv = _load_embeddings_direct(
        MAYO / "labeled" / "covariance64_pool", snv_sample)
    snv_df = pl.DataFrame({"variant_id": ids_snv}).join(
        labeled_meta.select("variant_id", "label", "consequence"), on="variant_id", how="left",
    ).with_columns(
        (pl.col("label") == "pathogenic").cast(pl.Int32).alias("pathogenic"),
        pl.lit("SNV").alias("variant_type"),
    )
    print(f"  Loaded {len(ids_snv)} SNV embeddings")

    # --- Sample indels ---
    indel_meta = pl.read_ipc(ARTIFACTS / "metadata_labeled_indels.feather")
    indel_benign = indel_meta.filter(pl.col("label") == "benign")["variant_id"].to_list()
    indel_path = indel_meta.filter(pl.col("label") == "pathogenic")["variant_id"].to_list()

    rng.shuffle(indel_benign)
    rng.shuffle(indel_path)
    indel_b = indel_benign[:1_000]
    indel_p = indel_path[:1_000]
    indel_sample = indel_b + indel_p
    print(f"Indel: {len(indel_b)} benign + {len(indel_p)} pathogenic = {len(indel_sample)}")

    X_indel, ids_indel = _load_embeddings_direct(
        MAYO / "indels" / "covariance64_pool", indel_sample)
    indel_df = pl.DataFrame({"variant_id": ids_indel}).join(
        indel_meta.select("variant_id", "label", "consequence"), on="variant_id", how="left",
    ).with_columns(
        (pl.col("label") == "pathogenic").cast(pl.Int32).alias("pathogenic"),
        pl.lit("Indel").alias("variant_type"),
    )
    print(f"  Loaded {len(ids_indel)} indel embeddings")

    # --- Sample VUS ---
    vus_meta = pl.read_ipc(ARTIFACTS / "metadata_vus.feather")
    vus_all = vus_meta["variant_id"].to_list()
    rng.shuffle(vus_all)
    vus_sample = vus_all[:5_000]
    print(f"VUS: {len(vus_sample)}")

    X_vus, ids_vus = _load_embeddings_direct(
        MAYO / "vus" / "covariance64_pool", vus_sample)
    vus_df = pl.DataFrame({"variant_id": ids_vus}).join(
        vus_meta.select("variant_id", "consequence"), on="variant_id", how="left",
    ).with_columns(
        pl.lit(-1).cast(pl.Int32).alias("pathogenic"),
        pl.lit("VUS").alias("variant_type"),
        pl.lit("VUS").alias("label"),
    )
    print(f"  Loaded {len(ids_vus)} VUS embeddings")

    # --- Combine ---
    joint_df = pl.concat([
        snv_df.select("variant_id", "pathogenic", "consequence", "variant_type"),
        indel_df.select("variant_id", "pathogenic", "consequence", "variant_type"),
        vus_df.select("variant_id", "pathogenic", "consequence", "variant_type"),
    ]).with_columns(
        pl.col("consequence").replace_strict(CONSEQ_MAP, default="Other").alias("csq"),
    )
    X = np.concatenate([X_snv, X_indel, X_vus])
    print(f"\nTotal: {X.shape[0]:,} embeddings, dim={X.shape[1]}")

    # L2-normalize
    X_normed = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    # PCA
    print("PCA(100)...")
    pca = PCA(n_components=100, random_state=42)
    X_pca = pca.fit_transform(X_normed)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    # UMAP
    print("UMAP...")
    reducer = UMAP(n_components=2, n_neighbors=40, min_dist=0.03,
                   spread=5.0, metric="correlation", random_state=42)
    coords = reducer.fit_transform(X_pca)

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sfnp.save_file({
        "coords": coords.astype(np.float32),
        "pathogenic": joint_df["pathogenic"].to_numpy().astype(np.int32),
    }, str(out_tensors))
    joint_df.select("csq", "variant_type").write_ipc(out_meta)
    print(f"\nSaved to {OUT_DIR}/umap_combined.*")
    print(f"  {coords.shape[0]:,} points")


if __name__ == "__main__":
    main()
