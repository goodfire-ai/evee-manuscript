#!/usr/bin/env python3
"""
UMAP grid search — run one parameter configuration.

Loads cached PCA embeddings (or computes + caches them on first run),
runs UMAP with specified parameters, generates a 2-panel test figure.

Usage:
    python scripts/prepare/umap_grid_search.py \
        --n-neighbors 40 --min-dist 0.03 --spread 1 --metric correlation

    # Or: generate PCA cache only (run this first before submitting jobs)
    python scripts/prepare/umap_grid_search.py --cache-only
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import polars as pl
import safetensors.numpy as sfnp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ARTIFACTS = ROOT / "artifacts"
MAYO = ROOT / "data" / "clinvar" / "evo2-7b"
CACHE_DIR = ROOT / "figures" / "test" / "umap_grid"
PCA_CACHE = CACHE_DIR / "_pca_cache.npz"

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

CONSEQ_ORDER = ("Missense", "Synonymous", "Nonsense", "Splice", "Intronic",
                "Frameshift", "In-frame", "UTR", "Other")


def _bf16_to_f32(raw: bytes, shape: tuple) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32).reshape(shape)


def _load_embeddings_direct(embed_dir: Path, variant_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    db_path = embed_dir / "index.sqlite"
    chunks_dir = embed_dir / "chunks"

    conn = sqlite3.connect(str(db_path))
    placeholders = ",".join("?" * len(variant_ids))
    rows = conn.execute(
        f"SELECT sequence_id, chunk_id, offset FROM sequence_locations "
        f"WHERE sequence_id IN ({placeholders})",
        variant_ids,
    ).fetchall()
    conn.close()

    by_chunk = {}
    for seq_id, chunk_id, offset in rows:
        by_chunk.setdefault(chunk_id, []).append((seq_id, offset))

    embeddings, found_ids = [], []
    for chunk_id in sorted(by_chunk):
        chunk_path = chunks_dir / f"chunk_{chunk_id:06d}" / "activations.safetensors"
        with open(chunk_path, "rb") as fh:
            header_size = int.from_bytes(fh.read(8), "little")
            header = json.loads(fh.read(header_size))
            data_start = 8 + header_size
            act_info = header["activations"]
            act_offset = act_info["data_offsets"]
            act_shape = act_info["shape"]
            item_elems = act_shape[1] * act_shape[2]
            item_bytes = item_elems * 2

            for seq_id, offset in by_chunk[chunk_id]:
                fh.seek(data_start + act_offset[0] + offset * item_bytes)
                raw = fh.read(item_bytes)
                if len(raw) < item_bytes:
                    continue
                embeddings.append(_bf16_to_f32(raw, (item_elems,)))
                found_ids.append(seq_id)

    return np.stack(embeddings), found_ids


def build_pca_cache():
    """Load embeddings, subsample, PCA, and cache to disk."""
    from sklearn.decomposition import PCA

    rng = np.random.RandomState(42)

    # SNVs
    labeled_meta = pl.read_ipc(ARTIFACTS / "metadata_deconfounded.feather")
    benign_ids = labeled_meta.filter(pl.col("label") == "benign")["variant_id"].to_list()
    path_ids = labeled_meta.filter(pl.col("label") == "pathogenic")["variant_id"].to_list()
    rng.shuffle(benign_ids); rng.shuffle(path_ids)
    snv_sample = benign_ids[:10_000] + path_ids[:10_000]
    print(f"Loading SNV embeddings ({len(snv_sample)})...")
    X_snv, ids_snv = _load_embeddings_direct(MAYO / "labeled" / "covariance64_pool", snv_sample)
    snv_df = pl.DataFrame({"variant_id": ids_snv}).join(
        labeled_meta.select("variant_id", "label", "consequence"), on="variant_id", how="left",
    ).with_columns(
        (pl.col("label") == "pathogenic").cast(pl.Int32).alias("pathogenic"),
        pl.lit("SNV").alias("variant_type"),
    )

    # Indels
    indel_meta = pl.read_ipc(ARTIFACTS / "metadata_labeled_indels.feather")
    indel_b = indel_meta.filter(pl.col("label") == "benign")["variant_id"].to_list()
    indel_p = indel_meta.filter(pl.col("label") == "pathogenic")["variant_id"].to_list()
    rng.shuffle(indel_b); rng.shuffle(indel_p)
    indel_sample = indel_b[:1_000] + indel_p[:1_000]
    print(f"Loading indel embeddings ({len(indel_sample)})...")
    X_indel, ids_indel = _load_embeddings_direct(MAYO / "indels" / "covariance64_pool", indel_sample)
    indel_df = pl.DataFrame({"variant_id": ids_indel}).join(
        indel_meta.select("variant_id", "label", "consequence"), on="variant_id", how="left",
    ).with_columns(
        (pl.col("label") == "pathogenic").cast(pl.Int32).alias("pathogenic"),
        pl.lit("Indel").alias("variant_type"),
    )

    # VUS
    vus_meta = pl.read_ipc(ARTIFACTS / "metadata_vus.feather")
    vus_all = vus_meta["variant_id"].to_list()
    rng.shuffle(vus_all)
    vus_sample = vus_all[:5_000]
    print(f"Loading VUS embeddings ({len(vus_sample)})...")
    X_vus, ids_vus = _load_embeddings_direct(MAYO / "vus" / "covariance64_pool", vus_sample)
    vus_df = pl.DataFrame({"variant_id": ids_vus}).join(
        vus_meta.select("variant_id", "consequence"), on="variant_id", how="left",
    ).with_columns(
        pl.lit(-1).cast(pl.Int32).alias("pathogenic"),
        pl.lit("VUS").alias("variant_type"),
    )

    # Combine
    joint_df = pl.concat([
        snv_df.select("variant_id", "pathogenic", "consequence", "variant_type"),
        indel_df.select("variant_id", "pathogenic", "consequence", "variant_type"),
        vus_df.select("variant_id", "pathogenic", "consequence", "variant_type"),
    ]).with_columns(
        pl.col("consequence").replace_strict(CONSEQ_MAP, default="Other").alias("csq"),
    )
    X = np.concatenate([X_snv, X_indel, X_vus])
    X_normed = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    print(f"PCA(100) on {X_normed.shape[0]:,} embeddings...")
    pca = PCA(n_components=100, random_state=42)
    X_pca = pca.fit_transform(X_normed)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    # Save cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PCA_CACHE,
        X_pca=X_pca,
        pathogenic=joint_df["pathogenic"].to_numpy(),
        csq=joint_df["csq"].to_numpy(),
        variant_type=joint_df["variant_type"].to_numpy(),
    )
    print(f"Cached PCA to {PCA_CACHE}")
    return X_pca, joint_df


def load_pca_cache():
    """Load cached PCA embeddings."""
    data = np.load(PCA_CACHE, allow_pickle=True)
    return data["X_pca"], data["pathogenic"], data["csq"], data["variant_type"]


def run_umap_and_plot(n_neighbors, min_dist, spread, metric):
    """Run UMAP with given params and generate test figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from umap import UMAP
    from theme.mayo_theme import apply_theme, COLORS, CONSEQ_COLORS

    apply_theme()

    tag = f"nn{n_neighbors}_md{min_dist}_sp{spread}_{metric}"
    out_png = CACHE_DIR / f"{tag}.png"

    if out_png.exists():
        print(f"Already exists: {out_png}")
        return

    # Load cached PCA
    X_pca, pathogenic, csq, variant_type = load_pca_cache()

    print(f"UMAP: nn={n_neighbors} md={min_dist} sp={spread} metric={metric}")
    reducer = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                   spread=spread, metric=metric, random_state=42)
    coords = reducer.fit_transform(X_pca)
    ux, uy = coords[:, 0], coords[:, 1]

    # --- Plot 2-panel figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    SCATTER = dict(s=3.0, marker="o", rasterized=True, edgecolors="none")
    is_snv = (variant_type == "SNV") | (variant_type == "VUS")
    is_indel = variant_type == "Indel"
    is_vus = variant_type == "VUS"

    # Panel 1: pathogenicity
    mask_b_snv = (pathogenic == 0) & is_snv
    mask_b_indel = (pathogenic == 0) & is_indel
    mask_vus = is_vus
    mask_p_snv = (pathogenic == 1) & is_snv
    mask_p_indel = (pathogenic == 1) & is_indel

    ax1.scatter(ux[mask_b_snv], uy[mask_b_snv], c=COLORS["benign"], alpha=0.10,
                label="Benign SNV", zorder=1, **SCATTER)
    ax1.scatter(ux[mask_vus], uy[mask_vus], c=COLORS["vus"], alpha=0.08, zorder=1, **SCATTER)
    ax1.scatter(ux[mask_b_indel], uy[mask_b_indel], c=COLORS["sage"], alpha=0.15,
                label="Benign indel", zorder=2, **SCATTER)
    ax1.scatter(ux[mask_p_snv], uy[mask_p_snv], c=COLORS["pathogenic"], alpha=0.35,
                label="Pathogenic SNV", zorder=3, **SCATTER)
    ax1.scatter(ux[mask_p_indel], uy[mask_p_indel], c=COLORS["crimson"], alpha=0.45,
                label="Pathogenic indel", zorder=3, **SCATTER)
    ax1.scatter([], [], c=COLORS["vus"], label="VUS", **SCATTER)

    leg1 = ax1.legend(fontsize=6, markerscale=3, loc="upper center",
                      bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)
    for h in leg1.legend_handles:
        h.set_alpha(1.0)
    ax1.set_title("Pathogenicity", fontsize=9, fontweight="semibold")

    # Panel 2: consequence
    COMBINED_CONSEQ_COLORS = {
        **CONSEQ_COLORS,
        "Noncoding": COLORS["light_gray"],
        "In-frame": COLORS["lavender"],
        "Frameshift": COLORS["crimson"],
    }
    for csq_name in CONSEQ_ORDER:
        color = COMBINED_CONSEQ_COLORS.get(csq_name, COLORS["light_gray"])
        mask = csq == csq_name
        if mask.sum() > 0:
            ax2.scatter(ux[mask], uy[mask], c=color, alpha=0.15, label=csq_name, **SCATTER)

    leg2 = ax2.legend(fontsize=6, markerscale=3, loc="upper center",
                      bbox_to_anchor=(0.5, -0.02), ncol=5, frameon=False)
    for h in leg2.legend_handles:
        h.set_alpha(1.0)
    ax2.set_title("Consequence", fontsize=9, fontweight="semibold")

    for ax in (ax1, ax2):
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(f"nn={n_neighbors}  min_dist={min_dist}  spread={spread}  metric={metric}",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-neighbors", type=int, default=40)
    parser.add_argument("--min-dist", type=float, default=0.03)
    parser.add_argument("--spread", type=float, default=1.0)
    parser.add_argument("--metric", type=str, default="correlation")
    parser.add_argument("--cache-only", action="store_true",
                        help="Only build PCA cache, don't run UMAP")
    args = parser.parse_args()

    if args.cache_only or not PCA_CACHE.exists():
        build_pca_cache()
        if args.cache_only:
            return

    run_umap_and_plot(args.n_neighbors, args.min_dist, args.spread, args.metric)


if __name__ == "__main__":
    main()
