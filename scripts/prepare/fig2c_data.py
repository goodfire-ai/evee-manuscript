#!/usr/bin/env python3
"""
Prepare Figure 2C panel data — UMAP of annotation disruption space.

Loads the v4 probe scores.feather, computes disruption (var - ref) for each
annotation head, subsamples 30k variants, runs UMAP, clusters with KMeans,
and exports coordinates + labels.

Input:  website_probes/v4/token/scores.feather
Output: data/panels/fig2c.csv

Requires: pyarrow, umap-learn, scikit-learn, pandas (~2 min)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as pf
from sklearn.cluster import KMeans
from umap import UMAP

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "fig2c.feather"

DATA = ROOT / "data"
PROBE_DIR = DATA / "probes" / "token"       # symlink to probe checkpoint dir
SCORES_FEATHER = PROBE_DIR / "scores.feather"

# Head prefix → semantic type for cluster labeling
_EPIGENOMICS_PREFIXES = {"chipseq", "chromhmm", "atacseq", "fstack", "ccre"}


def _head_type(head: str) -> str:
    """Classify a head into a semantic type for cluster assignment."""
    if head.startswith("pfam_"):
        return "pfam"
    if head.startswith("interpro_"):
        return "interpro"
    for prefix in _EPIGENOMICS_PREFIXES:
        if head.startswith(prefix + "_") or head.startswith(prefix):
            return "epigenomics"
    if head.startswith("amino_acid_") or head.startswith("secondary_structure_"):
        return "protein"
    if head.startswith("in_") or head.startswith("is_"):
        return "protein"
    if head.startswith("elm_") or head.startswith("ptm_"):
        return "motif"
    if head.startswith("dna_shape_"):
        return "protein"
    for prefix in ("gnomad_", "cadd_", "revel_", "clinpred_", "bayesdel_",
                    "alphamissense_", "sift_", "polyphen_", "eve_", "gerp_",
                    "phastcons_", "phylop_", "loeuf_", "remm_", "regulomedb_"):
        if head.startswith(prefix):
            return "clinical"
    if head.endswith("_c") and not head.startswith("interpro_"):
        return "clinical"
    if head.startswith("spliceai_") or head.startswith("gtex_") or head.startswith("region_"):
        return "other"
    return "other"


def _assign_cluster_names(labels: np.ndarray, head_names: list[str],
                          n_clusters: int) -> tuple[np.ndarray, dict]:
    """Assign semantic cluster names based on dominant head types."""
    from collections import Counter

    cluster_types = {}
    for cl in range(n_clusters):
        idx = np.where(labels == cl)[0]
        cluster_types[cl] = Counter(_head_type(head_names[i]) for i in idx)

    all_types = Counter()
    for ct in cluster_types.values():
        all_types.update(ct)

    type_display = [
        ("pfam", "Pfam Domains"),
        ("clinical", "Clinical / Conservation"),
        ("epigenomics", "Epigenomics"),
        ("interpro", "InterPro"),
        ("protein", "Protein / Structure"),
        ("motif", "Motifs / PTMs"),
        ("other", "Other"),
    ]
    available_names = [(t, name) for t, name in type_display if all_types.get(t, 0) > 0]

    assigned = {}
    used_names = set()

    def _purity(cl):
        total = sum(cluster_types[cl].values())
        return cluster_types[cl].most_common(1)[0][1] / total if total else 0

    for cl in sorted(range(n_clusters), key=_purity, reverse=True):
        types = cluster_types[cl]
        total = sum(types.values())
        best_name, best_frac = None, -1
        for t, name in available_names:
            if name in used_names:
                continue
            frac = types.get(t, 0) / total if total else 0
            if frac > best_frac:
                best_frac = frac
                best_name = name
        assigned[cl] = best_name or f"Cluster {cl}"
        used_names.add(assigned[cl])

    unique_names = sorted(set(assigned.values()))
    name_to_id = {name: i for i, name in enumerate(unique_names)}
    cluster_ids = np.array([name_to_id[assigned[l]] for l in labels])
    cluster_info = {i: name for name, i in name_to_id.items()}
    return cluster_ids, cluster_info


def main():
    # 1. Load var_ and ref_ columns
    print("Loading schema...")
    schema = pf.read_table(SCORES_FEATHER, columns=[]).schema
    all_cols = [f.name for f in schema]
    var_cols = sorted([c for c in all_cols if c.startswith("var_")])
    ref_cols = sorted([c for c in all_cols if c.startswith("ref_")])
    print(f"  {len(var_cols)} var_* + {len(ref_cols)} ref_* columns")

    # Verify matched pairs
    var_heads = [c[4:] for c in var_cols]
    ref_heads = [c[4:] for c in ref_cols]
    assert var_heads == ref_heads, "var_/ref_ column mismatch"

    print("Loading scores...")
    table = pf.read_table(SCORES_FEATHER, columns=var_cols + ref_cols)
    scores = table.to_pandas()
    del table
    print(f"  Shape: {scores.shape}")

    # 2. Compute disruption (var - ref) for each head
    print("Computing disruption scores...")
    disruption = np.column_stack([
        scores[f"var_{h}"].values - scores[f"ref_{h}"].values
        for h in var_heads
    ]).T  # (n_heads, n_variants)
    del scores
    disruption = np.nan_to_num(disruption, nan=0.0).astype(np.float32)
    print(f"  Disruption matrix: {disruption.shape}")

    # 3. Subsample 30k variants
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(disruption.shape[1], size=30000, replace=False)
    X_ann = disruption[:, sample_idx]
    del disruption
    print(f"  Subsampled: {X_ann.shape}")

    # 4. Filter low-variance heads
    stds = X_ann.std(axis=1)
    high_var = stds > 0.001
    X_ann_hv = X_ann[high_var]
    head_names_hv = [h for h, m in zip(var_heads, high_var) if m]
    print(f"  High-variance (std>0.001): {X_ann_hv.shape[0]} annotations")

    # 5. UMAP — tuned for disruption space
    print("Running UMAP...")
    reducer = UMAP(n_neighbors=20, min_dist=0.15, metric="correlation", random_state=42)
    emb = reducer.fit_transform(X_ann_hv)
    print(f"  UMAP done: {emb.shape}")

    # 6. KMeans + content-aware naming
    from collections import Counter
    type_counts = Counter(_head_type(h) for h in head_names_hv)
    k = min(max(len(type_counts), 3), 5)
    print(f"  Types: {dict(type_counts)}, k={k}")

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    raw_labels = km.fit_predict(emb)
    cluster_ids, cluster_info = _assign_cluster_names(raw_labels, head_names_hv, k)

    # 7. Write CSV
    import polars as pl
    df = pl.DataFrame({
        "head": head_names_hv,
        "umap1": emb[:, 0].tolist(),
        "umap2": emb[:, 1].tolist(),
        "cluster_id": cluster_ids.tolist(),
        "cluster_name": [cluster_info[c] for c in cluster_ids],
    })
    df.write_ipc(OUT)
    print(f"\nSaved {len(df)} rows to {OUT}")
    for cid in sorted(cluster_info):
        print(f"  {cluster_info[cid]}: n={(cluster_ids == cid).sum()}")


if __name__ == "__main__":
    main()
