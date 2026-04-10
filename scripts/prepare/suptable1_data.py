#!/usr/bin/env python3
"""
Supplementary Table 1 — Full classification results.

AUROC, AUPRC, accuracy, and F1 (each with bootstrap 95% CIs) for all
methods × consequence types × benchmarks (SNV and Indel).

For methods with per-variant predictions available (Evo2 Covariance,
CADD v1.7, AlphaMissense), all metrics are computed from raw scores.
For other methods (Evo2 Mean, Evo2 Loss, GPN-MSA, NTv3, AlphaGenome),
AUROC/CI and AUPRC are taken from pre-computed artifacts; accuracy and
F1 are left blank (no per-variant scores accessible).

Inputs:
    artifacts/fig1a.feather          — pre-computed SNV AUROC/AUPRC
    artifacts/supfig4.feather        — pre-computed indel AUROC
    variant-viewer labeled.parquet   — per-variant annotation scores
    mayo_manuscript scores.feather   — per-variant Evo2 Cov probe scores

Output:
    artifacts/suptable1.csv
"""
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"

LABELED_PARQUET = Path(
    "/mnt/polished-lake/home/thomasd/projects/variant-viewer/data/labeled.parquet"
)
GENES_FEATHER = Path(
    "/mnt/polished-lake/artifacts/fellows-shared/life-sciences/genomics/mendelian"
    "/gencode/genes.feather"
)
EVO2_COV_SCORES = Path(
    "/mnt/polished-lake/artifacts/fellows-shared/life-sciences/genomics/mendelian"
    "/mayo-mendelian/mayo_manuscript.bak/data/clinvar/evo2-7b/labeled"
    "/covariance64_pool/scores.feather"
)

N_BOOTSTRAP = 1000
SEED = 42

# ── Consequence grouping (matches _heatmap_common.py) ──────────────

CONSEQUENCE_GROUPS: dict[str, tuple[str, ...]] = {
    "Missense": ("missense_variant",),
    "Synonymous": ("synonymous_variant",),
    "Nonsense": ("nonsense", "stop_gained"),
    "Splice Donor": ("splice_donor_variant",),
    "Splice Acceptor": ("splice_acceptor_variant",),
    "Intronic": ("intron_variant",),
    "5' UTR": ("5_prime_UTR_variant",),
    "3' UTR": ("3_prime_UTR_variant",),
    "Initiator Codon": ("initiator_codon_variant",),
    "Non-coding Transcript": ("non_coding_transcript_variant", "non-coding_transcript_variant"),
}
_CSQ_TO_GROUP = {c: g for g, cs in CONSEQUENCE_GROUPS.items() for c in cs}

MERGE_GROUPS = {
    "Splice": ("Splice Donor", "Splice Acceptor"),
    "UTR": ("5' UTR", "3' UTR"),
}

# Consequence types shown in fig1b (after merging)
CONSEQ_ORDER = (
    "Overall", "Missense", "Synonymous", "Nonsense",
    "Splice", "UTR", "Intronic", "Other",
)

# ── Method definitions ──────────────────────────────────────────────

# Methods with per-variant data: (display_name, column_in_labeled_parquet_or_None)
# If column is None, scores come from a separate file.
METHODS_WITH_SCORES: dict[str, str | None] = {
    "Evo2 Covariance": None,  # from EVO2_COV_SCORES
    "CADD v1.7": "cadd_wg_c",
    "AlphaMissense": "alphamissense_c",
}

# Map from fig1a method names to display names
FIG1A_RENAME = {
    "Covpool probe": "Evo2 Covariance",
    "Evo2 probe": "Evo2 Mean",
    "Evo2 loss": "Evo2 Loss",
    "CADD v1.7": "CADD v1.7",
    "AlphaMissense": "AlphaMissense",
    "GPN-MSA": "GPN-MSA",
    "NTv3 cosine dist": "NTv3",
    "AlphaGenome composite": "AlphaGenome",
}

# Methods to include (in display order)
METHOD_ORDER = (
    "Evo2 Covariance", "Evo2 Mean", "Evo2 Loss",
    "CADD v1.7", "AlphaMissense",
    "GPN-MSA", "NTv3", "AlphaGenome",
)

# Methods that only apply to missense variants
MISSENSE_ONLY = {"AlphaMissense"}


# ── Bootstrap helper ────────────────────────────────────────────────

def _optimal_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Find threshold maximising Youden's J (sensitivity + specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def bootstrap_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict[str, float | None]:
    """Compute AUROC, AUPRC, accuracy, F1 with bootstrap 95% CIs."""
    n = len(labels)
    n_pos = int(labels.sum())
    n_neg = n - n_pos

    if n_pos < 5 or n_neg < 5:
        return {
            "auroc": None, "auroc_lo": None, "auroc_hi": None,
            "auprc": None, "auprc_lo": None, "auprc_hi": None,
            "accuracy": None, "accuracy_lo": None, "accuracy_hi": None,
            "f1": None, "f1_lo": None, "f1_hi": None,
        }

    # Point estimates
    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)
    threshold = _optimal_threshold(labels, scores)
    preds = (scores >= threshold).astype(int)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)

    # Bootstrap
    rng = np.random.default_rng(seed)
    aurocs = np.empty(n_bootstrap)
    auprcs = np.empty(n_bootstrap)
    accs = np.empty(n_bootstrap)
    f1s = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_b, s_b = labels[idx], scores[idx]
        if len(np.unique(y_b)) < 2:
            aurocs[i] = auprcs[i] = accs[i] = f1s[i] = np.nan
            continue
        aurocs[i] = roc_auc_score(y_b, s_b)
        auprcs[i] = average_precision_score(y_b, s_b)
        p_b = (s_b >= threshold).astype(int)
        accs[i] = accuracy_score(y_b, p_b)
        f1s[i] = f1_score(y_b, p_b, zero_division=0.0)

    def ci(arr: np.ndarray) -> tuple[float, float]:
        valid = arr[~np.isnan(arr)]
        if len(valid) < 10:
            return (np.nan, np.nan)
        return (float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5)))

    auroc_lo, auroc_hi = ci(aurocs)
    auprc_lo, auprc_hi = ci(auprcs)
    acc_lo, acc_hi = ci(accs)
    f1_lo, f1_hi = ci(f1s)

    return {
        "auroc": auroc, "auroc_lo": auroc_lo, "auroc_hi": auroc_hi,
        "auprc": auprc, "auprc_lo": auprc_lo, "auprc_hi": auprc_hi,
        "accuracy": accuracy, "accuracy_lo": acc_lo, "accuracy_hi": acc_hi,
        "f1": f1, "f1_lo": f1_lo, "f1_hi": f1_hi,
    }


# ── Data loading ────────────────────────────────────────────────────

def load_snv_data() -> pl.DataFrame:
    """Load 833K labeled SNVs in genes <= 100 kb with consequence groups."""
    genes = pl.read_ipc(
        str(GENES_FEATHER), columns=["gene_name", "length"]
    )
    small_genes = set(
        genes.filter(pl.col("length") <= 100_000)["gene_name"].to_list()
    )

    score_cols = [c for c in METHODS_WITH_SCORES.values() if c is not None]
    df = pl.read_parquet(
        str(LABELED_PARQUET),
        columns=["variant_id", "ref", "alt", "gene_name", "pathogenic",
                 "consequence"] + score_cols,
    )

    # Filter: SNVs in small genes
    df = df.filter(
        (pl.col("ref").str.len_chars() == 1)
        & (pl.col("alt").str.len_chars() == 1)
        & pl.col("gene_name").is_in(list(small_genes))
    )

    # Map consequence to groups
    df = df.with_columns(
        pl.col("consequence")
        .replace_strict(_CSQ_TO_GROUP, default="Other")
        .alias("consequence_group"),
    )

    # Join Evo2 Covariance probe scores
    evo2_scores = pl.read_ipc(str(EVO2_COV_SCORES)).select("variant_id", "score")
    df = df.join(evo2_scores.rename({"score": "evo2_cov_score"}), on="variant_id", how="left")

    print(f"Loaded {df.shape[0]:,} SNVs")
    return df


def compute_from_raw(
    df: pl.DataFrame,
    method_name: str,
    score_col: str,
) -> list[dict]:
    """Compute all metrics for one method across all consequence types."""
    rows = []

    for csq in CONSEQ_ORDER:
        # Apply missense-only filter
        if method_name in MISSENSE_ONLY and csq not in ("Overall", "Missense"):
            continue

        if csq == "Overall":
            subset = df
        elif csq in MERGE_GROUPS:
            subset = df.filter(
                pl.col("consequence_group").is_in(list(MERGE_GROUPS[csq]))
            )
        else:
            subset = df.filter(pl.col("consequence_group") == csq)

        # Filter to non-null scores
        subset = subset.filter(pl.col(score_col).is_not_null())
        if subset.shape[0] < 20:
            continue

        labels = subset["pathogenic"].to_numpy().astype(int)
        scores = subset[score_col].to_numpy().astype(float)

        n_total = len(labels)
        n_pathogenic = int(labels.sum())

        metrics = bootstrap_metrics(labels, scores)
        rows.append({
            "benchmark": "ClinVar SNV",
            "method": method_name,
            "consequence": csq,
            "n": n_total,
            "n_pathogenic": n_pathogenic,
            **metrics,
        })

        print(
            f"  {method_name:20s} {csq:15s}  "
            f"AUROC={metrics['auroc']:.4f}  F1={metrics['f1']:.4f}  n={n_total:,}"
        )

    return rows


def load_fig1a_fallback() -> list[dict]:
    """Load pre-computed AUROC/AUPRC from fig1a.feather for methods
    without per-variant scores."""
    fig1a = pl.read_ipc(str(ARTIFACTS / "fig1a.feather"))

    # Filter to methods in our display set
    fig1a = fig1a.with_columns(
        pl.col("method").replace_strict(FIG1A_RENAME, default=pl.first())
    )
    # Keep only methods NOT in METHODS_WITH_SCORES
    fallback_methods = set(METHOD_ORDER) - set(METHODS_WITH_SCORES.keys())
    fig1a = fig1a.filter(pl.col("method").is_in(list(fallback_methods)))

    # Merge Splice and UTR categories
    rows = []
    for row in fig1a.iter_rows(named=True):
        csq = row["consequence"]
        # Map sub-consequences to merged groups
        if csq in ("Splice Donor", "Splice Acceptor"):
            continue  # Skip — will use merged row from heatmap_common
        if csq in ("5' UTR", "3' UTR"):
            continue

        # Skip consequences not in our display set
        if csq in ("Initiator Codon", "Non-coding Transcript",
                    "Missense (canonical)", "Missense (non-canonical)"):
            continue

        rows.append({
            "benchmark": "ClinVar SNV",
            "method": row["method"],
            "consequence": csq,
            "n": row["n_valid"],
            "n_pathogenic": row["n_pathogenic"],
            "auroc": row["auroc"],
            "auroc_lo": row["auroc_lo"],
            "auroc_hi": row["auroc_hi"],
            "auprc": row["auprc"],
            "auprc_lo": None,
            "auprc_hi": None,
            "accuracy": None,
            "accuracy_lo": None,
            "accuracy_hi": None,
            "f1": None,
            "f1_lo": None,
            "f1_hi": None,
        })

    # Add merged Splice and UTR rows via n-weighted averaging
    for merged_name, sub_names in MERGE_GROUPS.items():
        for method in fallback_methods:
            subs = fig1a.filter(
                (pl.col("method") == method)
                & pl.col("consequence").is_in(list(sub_names))
            )
            if subs.shape[0] < len(sub_names):
                continue
            total_n = int(subs["n_valid"].sum())
            if total_n == 0:
                continue
            weights = subs["n_valid"].to_numpy() / total_n
            auroc_vals = subs["auroc"].to_numpy()
            auprc_vals = subs["auprc"].to_numpy()

            auroc_merged = float((np.nan_to_num(auroc_vals) * weights).sum())
            auprc_merged = float((np.nan_to_num(auprc_vals) * weights).sum())

            # Weighted CI bounds
            lo_vals = subs["auroc_lo"].to_numpy()
            hi_vals = subs["auroc_hi"].to_numpy()
            auroc_lo_merged = float((np.nan_to_num(lo_vals) * weights).sum())
            auroc_hi_merged = float((np.nan_to_num(hi_vals) * weights).sum())

            rows.append({
                "benchmark": "ClinVar SNV",
                "method": method,
                "consequence": merged_name,
                "n": total_n,
                "n_pathogenic": int(subs["n_pathogenic"].sum()),
                "auroc": auroc_merged,
                "auroc_lo": auroc_lo_merged,
                "auroc_hi": auroc_hi_merged,
                "auprc": auprc_merged,
                "auprc_lo": None, "auprc_hi": None,
                "accuracy": None, "accuracy_lo": None, "accuracy_hi": None,
                "f1": None, "f1_lo": None, "f1_hi": None,
            })

    return rows


def load_indel_data() -> list[dict]:
    """Load indel benchmark from supfig4.feather."""
    df = pl.read_ipc(str(ARTIFACTS / "supfig4.feather"))

    # Keep strata shown in fig1c
    keep_strata = ("Overall", "Frameshift", "In-frame", "Noncoding",
                   "Splice-adj.", "Insertion", "Deletion")
    df = df.filter(pl.col("stratum").is_in(list(keep_strata)))

    method_cols = (
        ("evo2_cov_probe_zeroshot", "Evo2 Covariance"),
        ("evo2_mean_probe_supervised", "Evo2 Mean"),
        ("cadd_v17_indel", "CADD v1.7"),
        ("ntv3_subref_probe_supervised", "NTv3"),
    )

    rows = []
    for row in df.iter_rows(named=True):
        stratum = row["stratum"]
        n = row["n"]
        n_path = int(round(n * row["pct_pathogenic"] / 100))

        for col, display_name in method_cols:
            val = row[col]
            rows.append({
                "benchmark": "ClinVar Indel",
                "method": display_name,
                "consequence": stratum,
                "n": n,
                "n_pathogenic": n_path,
                "auroc": val,
                "auroc_lo": None, "auroc_hi": None,
                "auprc": None, "auprc_lo": None, "auprc_hi": None,
                "accuracy": None, "accuracy_lo": None, "accuracy_hi": None,
                "f1": None, "f1_lo": None, "f1_hi": None,
            })

    return rows


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Supplementary Table 1: Full classification results")
    print("=" * 70)

    # 1. Load SNV data
    print("\n[1/4] Loading SNV data...")
    snv_df = load_snv_data()

    # 2. Compute metrics from per-variant scores
    print("\n[2/4] Computing metrics from per-variant scores...")
    all_rows: list[dict] = []

    for method_name, col in METHODS_WITH_SCORES.items():
        score_col = col if col is not None else "evo2_cov_score"
        print(f"\n  Method: {method_name} (column: {score_col})")
        method_rows = compute_from_raw(snv_df, method_name, score_col)
        all_rows.extend(method_rows)

    # 3. Fill in methods without per-variant data from fig1a
    print("\n[3/4] Loading pre-computed metrics for remaining methods...")
    fallback_rows = load_fig1a_fallback()
    all_rows.extend(fallback_rows)
    print(f"  Added {len(fallback_rows)} rows from fig1a.feather")

    # 4. Add indel benchmark
    print("\n[4/4] Loading indel benchmark...")
    indel_rows = load_indel_data()
    all_rows.extend(indel_rows)
    print(f"  Added {len(indel_rows)} rows from supfig4.feather")

    # Build output DataFrame
    result = pl.DataFrame(all_rows)

    # Sort by benchmark, method order, consequence order
    method_rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    all_csq = list(CONSEQ_ORDER) + [
        "Frameshift", "In-frame", "Noncoding", "Splice-adj.",
        "Insertion", "Deletion",
    ]
    csq_rank = {c: i for i, c in enumerate(all_csq)}

    result = result.with_columns(
        pl.col("method").replace_strict(method_rank, default=99).alias("_method_rank"),
        pl.col("consequence").replace_strict(csq_rank, default=99).alias("_csq_rank"),
    ).sort("benchmark", "_method_rank", "_csq_rank").drop("_method_rank", "_csq_rank")

    # Round numeric columns
    numeric_cols = [
        "auroc", "auroc_lo", "auroc_hi",
        "auprc", "auprc_lo", "auprc_hi",
        "accuracy", "accuracy_lo", "accuracy_hi",
        "f1", "f1_lo", "f1_hi",
    ]
    result = result.with_columns(
        [pl.col(c).round(6) for c in numeric_cols if c in result.columns]
    )

    # Save
    out_path = ARTIFACTS / "suptable1.csv"
    result.write_csv(str(out_path))
    print(f"\nSaved: {out_path}")
    print(f"Shape: {result.shape}")
    print(f"\nPreview (first 20 rows):")
    print(result.head(20))


if __name__ == "__main__":
    main()
