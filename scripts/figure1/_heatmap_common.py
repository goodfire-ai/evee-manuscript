"""Shared heatmap utilities for ClinVar consequence heatmaps (SNV/deconf, supplements)."""
import numpy as np
import polars as pl

# ── Consequence order (merged) ──────────────────────────────────
CONSEQ_ORDER = (
    "Overall", "Missense", "Synonymous", "Nonsense",
    "Splice", "UTR", "Intronic", "Other",
)

MERGE_GROUPS = {
    "Splice": ("Splice Donor", "Splice Acceptor"),
    "UTR": ("5' UTR", "3' UTR"),
}

DROP_CONSEQS = {
    "Missense (canonical)", "Missense (non-canonical)",
    "Splice Donor", "Splice Acceptor", "5' UTR", "3' UTR",
    "Initiator Codon", "Non-coding Transcript",
}
DROP_METHODS = {"REVEL", "PolyPhen2", "NTv3 L2 dist", "NTv3 subref probe", "CADD v1.6"}

# Method groups — each tuple is a visual category separated by whitespace
METHOD_GROUPS = [
    ("Evo2 Covariance", "Evo2 Mean", "Evo2 Loss"),   # Evo2 family
    ("CADD v1.7",),                                    # CADD
    ("AlphaMissense", "EVE"),                           # Protein-level methods
    ("GPN-MSA", "NTv3", "AlphaGenome"),                # Other genomic LMs
]

# Flat prefix for sorting within groups
METHOD_PREFIX = ("Evo2 Covariance", "Evo2 Mean", "Evo2 Loss")

RENAME_MAP = {
    "Covpool probe": "Evo2 Covariance",
    "Evo2 cov probe": "Evo2 Covariance",
    "Evo2 probe+": "Evo2 Covariance",
    "Evo2 probe": "Evo2 Mean",
    "Evo2 mean probe": "Evo2 Mean",
    "Evo2 loss": "Evo2 Loss",
    "NTv3 cosine dist": "NTv3",
    "AlphaGenome composite": "AlphaGenome",
}


def _merge_categories(strat_df: pl.DataFrame) -> pl.DataFrame:
    """Add merged Splice / UTR rows via n-weighted averaging."""
    merge_rows = []
    methods = strat_df["method"].unique().to_list()
    numeric_cols = [c for c in strat_df.columns if c not in ("method", "type", "consequence", "n_total", "n_pathogenic", "n_valid")]
    for merged_name, sub_conseqs in MERGE_GROUPS.items():
        for method in methods:
            sub = strat_df.filter(
                (pl.col("method") == method)
                & pl.col("consequence").is_in(list(sub_conseqs))
            )
            if len(sub) < len(sub_conseqs):
                continue
            total_n = sub["n_total"].sum()
            w = sub["n_total"].to_numpy() / total_n
            row = {
                "method": method,
                "type": sub[0, "type"],
                "consequence": merged_name,
                "n_total": int(total_n),
                "n_pathogenic": int(sub["n_pathogenic"].sum()),
                "n_valid": int(sub["n_valid"].sum()),
            }
            for col in numeric_cols:
                vals = sub[col].to_numpy()
                if np.all(np.isnan(vals)):
                    row[col] = None
                else:
                    row[col] = float((np.nan_to_num(vals) * w).sum())
            merge_rows.append(row)
    if merge_rows:
        strat_df = pl.concat([strat_df, pl.DataFrame(merge_rows)], how="diagonal_relaxed")
    return strat_df


def prepare(strat_df: pl.DataFrame) -> pl.DataFrame:
    """Merge categories, filter, and rename methods."""
    strat_df = _merge_categories(strat_df)
    df = strat_df.filter(
        ~pl.col("consequence").is_in(list(DROP_CONSEQS))
        & ~pl.col("method").is_in(list(DROP_METHODS))
    )
    df = df.with_columns(pl.col("method").replace_strict(RENAME_MAP, default=pl.first()))

    # AlphaMissense: blank non-missense/non-overall, set Overall = Missense
    am_non_miss = (pl.col("method") == "AlphaMissense") & ~pl.col("consequence").is_in(["Overall", "Missense"])
    blank_cols = [c for c in ("auroc", "auprc", "auroc_lo", "auroc_hi") if c in df.columns]
    df = df.with_columns(
        [pl.when(am_non_miss).then(None).otherwise(pl.col(c)).alias(c) for c in blank_cols]
    )

    # Copy AlphaMissense Missense values to Overall
    am_miss = df.filter((pl.col("method") == "AlphaMissense") & (pl.col("consequence") == "Missense"))
    if len(am_miss) == 1:
        am_ov_mask = (pl.col("method") == "AlphaMissense") & (pl.col("consequence") == "Overall")
        copy_cols = [c for c in ("auroc", "auroc_lo", "auroc_hi", "auprc") if c in df.columns]
        df = df.with_columns(
            [pl.when(am_ov_mask).then(pl.lit(am_miss[0, c])).otherwise(pl.col(c)).alias(c)
             for c in copy_cols if am_miss[0, c] is not None]
        )
    return df


def plot_heatmap(ax, strat_df: pl.DataFrame, metric: str,
                 font_size_tick=None, font_size_cell=None, aspect="equal"):
    """Draw heatmap on given axes: consequences (y) x methods (x)."""
    from theme.mayo_theme import (
        HEATMAP_CMAP,
        FONT_SIZE_TICK as _FSTK, FONT_SIZE_CELL as _FSC,
    )
    fstk = font_size_tick or _FSTK
    fsc = font_size_cell or _FSC

    conseqs = [c for c in CONSEQ_ORDER if c in strat_df["consequence"].to_list()]

    # Sort methods by group, then by Overall AUROC within each group
    overall = strat_df.filter(pl.col("consequence") == "Overall").select("method", metric)
    available = set(overall["method"].to_list())
    methods = []
    group_sep_positions = []  # x positions for vertical separators between groups
    for group in METHOD_GROUPS:
        group_methods = [m for m in group if m in available]
        if not group_methods:
            continue
        if methods:
            group_sep_positions.append(len(methods) - 0.5)
        methods.extend(group_methods)
    # Append any remaining methods not in any group
    ungrouped = [m for m in available if m not in methods]
    if ungrouped:
        if methods:
            group_sep_positions.append(len(methods) - 0.5)
        ug_sorted = overall.filter(pl.col("method").is_in(ungrouped)).sort(metric, descending=True)["method"].to_list()
        methods.extend(ug_sorted)

    # Consequence labels: bold name, regular stats
    conseq_names = []
    conseq_stats = []
    for c in conseqs:
        sub = strat_df.filter(pl.col("consequence") == c).row(0, named=True)
        n = int(sub.get("n_total", 0))
        p = int(sub.get("n_pathogenic", 0))
        conseq_names.append(c)
        conseq_stats.append(f"(n={n:,}, {100*p/n:.1f}%P)" if n else "")

    # Build matrix via pivot
    matrix = np.full((len(conseqs), len(methods)), np.nan)
    pivot = strat_df.pivot(on="method", index="consequence", values=metric)
    for i, c in enumerate(conseqs):
        row = pivot.filter(pl.col("consequence") == c)
        if len(row) == 1:
            for j, m in enumerate(methods):
                if m in pivot.columns:
                    v = row[0, m]
                    if v is not None:
                        matrix[i, j] = float(v)

    # Remove all-NaN cols
    valid = ~np.all(np.isnan(matrix), axis=0)
    matrix = matrix[:, valid]
    methods = [m for m, v in zip(methods, valid) if v]

    im = ax.imshow(matrix, aspect=aspect, cmap=HEATMAP_CMAP, vmin=0.5, vmax=1.0)

    # Vertical separators between method groups
    for sep_x in group_sep_positions:
        ax.axvline(sep_x, color="white", linewidth=2.5)

    # Horizontal separator between Overall and the rest
    ax.axhline(0.5, color="white", linewidth=2.5)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(
        methods, fontsize=fstk, fontweight="semibold",
        rotation=45, ha="right",
    )
    ax.set_yticks(range(len(conseqs)))
    ax.set_yticklabels([])  # Clear default labels
    # Manual two-part labels: bold name, regular stats
    for i, (name, stat) in enumerate(zip(conseq_names, conseq_stats)):
        ax.text(-0.02, i - 0.01, name, transform=ax.get_yaxis_transform(),
                ha="right", va="center",
                fontsize=fstk, fontweight="semibold")
        if stat:
            ax.text(-0.02, i + 0.21, stat, transform=ax.get_yaxis_transform(),
                    ha="right", va="center",
                    fontsize=fstk - 1, fontweight="normal", color="#444444")

    # Cell annotations
    for i in range(len(conseqs)):
        for j in range(len(methods)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=fsc, fontweight="semibold",
                        color="#222222")

    # Black border around heatmap
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

    return im
