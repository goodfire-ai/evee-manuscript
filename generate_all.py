#!/usr/bin/env python3
"""
Regenerate all manuscript figures from cached data.

Does NOT run prepare scripts (those need goodfire-core).
Validates all data files before plotting.

Usage:
    uv run python generate_all.py
"""
import subprocess
import sys
from pathlib import Path

import polars as pl
import safetensors.torch

ROOT = Path(__file__).resolve().parent
PANELS = ROOT / "data" / "panels"
EMBEDDINGS = ROOT / "data" / "embeddings"

# --- Data specifications: (file, expected columns or keys, min_rows) ---
PANEL_SPECS = {
    # Figure 1 data
    "fig1a.csv":         (("method", "consequence", "auroc", "n_total", "n_pathogenic"), 50),
    "fig1b.csv":         (("method", "consequence", "auroc", "n_total"), 50),
    "fig1c.csv":         (("method", "tier", "auroc", "auroc_lo", "auroc_hi"), 20),
    "fig1d.csv":         (("method", "tier", "auroc", "auroc_lo", "auroc_hi"), 15),
    "fig1f.csv":         (("gene", "method", "eval_set", "auroc", "spearman"), 30),
    "supfig4.csv":       (("stratum", "n", "pct_pathogenic"), 5),
    # Supplement data
    "fig2a.csv":         (("dataset", "sae_name", "k", "auc_roc"), 10),
    # Figure 2 panels
    "fig2b.csv":         (("head", "broad_group", "auroc"), 50),
    "fig2c.csv":         (("head", "umap1", "umap2", "cluster_id", "cluster_name"), 100),
    "fig2e.csv":         (("model", "config", "composite_mean", "composite_sem"), 10),
    "fig2_disrupt_auroc.csv": (("head", "browser_group", "auroc_best", "best_direction"), 50),
    "supfig2_naive.csv": (("method", "consequence", "auroc"), 50),
    "supfig2_deconf.csv":(("method", "consequence", "auroc"), 50),
    "supfig3.csv":       (("dataset", "consequence", "count", "pathogenic_rate"), 10),
}

PANEL_SAFETENSORS = {
    "supfig1.safetensors": ("pathogenic_deltas", "benign_deltas"),
}

EMBEDDING_SPECS = {
    # (tensors_file, meta_file, expected tensor keys, expected meta columns, prepare_script)
    "umap_snv": ("umap_snv.safetensors", "umap_snv_meta.feather",
                 ("coords", "pathogenic"), ("consequence",),
                 "scripts/prepare/umap_snv.py"),
    "umap_combined": ("umap_combined.safetensors", "umap_combined_meta.feather",
                      ("coords", "pathogenic"), ("csq", "variant_type"),
                      "scripts/prepare/umap_combined.py"),
}

OPTIONAL_EMBEDDING_SPECS = {
    "umap_indel": ("umap_indel.safetensors", "umap_indel_meta.feather",
                   ("coords", "pathogenic"), ("csq_type",),
                   "scripts/prepare/umap_indel.py"),
}

# Scripts to run in order
FIGURE_SCRIPTS = (
    # Main figures
    "scripts/figure1/assemble_figure1.py",
    "scripts/figure2/assemble_figure2.py",
    # Supplements (unchanged)
    "scripts/supplement/supfig1.py",
    "scripts/supplement/supfig2.py",
    "scripts/supplement/supfig3.py",
    "scripts/supplement/supfig6.py",
    # Supplements (demoted from main figures)
    "scripts/supplement/supfig_deconf_heatmap.py",
    "scripts/supplement/supfig_missense_conservation.py",
    "scripts/supplement/supfig_ksparse.py",
    "scripts/supplement/supfig_disrupt_auroc.py",
    "scripts/supplement/supfig_indel_full.py",
)
OPTIONAL_SCRIPTS = (
    ("scripts/supplement/supfig5.py", "umap_indel"),
    ("scripts/supplement/supfig_snv_umap.py", "umap_snv"),
)


def validate_csv(path: Path, expected_cols: tuple, min_rows: int) -> list[str]:
    """Validate a CSV panel file. Returns list of issues (empty = OK)."""
    issues = []
    if not path.exists():
        return [f"missing: {path.name}"]

    df = pl.read_csv(path)
    if len(df) < min_rows:
        issues.append(f"{path.name}: only {len(df)} rows (expected >= {min_rows})")

    for col in expected_cols:
        if col not in df.columns:
            issues.append(f"{path.name}: missing column '{col}'")

    # Check for fully empty numeric columns
    for col in expected_cols:
        if col in df.columns and df[col].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
            if df[col].is_null().all():
                issues.append(f"{path.name}: column '{col}' is entirely null")

    return issues


def validate_panel_safetensors(path: Path, expected_keys: tuple) -> list[str]:
    """Validate a safetensors panel file."""
    if not path.exists():
        return [f"missing: {path.name}"]
    tensors = safetensors.torch.load_file(str(path))
    issues = []
    for key in expected_keys:
        if key not in tensors:
            issues.append(f"{path.name}: missing tensor '{key}'")
        elif tensors[key].numel() == 0:
            issues.append(f"{path.name}: tensor '{key}' is empty")
    return issues


def validate_embedding(name: str, tensors_file: str, meta_file: str,
                       tensor_keys: tuple, meta_cols: tuple,
                       prepare_script: str) -> list[str]:
    """Validate a safetensors + feather embedding pair."""
    issues = []
    t_path = EMBEDDINGS / tensors_file
    m_path = EMBEDDINGS / meta_file

    if not t_path.exists():
        return [f"missing: {tensors_file} (run {prepare_script})"]
    if not m_path.exists():
        return [f"missing: {meta_file} (run {prepare_script})"]

    tensors = safetensors.torch.load_file(str(t_path))
    for key in tensor_keys:
        if key not in tensors:
            issues.append(f"{tensors_file}: missing tensor '{key}'")
        elif tensors[key].numel() == 0:
            issues.append(f"{tensors_file}: tensor '{key}' is empty")

    meta = pl.read_ipc(m_path)
    for col in meta_cols:
        if col not in meta.columns:
            issues.append(f"{meta_file}: missing column '{col}'")

    # Cross-check: coords rows should match meta rows
    if "coords" in tensors and len(meta) != tensors["coords"].shape[0]:
        issues.append(f"{name}: coords ({tensors['coords'].shape[0]}) != meta ({len(meta)}) row mismatch")

    return issues


def check_data() -> bool:
    """Validate all data. Returns True if all required data is OK."""
    all_ok = True

    print("Validating panel data...")
    for filename, (cols, min_rows) in PANEL_SPECS.items():
        issues = validate_csv(PANELS / filename, cols, min_rows)
        if issues:
            for issue in issues:
                print(f"  ERROR: {issue}")
            all_ok = False
        else:
            df = pl.read_csv(PANELS / filename)
            print(f"  {filename}: {len(df)} rows, {len(df.columns)} cols")

    # Panel safetensors
    for filename, keys in PANEL_SAFETENSORS.items():
        issues = validate_panel_safetensors(PANELS / filename, keys)
        if issues:
            for issue in issues:
                print(f"  ERROR: {issue}")
            all_ok = False
        else:
            tensors = safetensors.torch.load_file(str(PANELS / filename))
            shapes = ", ".join(f"{k}={list(v.shape)}" for k, v in tensors.items())
            print(f"  {filename}: {shapes}")

    print("\nValidating embeddings...")
    for name, (t, m, tk, mc, script) in EMBEDDING_SPECS.items():
        issues = validate_embedding(name, t, m, tk, mc, script)
        if issues:
            for issue in issues:
                print(f"  ERROR: {issue}")
            all_ok = False
        else:
            tensors = safetensors.torch.load_file(str(EMBEDDINGS / t))
            n = tensors["coords"].shape[0]
            print(f"  {name}: {n:,} points")

    for name, (t, m, tk, mc, script) in OPTIONAL_EMBEDDING_SPECS.items():
        if not (EMBEDDINGS / t).exists():
            print(f"  {name}: not prepared (run {script})")
            continue
        issues = validate_embedding(name, t, m, tk, mc, script)
        if issues:
            for issue in issues:
                print(f"  WARNING: {issue}")
        else:
            tensors = safetensors.torch.load_file(str(EMBEDDINGS / t))
            n = tensors["coords"].shape[0]
            print(f"  {name}: {n:,} points")

    return all_ok


def run_script(script_path: str) -> bool:
    """Run a figure script and report result."""
    path = ROOT / script_path
    if not path.exists():
        print(f"  SKIP {script_path} (not found)")
        return False

    result = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  FAIL {script_path}")
        stderr = result.stderr[-500:] if result.stderr else ""
        print(f"    {stderr.strip().split(chr(10))[-1]}")
        return False

    for line in result.stdout.strip().split("\n"):
        if "Saved:" in line:
            print(f"  {line.strip()}")
    return True


def main():
    ok = check_data()
    if not ok:
        print("\nData validation failed. Fix errors above before generating figures.")
        sys.exit(1)

    print("\nGenerating figures...")
    n_ok, n_fail = 0, 0

    for script in FIGURE_SCRIPTS:
        if run_script(script):
            n_ok += 1
        else:
            n_fail += 1

    for script, embed_name in OPTIONAL_SCRIPTS:
        specs = {**EMBEDDING_SPECS, **OPTIONAL_EMBEDDING_SPECS}
        t_file = specs[embed_name][0]
        if (EMBEDDINGS / t_file).exists():
            if run_script(script):
                n_ok += 1
            else:
                n_fail += 1
        else:
            print(f"  SKIP {script} ({embed_name} not prepared)")

    print(f"\nDone: {n_ok} succeeded, {n_fail} failed")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
