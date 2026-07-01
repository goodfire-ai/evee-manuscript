"""CLI for generating manuscript figures.

Two commands:
    evee-ms figures  — generate all figures from cached artifacts
    evee-ms prepare  — run data preparation (requires goodfire-core + GPU)
"""
import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(help="Evee manuscript figure generation.")

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
ARTIFACTS = ROOT / "artifacts"

# Figure scripts in generation order
FIGURE_SCRIPTS = (
    # Figure 1 — Pathogenicity classification
    "figure1/fig1b_snv_heatmap.py",
    "figure1/fig1c_deconf_heatmap.py",
    "figure1/fig1d_indel_heatmap.py",
    # Figure 2 — Representations & transfer
    "figure2/fig2a_conservation_lineplot.py",
    "figure2/fig2b_umap_pathogenicity.py",
    "figure2/fig2c_umap_consequence.py",
    "figure2/fig2d_dms_spearman.py",
    # Figure 3 — Mechanism decoding
    "figure3/fig3b_probe_auroc_boxplot.py",
    "figure3/fig3c_mechanism_recovery.py",
    "figure3/fig3d_interp_quality.py",
    # Figure 4 — RNA-seq validation
    "figure4/fig4ab_brca1_rna.py",
    "figure4/fig4c_splice_nmd_headline.py",
    # Figure 5 — Clinical & population validation
    "figure5/fig5a_ldlr_roc.py",
    "figure5/fig5b_ldlr_violin.py",
    "figure5/fig5c_finngen_qq.py",
    "figure5/fig5d_acmg_stacked.py",
    # Supplements
    "supplement/supfig_layer_sweep.py",
    "supplement/supfig_context_window.py",
    "supplement/supfig_topk_vs_window.py",
    "supplement/supfig_dataset_characterization.py",
    "supplement/supfig_autointerp_ablation.py",
    "supplement/supfig_splice_vaf_validation.py",
    "supplement/supfig_ra_cohort.py",
)

# Prepare scripts (require goodfire-core, torch, raw data)
PREPARE_SCRIPTS = (
    "prepare/context_ablation_data.py",
    "prepare/umap_combined.py",
    "prepare/prepare_finngen_r12_qq_pathogenicity.py",
    "prepare/finngen_acmg_artifact.py",
)


def _run_script(script: Path) -> bool:
    """Run a single script, return True on success."""
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        typer.echo(f"  FAIL  {script.relative_to(SCRIPTS)}", err=True)
        if result.stderr:
            last_line = result.stderr.strip().split("\n")[-1]
            typer.echo(f"        {last_line}", err=True)
        return False

    for line in result.stdout.strip().split("\n"):
        if line.strip():
            typer.echo(f"  {line.strip()}")
    return True


@app.command()
def figures():
    """Generate all manuscript figures from cached artifacts."""
    typer.echo(f"Artifacts: {ARTIFACTS}")
    feathers = list(ARTIFACTS.glob("*.feather"))
    typer.echo(f"Found {len(feathers)} artifact files\n")

    succeeded, failed = 0, 0
    for script_rel in FIGURE_SCRIPTS:
        script = SCRIPTS / script_rel
        typer.echo(f"Running {script_rel}...")
        if _run_script(script):
            succeeded += 1
        else:
            failed += 1

    typer.echo(f"\nDone: {succeeded} succeeded, {failed} failed")
    if failed:
        raise typer.Exit(1)


@app.command()
def prepare():
    """Run data preparation scripts (requires goodfire-core + raw data)."""
    succeeded, failed = 0, 0
    for script_rel in PREPARE_SCRIPTS:
        script = SCRIPTS / script_rel
        typer.echo(f"Running {script_rel}...")
        if _run_script(script):
            succeeded += 1
        else:
            failed += 1

    typer.echo(f"\nDone: {succeeded} succeeded, {failed} failed")
    if failed:
        raise typer.Exit(1)
