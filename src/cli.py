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
    # Figure 1 panels
    "figure1/fig1b_snv_heatmap.py",
    "figure1/fig1c_indel_heatmap.py",
    "figure1/fig1d_conservation_lineplot.py",
    "figure1/fig1e_umap_pathogenicity.py",
    "figure1/fig1f_umap_consequence.py",
    "figure1/fig1g_dms_spearman.py",
    # Figure 2 panels
    "figure2/fig2b_probe_auroc_boxplot.py",
    "figure2/fig2ce_autointerp_barchart.py",
    # Supplements (S1–S5, S9)
    "supplement/supfig1_layer_sweep.py",
    "supplement/supfig2_context_window.py",
    "supplement/supfig3_topk_vs_window.py",
    "supplement/supfig4_deconf_heatmap.py",
    "supplement/supfig5_dataset_characterization.py",
    "supplement/supfig9_autointerp_ablation.py",
)

# Prepare scripts (require goodfire-core, torch, raw data)
PREPARE_SCRIPTS = (
    "prepare/context_ablation_data.py",
    "prepare/umap_combined.py",
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
