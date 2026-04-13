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
    # Main figures (assembly calls individual panels internally)
    "figure1/assemble_figure1.py",
    "figure2/assemble_figure2.py",
    # Supplements
    "supplement/supfig_activation_delta.py",
    "supplement/supfig_full_heatmap.py",
    "supplement/supfig_dataset_characterization.py",
    "supplement/supfig_indel_heatmap.py",
    "supplement/supfig_dms_auroc.py",
    "supplement/supfig_context_window.py",
    "supplement/supfig_deconf_heatmap.py",
    "supplement/supfig_disrupt_auroc.py",
    "supplement/supfig_indel_full.py",
    "supplement/supfig_topk_sweep.py",
    "supplement/supfig_topk_vs_window.py",
)

# Prepare scripts (require goodfire-core, torch, raw data)
PREPARE_SCRIPTS = (
    "prepare/probe_auroc_data.py",
    "prepare/disruption_umap_data.py",
    "prepare/autointerp_eval_data.py",
    "prepare/fig2_disrupt_auroc_data.py",
    "prepare/umap_snv.py",
    "prepare/umap_combined.py",
    "prepare/umap_indel.py",
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
