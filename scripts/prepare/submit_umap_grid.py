#!/usr/bin/env python3
"""
Submit UMAP grid search as SLURM jobs.

1. Builds PCA cache (synchronous, ~60s)
2. Submits 72 SLURM jobs for all parameter combinations

Usage:
    python scripts/prepare/submit_umap_grid.py
"""
import itertools
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GRID_SCRIPT = ROOT / "scripts" / "prepare" / "umap_grid_search.py"
LOG_DIR = ROOT / "figures" / "test" / "umap_grid" / "logs"
PYTHON = ROOT / ".venv" / "bin" / "python"

N_NEIGHBORS = [40, 80, 100]
MIN_DISTS = [0.03, 0.1, 0.2, 0.3]
SPREADS = [1, 3, 5]
METRICS = ["correlation", "cosine"]


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Build PCA cache synchronously
    print("Step 1: Building PCA cache...")
    result = subprocess.run(
        [str(PYTHON), str(GRID_SCRIPT), "--cache-only"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"FAILED to build PCA cache:\n{result.stderr}")
        sys.exit(1)
    print("PCA cache ready.\n")

    # Step 2: Submit SLURM jobs
    combos = list(itertools.product(N_NEIGHBORS, MIN_DISTS, SPREADS, METRICS))
    print(f"Step 2: Submitting {len(combos)} SLURM jobs...")

    job_ids = []
    for nn, md, sp, metric in combos:
        tag = f"nn{nn}_md{md}_sp{sp}_{metric}"
        cmd = (
            f"{PYTHON} {GRID_SCRIPT} "
            f"--n-neighbors {nn} --min-dist {md} --spread {sp} --metric {metric}"
        )
        sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=umap_{tag}
#SBATCH --output={LOG_DIR}/{tag}.out
#SBATCH --error={LOG_DIR}/{tag}.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

{cmd}
"""
        result = subprocess.run(
            ["sbatch"],
            input=sbatch_script,
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            job_ids.append(job_id)
        else:
            print(f"  FAILED: {tag} — {result.stderr.strip()}")

    print(f"\nSubmitted {len(job_ids)} / {len(combos)} jobs")
    print(f"Monitor with: squeue -u $USER")
    print(f"Logs in: {LOG_DIR}")
    print(f"\nAfter all jobs complete, run:")
    print(f"  {PYTHON} scripts/prepare/assemble_umap_grid.py")


if __name__ == "__main__":
    main()
