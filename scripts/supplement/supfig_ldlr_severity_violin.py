#!/usr/bin/env python3
"""
Supplementary Figure — LDLR severity per-tier distribution (violin).

Per-tier violin of EVEE Pathogenicity scores across three FH severity tiers
(Clinical FH n=111, Suspected FH n=20, Presymptomatic carrier n=16) for 147
LDLR variants from the Mayo Clinic Tapestry biobank. The ROC summary (AUC=0.91)
is in the main figure (fig1g).

Passthrough from evee_resubmission_natgen/fig_ldlr_severity_violin.{png,pdf}.

Output: figures/supplement/supfig_ldlr_severity_violin.{png,pdf}
"""
import shutil
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "evee_resubmission_natgen"
OUT_DIR = ROOT / "figures" / "supplement"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STEM     = "fig_ldlr_severity_violin"
OUT_STEM = "supfig_ldlr_severity_violin"


def main():
    for ext in ("png", "pdf"):
        src = SRC_DIR / f"{STEM}.{ext}"
        dst = OUT_DIR / f"{OUT_STEM}.{ext}"
        if not src.exists():
            raise FileNotFoundError(f"Source not found: {src}")
        shutil.copy2(src, dst)
        print(f"Copied: {dst}")


if __name__ == "__main__":
    main()
