#!/usr/bin/env python3
"""
Fig 2F — BRCA1 rs80358099 splice-acceptor prediction validated by RNA-seq.

Multi-track browser figure: EVEE per-position disruption tracks (top) +
gene model (middle) + RNA-seq sashimi + coverage (bottom) for JHOS-4
(carrier, VAF ≈ 1.0) and two wild-type controls.

Wider x-axis version (fig_width=20) of the supplement figure.

Input:  artifacts/brca1_disruption_profile.feather
        artifacts/brca1_e22_data.json
Output: figures/figure2/fig2f_brca1_rna_validation.{png,pdf}
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "_brca1_src",
    ROOT / "scripts" / "supplement" / "supfig_brca1_rna_validation.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

OUT_STEM = ROOT / "figures" / "figure2" / "fig2f_brca1_rna_validation"


def main():
    _mod.main(out_stem=OUT_STEM, fig_width=14)


if __name__ == "__main__":
    main()
