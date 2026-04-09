#!/usr/bin/env python3
"""
Assemble Figure 2 — single vector figure with all 5 panels.

Layout:
    Row 0: [A schematic placeholder]   [B annotation probe perf stub]
    Row 1: [C probe UMAP stub]         [D auto-interp screenshot placeholder]
    Row 2: [E benchmarking stub — full width]

Input:  (mostly stubs/placeholders for now)
Output: figures/figure2/figure2_assembled.{png,pdf}
        + individual panels in figures/figure2/panels/
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "figure2"))
from theme.mayo_theme import apply_theme, save_figure, add_panel_label

OUT_STEM = ROOT / "figures" / "figure2" / "figure2_assembled"

apply_theme()


def main():
    from fig2a import plot as plot_a
    from fig2b import plot as plot_b
    from fig2c import plot as plot_c
    from fig2d import plot as plot_d
    from fig2e import plot as plot_e

    fig = plt.figure(figsize=(16, 20))
    gs = fig.add_gridspec(3, 2,
                          height_ratios=[0.45, 0.35, 0.30],
                          hspace=0.22, wspace=0.25)

    # Row 0: Schematic (A) + Annotation probe performance (B)
    ax_a = fig.add_subplot(gs[0, 0])
    plot_a(ax_a)
    add_panel_label(ax_a, "a")

    ax_b = fig.add_subplot(gs[0, 1])
    plot_b(ax_b)
    add_panel_label(ax_b, "b")

    # Row 1: Probe UMAP (C) + Auto-interp screenshot (D)
    ax_c = fig.add_subplot(gs[1, 0])
    plot_c(ax_c)
    add_panel_label(ax_c, "c")

    ax_d = fig.add_subplot(gs[1, 1])
    plot_d(ax_d)
    add_panel_label(ax_d, "d")

    # Row 2: Benchmarking (E) — full width
    ax_e = fig.add_subplot(gs[2, :])
    plot_e(ax_e)
    add_panel_label(ax_e, "e")

    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")

    # Also generate individual panels
    print("\nRegenerating individual panels...")
    from fig2a import main as main_a
    from fig2b import main as main_b
    from fig2c import main as main_c
    from fig2d import main as main_d
    from fig2e import main as main_e

    for panel_main in (main_a, main_b, main_c, main_d, main_e):
        panel_main()


if __name__ == "__main__":
    main()
