#!/usr/bin/env python3
"""
Assemble Figure 2 — single vector figure with all panels.

Layout:
    Row 0: [A schematic placeholder]   [B annotation probe AUROC]
    Row 1: [C disruption UMAP]         [D auto-interp screenshot placeholder]
    Row 2: [E autointerp composite]    [F autointerp per-axis breakdown (3 stacked)]

Input:  artifacts/*
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
    from fig_probe_auroc_boxplot import plot as plot_probes
    from fig_disruption_umap import plot as plot_disruption
    from fig_autointerp_lineplot import plot as plot_autointerp
    from fig_autointerp_axes import plot as plot_autointerp_axes

    fig = plt.figure(figsize=(16, 24))
    gs = fig.add_gridspec(3, 2,
                          height_ratios=[0.35, 0.28, 0.45],
                          hspace=0.22, wspace=0.25)

    # Row 0: Schematic placeholder (A) + Annotation probe performance (B)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor("#f5f5f5")
    ax_a.text(0.5, 0.5, "(framework schematic placeholder)",
              ha="center", va="center", fontsize=14, color="#999999",
              transform=ax_a.transAxes)
    ax_a.set_xticks([]); ax_a.set_yticks([])
    add_panel_label(ax_a, "a")

    ax_b = fig.add_subplot(gs[0, 1])
    plot_probes(ax_b)
    add_panel_label(ax_b, "b")

    # Row 1: Disruption UMAP (C) + Auto-interp screenshot placeholder (D)
    ax_c = fig.add_subplot(gs[1, 0])
    plot_disruption(ax_c)
    add_panel_label(ax_c, "c")

    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor("#f5f5f5")
    ax_d.text(0.5, 0.5, "(interpretation example placeholder)",
              ha="center", va="center", fontsize=14, color="#999999",
              transform=ax_d.transAxes)
    ax_d.set_xticks([]); ax_d.set_yticks([])
    add_panel_label(ax_d, "d")

    # Row 2: Autointerp composite (E) + Per-axis breakdown (F, 3 stacked)
    ax_e = fig.add_subplot(gs[2, 0])
    plot_autointerp(ax_e)
    add_panel_label(ax_e, "e")

    gs_f = gs[2, 1].subgridspec(3, 1, hspace=0.35)
    axes_f = [fig.add_subplot(gs_f[r, 0]) for r in range(3)]
    plot_autointerp_axes(axes_f)
    add_panel_label(axes_f[0], "f")

    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")

    # Also generate individual panels
    print("\nRegenerating individual panels...")
    from fig_probe_auroc_boxplot import main as main_probes
    from fig_disruption_umap import main as main_disruption
    from fig_autointerp_lineplot import main as main_autointerp
    from fig_autointerp_axes import main as main_autointerp_axes

    for panel_main in (main_probes, main_disruption, main_autointerp,
                       main_autointerp_axes):
        panel_main()


if __name__ == "__main__":
    main()
