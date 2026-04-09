#!/usr/bin/env python3
"""
Assemble Figure 1 — single vector figure with all 7 panels.

Calls each panel's plot() function directly into a shared gridspec,
ensuring perfect axis alignment. No PNG compositing.

Layout:
    Row 0: [A placeholder — full width]
    Row 1: [B SNV heatmap]  [C indel heatmap]  [D conservation lineplot]
    Row 2: [E UMAP pathogenicity]  [F UMAP consequence]
    Row 3: [G DMS bars — 1x4 subgrid]

Input:  data/panels/*, data/embeddings/*
Output: figures/figure1/figure1_assembled.{png,pdf}
        + individual panels in figures/figure1/panels/
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "figure1"))
from theme.mayo_theme import apply_theme, save_figure, add_panel_label

OUT_STEM = ROOT / "figures" / "figure1" / "figure1_assembled"

apply_theme()


def main():
    # Import panel plot functions
    from fig1a import plot as plot_a
    from fig1b import plot as plot_b
    from fig1c import plot as plot_c
    from fig1d import plot as plot_d
    from fig1e import plot as plot_e
    from fig1f import plot as plot_f
    from fig1g import plot as plot_g

    # Create composite figure — 4 rows, single column so every row spans
    # the same horizontal width.
    fig = plt.figure(figsize=(21, 27))
    gs = fig.add_gridspec(4, 1,
                          height_ratios=[0.45, 0.55, 0.35, 0.35],
                          hspace=0.22)

    # Row 0: Placeholder (A) — full width
    gs_a = gs[0].subgridspec(1, 1)
    ax_a = fig.add_subplot(gs_a[0, 0])
    plot_a(ax_a)
    add_panel_label(ax_a, "a")

    # Row 1: B + C + D side-by-side
    gs_bcd = gs[1].subgridspec(1, 3, width_ratios=[1.6, 0.8, 1.2], wspace=0.35)
    ax_b = fig.add_subplot(gs_bcd[0, 0])
    plot_b(ax_b, aspect="auto")
    add_panel_label(ax_b, "b")

    ax_c = fig.add_subplot(gs_bcd[0, 1])
    plot_c(ax_c, aspect="auto")
    add_panel_label(ax_c, "c")

    ax_d = fig.add_subplot(gs_bcd[0, 2])
    plot_f(ax_d)
    add_panel_label(ax_d, "d")

    # Row 2: UMAPs (E, F) — use subgridspec to match row width
    gs_ef = gs[2].subgridspec(1, 2, wspace=0.25)
    ax_e = fig.add_subplot(gs_ef[0, 0])
    plot_d(ax_e)
    add_panel_label(ax_e, "e")

    ax_f = fig.add_subplot(gs_ef[0, 1])
    plot_e(ax_f)
    add_panel_label(ax_f, "f")

    # Row 3: DMS bars (G) — full width, 1x4 subgrid
    gs_g = gs[3].subgridspec(1, 4, wspace=0.25)
    axes_g = [fig.add_subplot(gs_g[0, c]) for c in range(4)]
    plot_g(axes_g)
    add_panel_label(axes_g[0], "g")

    # Align y-axes vertically within each row
    fig.align_ylabels([ax_b, ax_c, ax_d])
    fig.align_ylabels([ax_e, ax_f])

    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")

    # Also generate individual panels
    print("\nRegenerating individual panels...")
    from fig1a import main as main_a
    from fig1b import main as main_b
    from fig1c import main as main_c
    from fig1d import main as main_d
    from fig1e import main as main_e
    from fig1f import main as main_f
    from fig1g import main as main_g

    for panel_main in (main_a, main_b, main_c, main_d, main_e, main_f, main_g):
        panel_main()


if __name__ == "__main__":
    main()
