"""
Shared plotting theme for the Mayo Mendelian manuscript.

Goodfire-branded color palette with Nature Methods–compliant typography.
All figures should import from here to ensure visual consistency.

Usage:
    from theme.mayo_theme import apply_theme, save_figure, add_panel_label
    from theme.mayo_theme import COLORS, METHOD_COLORS, CONSEQ_COLORS
    apply_theme()
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Color palette — Goodfire brand + semantic anchors
# ---------------------------------------------------------------------------
COLORS = {
    # Goodfire brand palette (warm tones)
    "gf_orange":   "#db8a48",  # Warm orange — cov probe (probe+)
    "gf_brown":    "#bbab8b",  # Warm taupe — mean probe (probe)
    "gf_beige":    "#D5CDBA",  # Warm beige background tint — Evo2 loss
    "gf_cream":    "#E5E3D3",  # Light background

    # Semantic
    "pathogenic":  "#db8a48",  # Goodfire orange (matches probe+)
    "benign":      "#5A7D9A",  # Steel blue (matches CADD)
    "vus":         "#999999",

    # UMAP accent
    "crimson":     "#C0392B",  # Rich red for indel pathogenic

    # Cool muted tones for external baselines
    "steel":       "#5A7D9A",
    "sage":        "#7B9E87",
    "lavender":    "#9B8EA8",
    "gray":        "#8C8C8C",
    "light_gray":  "#B0B0B0",
}

# Method → color mapping (consistent across ALL figures)
METHOD_COLORS = {
    # Evo2 family — Goodfire brand (warm, visually prominent)
    "Evo2 Covariance":         COLORS["gf_orange"],   # Cov probe — THE star method
    "Evo2 Mean":               COLORS["gf_brown"],    # Linear/mean probe
    "Evo2 Loss":               COLORS["gf_beige"],    # Warm beige

    # Legacy aliases (other scripts may still use old names)
    "Evo2 probe+":             COLORS["gf_orange"],
    "Evo2 probe":              COLORS["gf_brown"],
    "Evo2 loss":               COLORS["gf_beige"],

    # External baselines — cool muted tones (visually recede)
    "CADD v1.7":               COLORS["steel"],
    "CADD v1.6":               "#6B8DAA",
    "CADD v1.7 InDel":         COLORS["steel"],
    "AlphaMissense":           COLORS["sage"],
    "GPN-MSA":                 COLORS["lavender"],
    "NTv3":                    COLORS["gray"],
    "NTv3 subref probe\n(supervised)": COLORS["gray"],
    "AlphaGenome":             COLORS["light_gray"],
    "AlphaGenome composite":   COLORS["light_gray"],  # Legacy alias
    "REVEL":                   COLORS["light_gray"],
}

# DMS barplot: method key → (display label, color)
# Grouping: [baselines] gap [Evo2 loss + ClinVar probes] gap [DMS probes]
DMS_METHOD_SPEC = {
    "cadd_phred":          ("CADD v1.7",         COLORS["steel"]),
    "alphamissense":       ("AlphaMissense",     COLORS["sage"]),
    "evo2_loss":           ("Evo2 Loss",         COLORS["gf_beige"]),
    "clinvar_gfc_emb":     ("Evo2 Mean",         COLORS["gf_brown"]),
    "clinvar_covprobe64":  ("Evo2 Covariance",   COLORS["gf_orange"]),
    "dms_iid_L27_w64":     ("DMS Mean",          COLORS["gf_brown"]),
    "dms_iid_covprobe64":  ("DMS Covariance",    COLORS["gf_orange"]),
}

# Consequence type colors — unified with EVEE website (UmapCanvas.svelte)
CONSEQ_COLORS = {
    "Missense":       "#d98033",             # Warm orange
    "Synonymous":     "#66b366",             # Green
    "Frameshift":     "#cc4040",             # Red
    "Nonsense":       "#c97088",             # Dusty pink
    "Splice":         "#8c4db3",             # Purple
    "Intronic":       "#8099b3",             # Steel blue
    "In-frame":       "#c4a035",             # Teal
    "UTR":            "#2685d2",             # Bright blue
    "Other":          "#b0b0b0",             # Gray
    # Legacy keys for supfig5 (indel-specific breakdown)
    "In-frame del":   "#2a9d8f",
    "In-frame ins":   "#2a9d8f",
}

# Ordered categorical palette for fallback / generic plots
PALETTE = [
    COLORS["gf_orange"], COLORS["gf_brown"], COLORS["gf_beige"],
    COLORS["steel"], COLORS["sage"], COLORS["lavender"],
    COLORS["gray"], COLORS["light_gray"],
]

# ---------------------------------------------------------------------------
# Heatmap colormap — red → green (classic, no pastel hues)
# ---------------------------------------------------------------------------
_HEATMAP_COLORS = [
    "#D73027",  # Red (bad, ~0.5)
    "#FDAE61",  # Orange
    "#FFFFBF",  # Yellow (mid, ~0.75)
    "#A6D96A",  # Light green
    "#1A9850",  # Green (good, ~1.0)
]
HEATMAP_CMAP = mcolors.LinearSegmentedColormap.from_list("rdylgn", _HEATMAP_COLORS)

# ---------------------------------------------------------------------------
# Typography & layout
# ---------------------------------------------------------------------------
FONT_FAMILY = "Helvetica"
FONT_SIZE_PANEL_LABEL = 16   # Bold panel label (a, b, c)
FONT_SIZE_TITLE = 9           # Panel titles
FONT_SIZE_LABEL = 8           # Axis labels
FONT_SIZE_TICK = 8            # Tick labels
FONT_SIZE_LEGEND = 7          # Legend text
FONT_SIZE_CELL = 8            # Heatmap cell annotations

DPI_PNG = 300
LINEWIDTH = 1.2

# Nature figure widths
FIG_WIDTH_SINGLE = 3.5       # 89 mm — single column
FIG_WIDTH_DOUBLE = 7.2       # 183 mm — double column


# ---------------------------------------------------------------------------
# Method line styles for line plots (fig1c/d) — NO different markers
# ---------------------------------------------------------------------------
METHOD_LINE_STYLES = {
    "Evo2 Covariance": dict(color=METHOD_COLORS["Evo2 Covariance"], linewidth=2.2, zorder=5),
    "Evo2 Mean":       dict(color=METHOD_COLORS["Evo2 Mean"],       linewidth=2.2, zorder=5),
    "Evo2 Loss":       dict(color=METHOD_COLORS["Evo2 Loss"],       linewidth=1.8, zorder=4),
    "AlphaMissense":   dict(color=METHOD_COLORS["AlphaMissense"],   linewidth=1.5, zorder=3),
    "CADD v1.7":       dict(color=METHOD_COLORS["CADD v1.7"],       linewidth=1.5, zorder=3),
    "GPN-MSA":         dict(color=METHOD_COLORS["GPN-MSA"],         linewidth=1.5, zorder=3),
}


# ---------------------------------------------------------------------------
# Theme application
# ---------------------------------------------------------------------------
def apply_theme():
    """Apply the manuscript theme globally via matplotlib rcParams."""
    mpl.rcParams.update({
        # Font — Helvetica with semibold labels
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY, "Arial", "DejaVu Sans"],
        "font.size": FONT_SIZE_LABEL,
        "font.weight": "medium",

        # Axes — clean, modern
        "axes.titlesize": FONT_SIZE_TITLE,
        "axes.labelsize": FONT_SIZE_LABEL,
        "axes.titleweight": "semibold",
        "axes.labelweight": "semibold",
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),

        # Ticks
        "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,

        # Legend — Chart.js style: top-center, horizontal
        "legend.fontsize": FONT_SIZE_LEGEND,
        "legend.frameon": False,
        "legend.loc": "upper center",

        # Grid — very subtle
        "grid.alpha": 0.15,
        "grid.linewidth": 0.4,

        # Lines
        "lines.linewidth": LINEWIDTH,
        "lines.markersize": 4,

        # Figure
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": DPI_PNG,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.facecolor": "white",

        # PDF: TrueType 42 embedding (Nature requirement)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def method_color(name: str) -> str:
    """Get the consistent color for a method name."""
    return METHOD_COLORS.get(name, COLORS["light_gray"])


def method_colors(names: list[str]) -> list[str]:
    """Get colors for a list of methods, in order."""
    return [method_color(m) for m in names]


def save_figure(fig, path_stem, dpi=DPI_PNG, close=True):
    """Save a figure as both PNG and PDF.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path_stem : str or Path
        Output path WITHOUT extension (e.g., "figures/figure1/panels/fig1a").
    dpi : int
        Resolution for the PNG.
    close : bool
        Whether to close the figure after saving.
    """
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path_stem.with_suffix(".png"), dpi=dpi)
    fig.savefig(path_stem.with_suffix(".pdf"))

    if close:
        plt.close(fig)


def add_panel_label(ax, label, x=-0.10, y=1.05, fontsize=FONT_SIZE_PANEL_LABEL):
    """Add a bold lowercase panel label (a, b, c, ...) to an axes."""
    ax.text(
        x, y, label.lower(),
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="top",
        ha="left",
    )
