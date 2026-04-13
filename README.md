# Evee Manuscript Figures

Reproducible figures for the Evee paper: variant pathogenicity prediction using Evo2-7B embeddings.

## Quick Start

```bash
uv sync                    # install dependencies
uv run evee-ms figures     # generate all 17 figures
```

Individual figures:

```bash
uv run python scripts/figure1/fig_snv_heatmap.py
uv run python scripts/figure1/assemble_figure1.py   # composite + individual panels
```

All scripts produce `.png` (300 DPI) and `.pdf` (vector, TrueType 42).

## Structure

```
evee-manuscript/
├── pyproject.toml           # uv project (Python 3.12)
├── src/                     # CLI package (evee-ms)
├── theme/                   # Shared matplotlib theme (Goodfire palette)
├── artifacts/               # Pre-computed feather files (gitignored)
├── figures/                 # Generated output (gitignored)
└── scripts/
    ├── figure1/             # Main figure 1 panels + assembly
    ├── figure2/             # Main figure 2 panels + assembly
    ├── supplement/          # Supplementary figures
    └── prepare/             # Data generation (internal, needs goodfire-core)
```

## Commands

| Command | Description |
|---------|-------------|
| `uv run evee-ms figures` | Generate all figures from cached artifacts |
| `uv run evee-ms prepare` | Regenerate artifacts from raw data (internal) |

## Artifacts

All data is stored as Apache Arrow feather files in `artifacts/`. Figure scripts read these via `polars.read_ipc()`.

## Figures

### Figure 1 — Variant effect prediction

| Panel | Script | Description |
|-------|--------|-------------|
| B | `fig_snv_heatmap.py` | ClinVar SNV AUROC heatmap |
| C | `fig_indel_heatmap.py` | Indel stratified AUROC |
| D | `fig_conservation_lineplot.py` | AUROC by conservation tier |
| E | `fig_umap_pathogenicity.py` | UMAP colored by pathogenicity |
| F | `fig_umap_consequence.py` | UMAP colored by consequence |
| G | `fig_dms_spearman.py` | DMS Spearman barplot (4 genes) |

### Figure 2 — Interpretability

| Panel | Script | Description |
|-------|--------|-------------|
| B | `fig_probe_auroc_boxplot.py` | Annotation probe AUROC by category |
| C | `fig_disruption_umap.py` | Annotation head UMAP clusters |
| E | `fig_autointerp_lineplot.py` | Auto-interpretation benchmarking |

### Supplementary Figures

| Script | Description |
|--------|-------------|
| `supfig_activation_delta` | Per-position activation delta around SNV |
| `supfig_full_heatmap` | Full consequence heatmap (naive + deconfounded) |
| `supfig_dataset_characterization` | Dataset composition |
| `supfig_indel_heatmap` | Indel stratified AUROC heatmap |
| `supfig_indel_umap` | Indel UMAP embeddings |
| `supfig_dms_auroc` | DMS AUROC barplot (4 genes) |
| `supfig_context_window` | Context window sweep + gene clamping |
| `supfig_deconf_heatmap` | Deconfounded ClinVar heatmap |
| `supfig_disrupt_auroc` | Per-annotation disruption AUROC |
| `supfig_indel_full` | Full indel analysis |
| `supfig_topk_sweep` | Top-K position selection sweep |
| `supfig_topk_vs_window` | Top-K divergent vs contiguous window |

## Theme

`theme/mayo_theme.py` defines colors, fonts, and layout for all figures:

- **Goodfire palette**: warm orange for Evo2, cool tones for baselines
- **Typography**: Helvetica, Nature Methods sizing
- **Semantic colors**: pathogenic = orange, benign = steel blue
