# Evee Manuscript Figures

Reproducible figures for the Evee paper: variant pathogenicity prediction using Evo2-7B embeddings.

## Quick Start

```bash
uv sync                    # install dependencies
uv run evee-ms figures     # generate all 17 figures
```

Individual figures:

```bash
uv run python scripts/figure1/fig1b.py
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
| A | `fig1a.py` | Experimental design (placeholder) |
| B | `fig1b.py` | ClinVar SNV AUROC heatmap |
| C | `fig1c.py` | Indel stratified AUROC |
| D | `fig1d.py` | UMAP colored by pathogenicity |
| E | `fig1e.py` | UMAP colored by consequence |
| F | `fig1f.py` | AUROC by conservation tier |
| G | `fig1g.py` | DMS Spearman barplot (4 genes) |

### Figure 2 — Interpretability

| Panel | Script | Description |
|-------|--------|-------------|
| A | `fig2a.py` | Framework schematic (placeholder) |
| B | `fig2b.py` | Annotation probe AUROC by category |
| C | `fig2c.py` | Annotation head UMAP clusters |
| D | `fig2d.py` | Auto-interpretation example (placeholder) |
| E | `fig2e.py` | Auto-interpretation benchmarking |

### Supplementary Figures

| Script | Description |
|--------|-------------|
| `supfig_activation_delta` | Per-position activation delta around SNV |
| `supfig_full_heatmap` | Full consequence heatmap (naive + deconfounded) |
| `supfig_dataset_characterization` | Dataset composition |
| `supfig_indel_heatmap` | Indel stratified AUROC heatmap |
| `supfig_indel_umap` | Indel UMAP embeddings |
| `supfig_dms_auroc` | DMS AUROC barplot (4 genes) |
| `supfig_combined_umap` | Combined SNV + indel + VUS UMAP |
| `supfig_context_window` | Context window sweep + gene clamping |
| `supfig_deconf_heatmap` | Deconfounded ClinVar heatmap |
| `supfig_disrupt_auroc` | Per-annotation disruption AUROC |
| `supfig_indel_full` | Full indel analysis |
| `supfig_ksparse` | K-sparse SAE probe performance |
| `supfig_missense_conservation` | Conservation tiers (missense only) |
| `supfig_snv_umap` | SNV-only UMAP |
| `supfig_topk_sweep` | Top-K position selection sweep |
| `supfig_topk_vs_window` | Top-K divergent vs contiguous window |

## Theme

`theme/mayo_theme.py` defines colors, fonts, and layout for all figures:

- **Goodfire palette**: warm orange for Evo2, cool tones for baselines
- **Typography**: Helvetica, Nature Methods sizing
- **Semantic colors**: pathogenic = orange, benign = steel blue
