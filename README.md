# EVEE Manuscript Figures

Reproducible figures for [EVEE: Interpretable variant effect prediction from genomic foundation model embeddings](https://github.com/goodfire-ai/evee-manuscript).

## Quick Start

```bash
uv sync                    # install dependencies
uv run evee-ms figures     # generate all 14 figures
```

Individual figures:

```bash
uv run python scripts/figure1/fig1b_snv_heatmap.py
uv run python scripts/figure2/fig2ce_autointerp_barchart.py
```

All scripts produce `.png` (300 DPI) and `.pdf` (vector, TrueType 42).

## Structure

```
evee-manuscript/
├── pyproject.toml           # uv project (Python 3.12)
├── src/cli.py               # CLI entry point (evee-ms)
├── theme/theme.py           # Shared matplotlib theme (Goodfire palette)
├── artifacts/               # Pre-computed feather/csv/json data files
├── figures/                 # Generated output (png + pdf)
├── notebooks/               # Demo notebook (probe inference)
└── scripts/
    ├── figure1/             # Main figure 1 panels (b–g)
    ├── figure2/             # Main figure 2 panels (b, c, e)
    ├── supplement/          # Supplementary figures (S1–S5, S9–S10)
    └── prepare/             # Data generation (internal, needs goodfire-core)
```

## Commands

| Command | Description |
|---------|-------------|
| `uv run evee-ms figures` | Generate all figures from cached artifacts |
| `uv run evee-ms prepare` | Regenerate artifacts from raw data (internal) |

## Figures

### Figure 1 — Variant effect prediction

| Panel | Script | Description |
|-------|--------|-------------|
| b | `fig1b_snv_heatmap.py` | ClinVar SNV AUROC heatmap by consequence type |
| c | `fig1c_indel_heatmap.py` | Zero-shot indel AUROC by consequence/size/direction |
| d | `fig1d_conservation_lineplot.py` | AUROC by phyloP100way conservation tier |
| e | `fig1e_umap_pathogenicity.py` | UMAP colored by pathogenicity label |
| f | `fig1f_umap_consequence.py` | UMAP colored by VEP consequence type |
| g | `fig1g_dms_spearman.py` | DMS Spearman correlation (BRCA1, BRCA2, TP53, LDLR) |

### Figure 2 — Interpretability

| Panel | Script | Description |
|-------|--------|-------------|
| b | `fig2b_probe_auroc_boxplot.py` | Annotation probe AUROC by category |
| c | `fig2ce_autointerp_barchart.py` | Context ablation composite score bar chart |
| e | `fig2ce_autointerp_barchart.py` | Per-axis breakdown (mechanism, accuracy, specificity) |

### Supplementary Figures

| Figure | Script | Description |
|--------|--------|-------------|
| S1 | `supfig1_layer_sweep.py` | Layer sweep across Evo 2-7B blocks |
| S2 | `supfig2_context_window.py` | Context window sweep + gene clamping |
| S3 | `supfig3_topk_vs_window.py` | Top-K divergent vs contiguous window |
| S4 | `supfig4_deconf_heatmap.py` | Deconfounded ClinVar benchmark |
| S5 | `supfig5_dataset_characterization.py` | Dataset composition and pathogenic rates |
| S9 | `supfig9_autointerp_ablation.py` | Interpretation quality by pathogenicity class + per-axis line plots |
| S10 | `supfig9_autointerp_ablation.py` | Score distributions across context configurations |

## Artifacts

Pre-computed data files in `artifacts/`, read via `polars.read_ipc()` (feather) or standard csv/json:

| File | Used by | Description |
|------|---------|-------------|
| `snv_benchmark.feather` | Fig 1b | SNV AUROC by consequence type and method |
| `indel_stratified.feather` | Fig 1c | Indel AUROC by consequence, size, direction |
| `conservation_benchmark.feather` | Fig 1d | AUROC by phyloP conservation tier |
| `umap_combined.feather` | Fig 1e, 1f | UMAP coordinates + pathogenicity/consequence labels |
| `dms_benchmark.feather` | Fig 1g | DMS Spearman correlations for 4 genes |
| `heads.feather` | Fig 2b | Annotation probe metadata (names, categories) |
| `token_eval.json` | Fig 2b | Per-head binary AUROC values (357 probes) |
| `context_ablation_eval.feather` | Fig 2c/e, S9, S10 | LLM interpretation scores across context configs |
| `layer_sweep_evo2_7b.csv` | S1 | AUROC by Evo 2 transformer layer |
| `context_window_sweep.feather` | S2 | AUROC vs context window size |
| `topk_vs_window.feather` | S3 | Top-K vs contiguous window comparison |
| `deconf_benchmark.feather` | S4 | Deconfounded benchmark AUROC |
| `dataset_characterization.feather` | S5 | Variant counts and pathogenic rates |

## Theme

`theme/theme.py` defines colors, fonts, and layout for all figures:

- **Goodfire palette**: warm orange for Evo 2, cool tones for baselines
- **Typography**: Helvetica, Nature Methods sizing
- **Semantic colors**: pathogenic (orange), benign (steel blue), VUS (gray)
