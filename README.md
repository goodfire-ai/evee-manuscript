# EVEE Manuscript Figures

Reproducible figures for [Interpretable variant effect prediction from genomic foundation model representations](https://www.biorxiv.org/content/10.64898/2026.04.10.717844).

## Data availability

The full per-variant table backing the EVEE web app — one row per ClinVar variant (4.25 M total) with the Evo 2 pathogenicity score and ~4,900 probe outputs — is archived on Zenodo:

- **DOI**: [10.5281/zenodo.19701997](https://doi.org/10.5281/zenodo.19701997)
- **Record**: https://zenodo.org/records/19701997

Released as five chromosome-balanced Parquet shards (`clean_shard_0.parquet` … `clean_shard_4.parquet`, 6.8–7.3 GB each) plus a `manifest.json`. Read all shards as one logical table:

```python
import polars as pl
df = pl.scan_parquet("clean_shard_*.parquet")
```

## MCP server

EVEE is also available as an MCP (Model Context Protocol) server, allowing Claude and other LLM agents to query variant predictions, disruption profiles, and interpretations programmatically: [goodfire-ai/evee-mcp](https://github.com/goodfire-ai/evee-mcp).

## Quick Start

```bash
uv sync                    # install dependencies
uv run evee-ms figures     # generate all main + supplementary figures
```

Individual figures:

```bash
uv run python scripts/figure1/fig1b_snv_heatmap.py
uv run python scripts/figure3/fig3d_interp_quality.py
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
    ├── figure1/             # Figure 1 — pathogenicity classification (b–d)
    ├── figure2/             # Figure 2 — representations & transfer (a–d)
    ├── figure3/             # Figure 3 — mechanism decoding (b–d)
    ├── figure4/             # Figure 4 — RNA-seq validation (a–c)
    ├── figure5/             # Figure 5 — clinical & population validation (a–d)
    ├── supplement/          # Supplementary figures
    └── prepare/             # Data generation (internal, needs goodfire-core)
```

## Commands

| Command | Description |
|---------|-------------|
| `uv run evee-ms figures` | Generate all figures from cached artifacts |
| `uv run evee-ms prepare` | Regenerate artifacts from raw data (internal) |

## Figures

### Figure 1 — Pathogenicity classification

| Panel | Script | Description |
|-------|--------|-------------|
| b | `figure1/fig1b_snv_heatmap.py` | ClinVar SNV AUROC heatmap by consequence type |
| c | `figure1/fig1c_deconf_heatmap.py` | Deconfounded ClinVar benchmark AUROC heatmap |
| d | `figure1/fig1d_indel_heatmap.py` | Zero-shot indel AUROC by consequence/size/direction |

### Figure 2 — Representations & transfer

| Panel | Script | Description |
|-------|--------|-------------|
| a | `figure2/fig2a_conservation_lineplot.py` | AUROC by phyloP100way conservation tier |
| b | `figure2/fig2b_umap_pathogenicity.py` | UMAP colored by pathogenicity label |
| c | `figure2/fig2c_umap_consequence.py` | UMAP colored by VEP consequence type |
| d | `figure2/fig2d_dms_spearman.py` | DMS Spearman correlation (BRCA1, BRCA2, TP53, LDLR) |

### Figure 3 — Mechanism decoding

| Panel | Script | Description |
|-------|--------|-------------|
| b | `figure3/fig3b_probe_auroc_boxplot.py` | Annotation probe AUROC by category |
| c | `figure3/fig3c_mechanism_recovery.py` | Within-gene mechanism-class recovery across four genes (5-fold CV balanced accuracy) |
| d | `figure3/fig3d_interp_quality.py` | Context ablation interpretation-quality composite bar chart |

### Figure 4 — RNA-seq validation

| Panel | Script | Description |
|-------|--------|-------------|
| a,b | `figure4/fig4ab_brca1_rna.py` | BRCA1 rs80358099 / JHOS-4 browser figure: EVEE disruption + RNA-seq sashimi |
| c | `figure4/fig4c_splice_nmd_headline.py` | Combined splice + NMD validation bar chart |

### Figure 5 — Clinical & population validation

| Panel | Script | Description |
|-------|--------|-------------|
| a | `figure5/fig5a_ldlr_roc.py` | LDLR severity ROC: any-FH vs presymptomatic carrier (EVEE / AlphaMissense / CADD) |
| b | `figure5/fig5b_ldlr_violin.py` | LDLR EVEE pathogenicity distribution by clinical tier |
| c | `figure5/fig5c_finngen_qq.py` | EVEE-score dose-response QQ in FinnGen R12: EVEE≥0.95 / ≥0.80 vs 5:1 AF-matched <0.80 control |
| d | `figure5/fig5d_acmg_stacked.py` | ACMG/AMP evidence points for 6 candidate FinnGen R12 ClinVar variants |

### Supplementary Figures

| Script | Description |
|--------|-------------|
| `supplement/supfig_layer_sweep.py` | Layer sweep across Evo 2-7B blocks |
| `supplement/supfig_context_window.py` | Context window sweep + gene clamping |
| `supplement/supfig_topk_vs_window.py` | Top-K divergent vs contiguous window |
| `supplement/supfig_dataset_characterization.py` | Dataset composition and pathogenic rates |
| `supplement/supfig_autointerp_ablation.py` | Interpretation quality by pathogenicity class, score distributions, per-axis line plots |
| `supplement/supfig_splice_vaf_validation.py` | VAF vs 1−PSI scatter: EVEE splice + branchpoint variant validation (n=66 observations) |
| `supplement/supfig_ra_cohort.py` | Mayo RA cohort: gene-level pathogenicity score distribution by pathway |
| (peraxis) | `figure3/fig3d_interp_quality.py` also writes `supfig_autointerp_peraxis_barchart` (per-axis breakdown) |

## Artifacts

Pre-computed data files in `artifacts/`, read via `polars.read_ipc()` (feather) or standard csv/json:

| File | Used by | Description |
|------|---------|-------------|
| `snv_benchmark.feather` | Fig 1b | SNV AUROC by consequence type and method |
| `deconf_benchmark.feather` | Fig 1c | Deconfounded benchmark AUROC |
| `indel_stratified.feather` | Fig 1d | Indel AUROC by consequence, size, direction |
| `conservation_benchmark.feather` | Fig 2a | AUROC by phyloP conservation tier |
| `umap_combined.feather` | Fig 2b, 2c | UMAP coordinates + pathogenicity/consequence labels |
| `dms_benchmark.feather` | Fig 2d | DMS Spearman correlations for 4 genes |
| `heads.feather` | Fig 3b | Annotation probe metadata (names, categories) |
| `token_eval.json` | Fig 3b | Per-head binary AUROC values (357 probes) |
| `mechanism_recovery.json` | Fig 3c | Per-gene 5-fold CV balanced accuracy for mechanism-class recovery (LDLR, LMNA, MYH7, TP53) |
| `context_ablation_eval.feather` | Fig 3d, supplement | LLM interpretation scores across context configs |
| `brca1_disruption_profile.feather` | Fig 4a,b | Per-1bp EVEE Δ profile for BRCA1 rs80358099 (512 rows: 256 fwd + 256 bwd positions) |
| `brca1_e22_data.json` | Fig 4a,b | Snaptron junction read counts + variant metadata for BRCA1 clinical exon 20 (legacy GENCODE-rank filename; JHOS-4, JHOS-2, A2780) |
| `splice_nmd_validation_per_variant.csv` | Fig 4c | 82 unique variant--carrier pairs: tier, gene, HGVSc, expected mechanism, CCLE carrier line, VAF, zygosity, RNA-seq readout, disruption, Validated, EVEE path |
| `ldlr_severity_per_variant.csv` | Fig 5a, 5b | LDLR clinical cohort per-variant table: tier3, EVEE/AlphaMissense/CADD scores, consequence |
| `finngen_r12_qq_pathogenicity.feather` | Fig 5c | Per-variant min(p) for FinnGen R12 QQ grouped by EVEE score (≥0.95 / ≥0.80 / 5:1 AF-matched <0.80 control); generated by `scripts/prepare/prepare_finngen_r12_qq_pathogenicity.py` |
| `finngen_acmg.feather` | Fig 5d | ACMG points for 6 candidate FinnGen R12 ClinVar variants (generated by `scripts/prepare/finngen_acmg_artifact.py`) |
| `splice_vaf_psi.feather` | supplement | 66 splice/branchpoint variant observations: VAF, 1−PSI, zygosity (from Sanger CMP + Snaptron) |
| `layer_sweep_evo2_7b.csv` | supplement | AUROC by Evo 2 transformer layer |
| `context_window_sweep.feather` | supplement | AUROC vs context window size |
| `topk_vs_window.feather` | supplement | Top-K vs contiguous window comparison |
| `dataset_characterization.feather` | supplement | Variant counts and pathogenic rates |
| `ra_cohort.feather` | supplement (RA cohort) | De-identified Mayo RA cohort: 299 rare variants (gene, EVEE pathogenicity, ClinVar candidate bin) |

## Theme

`theme/theme.py` defines colors, fonts, and layout for all figures:

- **Goodfire palette**: warm orange for Evo 2, cool tones for baselines
- **Typography**: Helvetica, journal-standard sizing
- **Semantic colors**: pathogenic (orange), benign (steel blue), VUS (gray)
