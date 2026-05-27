# EVEE Manuscript Figures

Reproducible figures for [EVEE: Interpretable variant effect prediction from genomic foundation model embeddings](https://www.biorxiv.org/content/10.64898/2026.04.10.717844).

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
| i | `fig1i_finngen_qq.py` | FinnGen R12 QQ plot: ClinVar P/LP vs AF-matched B/LB variants |

### Figure 2 — Interpretability

| Panel | Script | Description |
|-------|--------|-------------|
| b | `fig2b_probe_auroc_boxplot.py` | Annotation probe AUROC by category |
| c | `fig2c_autointerp_barchart.py` | Context ablation composite score bar chart |
| e | `fig2e_cohort_genes.py` | Mayo RA cohort: gene-level pathogenicity score distribution by pathway |

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
| S11 | `fig2c_autointerp_barchart.py` | Per-axis breakdown bar chart (mechanism, accuracy, specificity) — moved from old Fig 2e |
| FinnGen | `supfig_finngen_acmg.py` | ACMG/AMP evidence points for 6 FinnGen R12 ClinVar resubmission candidates |
| FinnGen QQ | `evee_resubmission_natgen/fig_finngen_r12_qq_clinvar_af_matched.py` | AF-matched QQ plot: pathogenic/LP vs benign ClinVar variants in FinnGen R12 |
| RNA-seq splice | `supfig_splice_vaf_validation.py` | VAF vs 1−PSI scatter: EVEE splice + branchpoint variant validation (n=66 observations) |
| RNA-seq headline | `supfig_splice_nmd_headline.py` | Combined splice + NMD validation bar chart (splice 30/47=64%, NMD 14/32=44%) |
| BRCA1 case | `supfig_brca1_rna_validation.py` | BRCA1 rs80358099 / JHOS-4 multi-track browser figure: EVEE disruption + RNA-seq sashimi |

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
| `context_ablation_eval.feather` | Fig 2c, S9–S11 | LLM interpretation scores across context configs |
| `handoff_final.parquet` (external) | Fig 2e | Mayo RA cohort 299 rare variants with EVEE pathogenicity scores |
| `layer_sweep_evo2_7b.csv` | S1 | AUROC by Evo 2 transformer layer |
| `context_window_sweep.feather` | S2 | AUROC vs context window size |
| `topk_vs_window.feather` | S3 | Top-K vs contiguous window comparison |
| `deconf_benchmark.feather` | S4 | Deconfounded benchmark AUROC |
| `dataset_characterization.feather` | S5 | Variant counts and pathogenic rates |
| `finngen_r12_qq_clinvar.feather` | Fig 1i | Per-variant min(p) and ClinVar label for FinnGen R12 QQ plot (AF-matched P/LP vs B/LB) |
| `finngen_r12_qq_clinvar_pertest.feather` | Fig 1i | Per-test FinnGen association data for QQ plot stratified by phenotype category |
| `finngen_resubmission.feather` | FinnGen supfig | ACMG points for 6 FinnGen R12 resubmission candidates (generated by `scripts/prepare/finngen_resubmission_artifact.py`) |
| `finngen_r12_qq_clinvar.feather` | FinnGen QQ fig | Pre-computed QQ curve points + λ_GC for ClinVar-labeled FinnGen R12 variants (pathogenic/LP vs benign/LB), AF-matched; both per-test and per-variant Bonferroni panels (generated by `scripts/prepare/prepare_finngen_r12_qq_clinvar.py`) |
| `splice_vaf_psi.feather` | RNA-seq splice | 66 splice/branchpoint variant observations: VAF, 1−PSI, zygosity (from Sanger CMP + Snaptron) |
| `splice_nmd_validation.feather` | RNA-seq headline | 180 raw observations (82 splice, 98 NMD): variant_id, gene, VAF, dm (VAF-corrected disruption), tier |
| `brca1_disruption_profile.feather` | BRCA1 case | Per-1bp EVEE Δ profile for BRCA1 rs80358099 (512 rows: 256 fwd + 256 bwd positions) |
| `brca1_e22_data.json` | BRCA1 case | Snaptron junction read counts + variant metadata for BRCA1 exon 22 (JHOS-4, JHOS-2, A2780) |

## Theme

`theme/theme.py` defines colors, fonts, and layout for all figures:

- **Goodfire palette**: warm orange for Evo 2, cool tones for baselines
- **Typography**: Helvetica, Nature Methods sizing
- **Semantic colors**: pathogenic (orange), benign (steel blue), VUS (gray)
