# Mayo Mendelian Manuscript — Figures

Variant pathogenicity prediction using Evo2-7B embeddings and Sparse Autoencoders (SAEs) for interpretability.

## Repository Structure

```
mayo_manuscript/
├── data/
│   ├── panels/              # Small derived CSVs/NPZ, one per panel
│   ├── embeddings/          # Pre-computed UMAP coordinates (safetensors + feather)
│   ├── clinvar/             # Symlinks to production activation datasets (44 TB)
│   ├── dms/                 # DMS activation datasets
│   └── paper_metrics/       # Summary metrics for text
├── scripts/
│   ├── prepare/             # Data generation (needs goodfire-core, run rarely)
│   ├── figure1/             # Pure plotting (reads only from data/, no ML imports)
│   ├── figure2/
│   ├── supplement/
│   └── _archive/            # Old scripts preserved for reference
├── figures/                 # Output PNGs and PDFs
├── theme/                   # Shared plotting theme
└── generate_all.py          # Single entry point to regenerate all figures
```

## Two-Phase Workflow

### Phase 1: Prepare data (rare)

Run only when embeddings or benchmark data change. Requires goodfire-core and possibly a GPU.

```bash
# UMAP embeddings (PCA + UMAP, ~5 min each)
uv run python scripts/prepare/umap_snv.py
uv run python scripts/prepare/umap_indel.py
uv run python scripts/prepare/umap_combined.py
```

Panel CSVs in `data/panels/` are generated upstream and committed — no prepare step needed.

### Phase 2: Plot (frequent)

Pure matplotlib. No goodfire-core dependency. Regenerates all figures from cached data:

```bash
uv run python generate_all.py
```

Or individual panels:

```bash
uv run python scripts/figure1/fig1b.py
uv run python scripts/figure1/assemble_figure1.py  # Full composite + individual panels
```

All scripts produce both `.png` (300 DPI) and `.pdf` (vector, TrueType 42).

## Figure Descriptions

### Figure 1 — Evo2 covariance probe achieves state-of-the-art variant effect prediction across variant types

| Panel | Description | Data |
|-------|-------------|------|
| **A** | Experimental design (external diagram) | None — placeholder |
| **B** | ClinVar SNV pathogenicity prediction — AUROC heatmap | `panels/fig1a.csv` |
| **C** | Zero-shot generalization to indels — stratified AUROC | `panels/supfig4.csv` |
| **D** | UMAP — covariance embeddings colored by pathogenicity | `embeddings/umap_combined.*` |
| **E** | UMAP — covariance embeddings colored by consequence | `embeddings/umap_combined.*` |
| **F** | Performance vs conservation tier (all variants) | `panels/fig1c.csv` |
| **G** | DMS Spearman |ρ| barplot, 4 genes | `panels/fig1f.csv` |

### Figure 2 — Interpretable predictions through annotation disruption profiling and LLM-based synthesis

| Panel | Description | Data |
|-------|-------------|------|
| **A** | Interpretability framework schematic (external) | None — placeholder |
| **B** | Annotation probe AUROC by category (boxplot) | `panels/fig2b.csv` |
| **C** | Score-space UMAP — annotation head clusters | `panels/fig2c.csv` |
| **D** | Auto-interpretation example (external screenshot) | None — placeholder |
| **E** | Auto-interpretation benchmarking (line plot) | `panels/fig2e.csv` |

### Supplementary Figures

| Figure | Description | Data |
|--------|-------------|------|
| **Sup 1** | Per-position activation delta | `panels/supfig1.npz` |
| **Sup 2** | Full heatmap — all consequences, naive + deconf | `panels/supfig2_*.csv` |
| **Sup 3** | Dataset characterization | `panels/supfig3.csv` |
| **Sup 5** | Indel UMAP embeddings | `embeddings/umap_indel.*` |
| **Sup 6** | DMS AUROC barplot, 4 genes | `panels/fig1f.csv` |
| **Deconf heatmap** | Deconfounded ClinVar AUROC heatmap | `panels/fig1b.csv` |
| **Missense conservation** | Conservation tiers, missense only | `panels/fig1d.csv` |
| **SNV UMAP** | SNV-only covariance probe UMAP | `embeddings/umap_snv.*` |
| **K-sparse** | K-sparse SAE probe performance | `panels/fig2a.csv` |
| **Disruption AUROC** | Per-annotation disruption vs pathogenicity | `panels/fig2_disrupt_auroc.csv` |

## Architecture

### How panels connect

Each panel script exposes `plot(ax)` which draws onto given matplotlib axes. The assembly script creates one Figure with gridspec and calls each `plot()`:

```
assemble_figure1.py
  ├── fig1a.plot(ax)      # Placeholder
  ├── fig1b.plot(ax)      # Uses _heatmap_common
  ├── fig1c.plot(ax)      # Indel heatmap (promoted from supfig4)
  ├── fig1d.plot(ax)      # Uses _umap_common (combined UMAP left)
  ├── fig1e.plot(ax)      # Uses _umap_common (combined UMAP right)
  ├── fig1f.plot(ax)      # Conservation line plot
  └── fig1g.plot(axes)    # Takes 4 axes: flattened 2×2

assemble_figure2.py
  ├── fig2a.plot(ax)      # Placeholder
  ├── fig2b.plot(ax)      # Stub — data TBD
  ├── fig2c.plot(ax)      # Stub — data TBD
  ├── fig2d.plot(ax)      # Placeholder
  └── fig2e.plot(ax)      # Stub — data TBD
```

### Shared modules

- `scripts/figure1/_heatmap_common.py` — Heatmap preparation and plotting (used by fig1b, supfig_deconf_heatmap)
- `scripts/figure1/_umap_common.py` — UMAP data loading and formatting (used by fig1d, fig1e)

### How to modify

**Colors, fonts, method names**: Edit `theme/mayo_theme.py`. Key dicts:
- `METHOD_COLORS` — method → hex
- `DMS_METHOD_SPEC` — DMS barplot labels + colors
- `METHOD_LINE_STYLES` — line plot styles
- `CONSEQ_COLORS` — consequence colors for UMAPs
- `HEATMAP_CMAP` — pastel RdYlGn colormap

**Method display names**: Update rename dicts in `_heatmap_common.py` and `fig1f.METHOD_RENAME`, plus `METHOD_COLORS` keys.

**Composite layout**: Edit `assemble_figure1.py` — adjust `height_ratios`, `hspace`, `wspace`.

**Add a new panel**: Create `scripts/figure1/fig1x.py` with `plot(ax)` + `main()`. Add to assembly gridspec. Add data to `data/panels/`.

**Add a new UMAP**: Create `scripts/prepare/umap_foo.py` to compute + cache. Create plotting script to use the cache.

## Theme

`theme/mayo_theme.py`:
- **Goodfire palette**: warm orange (`#db8a48`) for probe+, taupe (`#bbab8b`) for probe, cool tones for baselines
- **Semantic**: pathogenic = orange, benign = steel blue, crimson for frameshift/indel pathogenic
- **Typography**: Helvetica, semibold labels
- **Heatmap**: Pastel RdYlGn (muted rose → cream → sage → green)
