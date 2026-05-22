# Figure: EVEE Pathogenicity vs FinnGen R12 — ClinVar-labeled Variants, AF-matched QQ Plot

**File:** `fig_finngen_r12_qq_clinvar_af_matched.png`

---

## Figure Legend

**Figure X. EVEE pathogenicity discriminates disease association in FinnGen R12 for ClinVar-annotated variants, independent of allele frequency.**

Quantile–quantile (QQ) plots comparing the distribution of association p-values from FinnGen R12 GWAS (REGENIE) between ClinVar-classified pathogenic and benign variant groups. All variants are restricted to those with a ClinVar classification of Pathogenic (P), Likely pathogenic (LP), Benign (B), or Likely benign (LB), as annotated by EVEE.

**(Left) Per-test QQ.** Each point represents one variant–phenotype association test. Pathogenic/LP variants (yellow, n_path shown in title) are compared against two allele frequency–matched benign control sets: Benign/LB (teal) and strict Benign-only (dark purple), each matched to the pathogenic allele frequency distribution across 30 log-spaced MAF bins. The genomic inflation factor (λ_GC) is shown in the legend for both AF-matched and unmatched (full) distributions.

**(Right) Per-variant min(p) QQ.** Each point represents a single variant, summarised by its minimum p-value across all FinnGen R12 phenotypes, Bonferroni-corrected for K phenotypes (1 − (1 − min p)^K). AF-matched controls are applied identically to the per-test panel.

Diagonal dashed line indicates the expected null distribution. Y-axis uses a symlog scale (linear below −log₁₀(p) = 30, logarithmic above). Colors follow the viridis colormap sampled at 0.05 (dark purple, strict benign), 0.55 (teal, benign/LB complement), and 0.95 (yellow, pathogenic/LP).

---

## Methods

### Data Sources

FinnGen R12 GWAS summary statistics (REGENIE, ~500,000 Finnish participants, 2,489 disease endpoints) were obtained from the FinnGen Data Freeze R12 public release. Per-variant EVEE pathogenicity scores and ClinVar classification labels (`eevee_label`) were pre-joined for each variant–phenotype pair and stored in per-phenotype `.eevee.tsv.gz` files.

### ClinVar Variant Subset

Analyses were restricted to variants with a ClinVar germline classification of Pathogenic (P), Likely pathogenic (LP), Benign (B), or Likely benign (LB) as recorded in the `eevee_label` field. Variants with Uncertain Significance (VUS) or no ClinVar entry were excluded.

Three groups were defined:
- **Pathogenic/LP**: `eevee_label ∈ {pathogenic, likely_pathogenic}`
- **Benign/LB complement**: `eevee_label ∈ {benign, likely_benign}`
- **Strict benign**: `eevee_label = benign` (ClinVar Benign only, excluding LB)

### Allele Frequency Matching

To control for the known confounding effect of allele frequency on GWAS power, benign control variants were matched to the pathogenic variant allele frequency (MAF = min(AF, 1 − AF)) distribution. MAF was divided into 30 logarithmically-spaced bins spanning [10⁻⁵, 0.5]. Within each bin, benign variants were randomly sampled (with replacement if necessary) to match the count of pathogenic variants in that bin (seed = 42). This procedure was applied separately to per-test and per-variant analyses.

### Per-test QQ Analysis

All variant–phenotype association rows passing filters (pval > 0, af_alt > 0) were collected. Expected p-values were computed as −log₁₀((rank − 0.5) / N) for each group independently after AF matching. The genomic inflation factor λ_GC was computed as the ratio of the observed median chi-squared statistic to the expected median under the null (0.455).

### Per-variant QQ Analysis

For each variant, the minimum p-value across all FinnGen R12 phenotypes was computed. A Bonferroni correction for multiple testing across K phenotypes was applied: p_corrected = 1 − (1 − p_min)^K. AF matching was applied to the per-variant distributions. λ_GC was computed on the Bonferroni-corrected p-values.

### Visualization

QQ plots were generated using matplotlib (Python). Points are rendered as semi-transparent scatter (α = 0.6, size = 4) with rasterization for efficiency. The y-axis uses a symmetric log (symlog) scale, switching from linear to log₁₀ above −log₁₀(p) = 30. Colors are sampled from the viridis colormap at positions 0.05 (#471365, strict benign), 0.55 (#1e9c89, benign/LB), and 0.95 (#dfe318, pathogenic/LP), matching the color scheme in the associated repository (yuj1r0/evee-finngen-r12). Figures were exported at 150 DPI (PNG) and as vector graphics (SVG).
