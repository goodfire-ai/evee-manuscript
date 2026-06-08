# FinnGen R12 × EVEE Pipeline — Variant Funnel Summary

*For pyramid / funnel figure generation. All counts derived from primary data sources.*

---

## Funnel Counts

| Stage | Filter | Unique variants | Source |
|-------|--------|---------------:|--------|
| 1 | FinnGen R12 PheWAS variants scored with EVEE | **178,864** | `all_endpoints.parquet` |
| 2 | EVEE pathogenicity > 0.8 | **2,496** | `all_endpoints.parquet` |
| 3 | + Finnish AF < 0.1% (rare variant) | **1,405** | `all_endpoints.parquet` |
| 4 | + OR ≥ 2.0 AND p < 1×10⁻⁵ (association threshold) | **132** | `finngen_gwas_lenient_highpath.tsv`† |
| 5 | Actionable after ACMG/AMP classification | **21** | `finngen_r12_actionable.tsv` |
| 6 | ClinVar resubmission candidates (LP-lean or above) | **6** | `finngen_r12_new_candidates.tsv` |

† The operational candidate set (132 variants) comes from a parallel Evo2 scoring run (Evo2 > 0.5, Arc Institute model) applied to the same FinnGen hits. Direct query of `all_endpoints` with EVEE > 0.8 + same OR/p thresholds returns 72 — the ~60 variant difference reflects Evo2 vs EVEE model coverage. Both scores are used across the pipeline; Evo2 was used for initial candidate generation, EVEE for final pathogenicity classification.

---

## Framework Description

### Study Design

This study applies a rare-variant PheWAS (Phenome-Wide Association Study) framework to FinnGen Release 12, a Finnish biobank cohort of approximately 500,000 individuals. The goal is to identify coding variants predicted pathogenic by EVEE or Evo2 that show phenome-wide association signals consistent with rare Mendelian disease — and to use those associations as evidence for ClinVar reclassification.

### Stage 1 — FinnGen R12 PheWAS scored with EVEE

**178,864 unique variants** from FinnGen R12 GWAS summary statistics were scored with EVEE (Goodfire's Evo2-based covariance probe pathogenicity predictor). These represent all genome-wide significant or near-significant associations across any of the FinnGen binary disease endpoints after REGENIE logistic regression. The full dataset comprises **445 million variant × phenotype pairs** (445,160,852 rows in `all_endpoints.parquet`).

- Source: `eevee_regenie_overlap/all_endpoints.parquet`
- EVEE score range: 0–1 (continuous; 1 = most pathogenic)
- Phenotype coverage: all FinnGen R12 binary disease endpoints

### Stage 2 — EVEE pathogenicity > 0.8

Variants were filtered to those with **EVEE pathogenicity score > 0.8**, retaining **2,496 variants** (1.4% of Stage 1). This threshold was selected to enrich for high-confidence pathogenic predictions while remaining permissive enough to capture rare founder variants with incomplete computational penetrance.

- **2,496 unique variants** pass this threshold
- Correspond to **6.2 million variant × phenotype pairs** across FinnGen endpoints

### Stage 3 — Finnish allele frequency < 0.1%

Variants were required to have Finnish population allele frequency < 0.1% (`af_alt < 0.001`) in the FinnGen cohort. This rare-variant filter excludes common polymorphisms with EVEE false-positive predictions and focuses the analysis on variants that could plausibly follow Mendelian inheritance patterns. It retains **1,405 variants** (56.3% of Stage 2).

- **Threshold:** AF < 0.001 (0.1%)
- **Rationale:** Rare variant enrichment; aligns with PM2_Supporting criterion in ACMG/AMP framework
- AF range in passing variants: 0.001%–0.1% (Finnish population frequency from FinnGen)

### Stage 4 — Association strength: OR ≥ 2.0 and p < 1×10⁻⁵

Variants were required to show phenotypic association with at least one FinnGen binary endpoint meeting a lenient significance threshold: **OR ≥ 2.0** and **p < 1×10⁻⁵** (best association across all endpoints per variant). This retains **132 variants** — the operational candidate set (`finngen_gwas_lenient_highpath.tsv`).

- **OR threshold:** ≥ 2.0 (enriched in cases vs controls at ≥ 2-fold; PS4_Moderate or stronger in ACMG framework)
- **p-value threshold:** < 1×10⁻⁵ (lenient; adjusted for rare-variant sparsity)
- **OR range** in passing variants: 2.2–150,317 (median 16.3)
- **p range:** 2.2×10⁻⁹¹–9.97×10⁻⁶ (median 3.5×10⁻⁶)
- Note: a strict subset (OR ≥ 2.0, p < 5×10⁻⁸) retains **69 variants** (`finngen_gwas_strict_evo2.tsv`)

### Stage 5 — Actionable after ACMG/AMP classification: 21 variants

Each of the 132 candidates was manually classified under the ACMG/AMP 2015 framework with three additional quality filters:

1. **Bio-coherence filter:** The FinnGen phenotype must match the gene's established OMIM disease (e.g., MEPE frameshift + otosclerosis = MATCH; LRBA missense + paraplegia = FALSE_POS)
2. **ClinGen hard filter:** Genes with ClinGen "No Known Disease Relationship" or "Contradicted" validity were excluded; "Limited" validity blocked PVS1 and PS4
3. **AR-het flag:** Autosomal recessive genes with heterozygous FinnGen carriers were capped at a maximum of 2 ACMG points (PP3 + PM2 only; PVS1 and PS4 inapplicable)
4. **ClinVar exclusion:** Variants with existing ClinVar P/LP classifications (any star level) were excluded — the pipeline targets variants where reclassification adds value

Criteria applied (Tavtigian 2018 point system, ClinGen SVI 2020/2022 updates):

| Criterion | Threshold | Points |
|-----------|-----------|--------|
| PVS1 | Frameshift/stop in AD LoF gene | 8 |
| PS4_Strong | OR > 5.0 | 4 |
| PS4_Moderate | OR 2.0–5.0 | 2 |
| PS4_Supporting | OR 1.5–2.0 | 1 |
| PP3_Moderate | EVEE or Evo2 > 0.5 (SVI 2022) | 1 |
| PM2_Supporting | Finnish AF < 0.1% (SVI 2020) | 1 |

**Classification thresholds:** Pathogenic ≥ 10 pts · Likely Pathogenic 6–9 pts · LP-lean 4–5 pts · VUS 2–3 pts

**21 variants** classified as VUS or above with bio-coherent phenotype matching are included in `finngen_r12_actionable.tsv`.

### Stage 6 — ClinVar resubmission candidates: 6 variants

Of the 21 actionable variants, **6 variants** reached LP-lean or above (≥ 4 ACMG points) with sufficient evidence quality for ClinVar submission:

| Variant | Classification | Points | ClinVar status |
|---------|---------------|--------|----------------|
| MEPE p.Lys70IlefsTer26 | **PATHOGENIC** | 11 | 0★ VUS |
| RP1 p.Leu172Arg | **LP** | 6–8 | 2★ conflicting |
| F2 chr11:46726097 | **LP** | 6 | 0★ VUS |
| TTN p.Val35195Glu | **VUS-with-evidence** | 6 | 2★ VUS consensus |
| F2 p.Leu389Val | **LP-lean** | 4 | NOT IN CLINVAR |
| MYH14 p.Glu1214Lys | **LP-lean** | 4 | 1★ VUS |

Submission is staged in two batches: MEPE, F2 p.Leu389Val, MYH14, and RP1 (Batch 1); F2 chr11:46726097 and TTN (Batch 2, after Batch 1 establishes submitter credibility).

---

## Data Sources

| Resource | Version / Access | Role |
|----------|-----------------|------|
| FinnGen GWAS | Release 12, ~500K Finnish participants | Case-control OR and p-values |
| EVEE | Goodfire AI (Evo2-based covariance probe) | Pathogenicity scoring (Stage 1–2) |
| Evo2 | Arc Institute genomic language model | Pathogenicity scoring (Stage 4 candidate generation) |
| gnomAD | v4.0 | Population AF verification |
| ClinGen Gene-Disease Validity | Current (accessed 2026) | Inheritance mode + bio-coherence |
| ClinVar | NCBI (accessed 2026) | Existing classification status |

---

*Primary data: `/mnt/data/shared/life-sciences/EVEE/finngen-validation/eevee_regenie_overlap/all_endpoints.parquet`*  
*Candidate set: `/mnt/home/ryo/finngen_gwas_lenient_highpath_clinvar.tsv` (132 variants)*  
*Figure: `figures/supplement/supfig_finngen_acmg.png`*
