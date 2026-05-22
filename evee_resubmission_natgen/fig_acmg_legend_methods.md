# Figure: ACMG/AMP Evidence Points for FinnGen R12 EVEE Candidates

## Figure Legend

**ACMG/AMP evidence scores for FinnGen R12 rare variant candidates identified by the EVEE
pathogenicity pipeline.**
Each bar represents one variant, stacked by contributing evidence criterion
(colors, see key). Dashed horizontal lines mark classification thresholds under the
Tavtigian 2018 point-weighting system: LP-lean (≥ 4 points), Likely Pathogenic (LP, ≥ 6),
and Pathogenic (P, ≥ 10). Stars (★) denote the six variants recommended for ClinVar
submission. Bars are sorted by total evidence score, with submission candidates shown
first (left block, separated by dotted line), followed by remaining actionable variants
with insufficient evidence for reclassification (right block).

**Criterion colors:**

| Color | Criterion | Evidence type |
|-------|-----------|---------------|
| Red (PVS1) | Null variant (frameshift / stop-gained) | Loss-of-function in a gene where LoF is established disease mechanism |
| Blue (PS4) | Case-control odds ratio from FinnGen R12 | OR > 5 = Strong (+4 pts); OR 2–5 = Moderate (+2 pts); OR 1.5–2 = Supporting (+1 pt) |
| Orange (PP3) | EVEE/Evo2 computational pathogenicity score | Score > 0.5 = PP3\_Moderate (+1 pt per SVI 2022 calibration) |
| Green (PM2) | Finnish population allele frequency | AF < 0.1% = PM2\_Supporting (+1 pt per ClinGen SVI 2020) |

All 21 variants shown passed the initial actionable filter (ACMG\_computed ∈ {PATHOGENIC,
LP, LP-lean, VUS} with bio-coherent phenotype–gene pairing or clear high-evidence status).

---

## Methods

### Variant Discovery

Rare variant candidates were identified from the FinnGen R12 GWAS summary statistics
(release 12, N ≈ 500,000 Finnish biobank participants) using a lenient discovery filter:
genome-wide significance (p < 5 × 10⁻⁸ across any binary disease endpoint), minor allele
frequency < 0.5% in the Finnish population, and EVEE pathogenicity score > 0.6 or Evo2
pathogenicity score > 0.5. Variants with active ClinVar Pathogenic/Likely Pathogenic
classifications (any star level) were excluded, as the pipeline targets variants where
reclassification adds value.

### Pathogenicity Scoring

Variants were scored under a structured ACMG/AMP framework incorporating three sequential
filters and four quantitative criteria:

**Three-fix quality filters (applied before scoring):**

1. *Bio-coherence*: The FinnGen phenotype must map to a disease plausibly caused by the
   implicated gene under its established OMIM condition (e.g., MEPE frameshift + otosclerosis
   = MATCH; LRBA missense + paraplegia = FALSE\_POS). Variants with bio-coherence = FALSE\_POS
   were excluded from resubmission consideration.

2. *ClinGen hard filter*: Variants in genes with ClinGen gene-disease validity of
   "No Known Disease Relationship" or "Contradicted" were excluded. ClinGen designation of
   "Limited" blocked PS4 and PVS1 from being applied.

3. *AR-het flag*: For autosomal recessive (AR) genes, heterozygous FinnGen carriers cannot
   satisfy PVS1 (heterozygous LoF is not disease-causing) or PS4 in the standard sense
   (case enrichment may reflect cryptic biallelic cases rather than pathogenic heterozygosity).
   AR-het variants were assigned a maximum achievable score of 2 points (PP3 + PM2 only).

**Criteria and point weights (Tavtigian 2018 / ClinGen SVI updates):**

| Criterion | Application rule | Points |
|-----------|-----------------|--------|
| PVS1 | Frameshift or stop-gained in AD gene with established LoF mechanism; full strength if predicted NMD | 8 |
| PS4\_Strong | FinnGen OR > 5 | 4 |
| PS4\_Moderate | FinnGen OR 2–5 | 2 |
| PS4\_Supporting | FinnGen OR 1.5–2 | 1 |
| PP3\_Moderate | EVEE > 0.5 or Evo2 > 0.5 (per ClinGen SVI 2022 calibrated computational evidence) | 1 |
| PM2\_Supporting | Finnish AF < 0.001 (per ClinGen SVI 2020 PM2 update) | 1 |

**Classification thresholds:**
≥ 10 pts = Pathogenic; 6–9 pts = Likely Pathogenic (LP); 4–5 pts = LP-lean (borderline LP);
2–3 pts = VUS; 0–1 pts = Benign/VUS.

### Data Sources

- **FinnGen R12**: Genome-wide association summary statistics, Finnish biobank cohort
  (≈ 500,000 individuals); REGENIE logistic regression; binary disease endpoints (ICD-10 blocks).
  Accessed via the FinnGen R12 EVEE overlap parquet files.
- **EVEE**: Goodfire AI variant pathogenicity predictor (Evo2-based covariance probe);
  scores range 0–1 (1 = most pathogenic). EVEE scores were used for the PP3 criterion.
- **Evo2**: Arc Institute genomic language model pathogenicity scores (where EVEE scores
  were unavailable); applied identically to EVEE for PP3 thresholding.
- **ClinVar** (NCBI): Existing variant classifications and review status retrieved
  programmatically to identify upgrade opportunities (0–2 star VUS or absent records).
- **ClinGen Gene-Disease Validity**: Used to determine inheritance mode and disease
  validity score for bio-coherence and hard-filter application.
- **gnomAD v4.0**: gnomAD population allele frequencies used for PM2 (Finnish AF < 0.1%
  threshold) and to verify absence from non-Finnish populations.

### ClinVar Submission Strategy

Six variants met criteria for ClinVar submission. The submission strategy was tiered:

**Batch 1** (four variants): MEPE p.Lys70IlefsTer26 (Pathogenic, 11 pts), F2 p.Leu389Val
(LP-lean, novel addition), MYH14 p.Glu1214Lys (LP-lean, upgrades 1★ VUS), and
RP1 p.Leu172Arg (LP for autosomal recessive RP1, conflict resolution). These four
represent the clearest evidence packages and the least submission risk.

**Batch 2** (two variants, after Batch 1 lands): F2 chr11:46726097 (LP, upgrades 0★ VUS),
submitted as a second F2 variant after the first F2 establishes the gene-level evidence;
and TTN p.Val35195Glu (submitted as VUS-with-evidence rather than LP, to respect existing
2★ multi-submitter VUS consensus while depositing exceptional FinnGen statistical evidence:
OR = 119.6, p = 5.65 × 10⁻⁸⁶).

All submissions must include explicit inheritance mode framing (especially RP1 = AR, not AD),
specific OMIM-coded condition terms (not generic "inborn genetic diseases"), and Mayo Clinic
co-authorship with clinical oversight per institutional guidelines.
