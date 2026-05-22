# Figure Legends and Methods
## EVEE RNA-seq Validation — Nature Genetics Resubmission

---

## Figure: BRCA1 rs80358099 splice-acceptor prediction validated by RNA-seq
**File:** `figure_brca1_rna_validation.png`

EVEE predicts disruption of the canonical splice acceptor at the intron 21 / exon 22
boundary of BRCA1 (rs80358099, c.5278-1G>A, pathogenicity = 0.999). Per-position
annotation disruption tracks (top, signed Δ = var − ref) show a large gain in
`Is_Splice_Acceptor` (Δ = +0.54) and `Intron_Region` (Δ = +0.63) at −8 bp from the
canonical acceptor, alongside loss of `Polypyrimidine_Tract` (Δ = −0.47) and
`Exon→Intron_Boundary` (Δ = −0.48), consistent with cryptic acceptor activation 8 bp
into exon 22. RNA-seq sashimi plots (bottom) from three CCLE ovarian cancer cell lines
confirm the prediction: JHOS-4 (BRCA1 c.5278-1G>A carrier, VAF = 0.96 by Sanger CMP
WES) shows complete loss of canonical exon 22 inclusion (canonical reads = 0), with
reads split approximately equally between the cryptic +8 acceptor junction (red arcs,
max 58 reads) and full exon-22 skip (gold arcs). Sister cell line JHOS-2 and unrelated
control A2780 show canonical splicing with continuous exon 22 coverage (blue arcs,
max 192 and 294 reads respectively). The 8 bp truncated region of exon 22 (chr17:
43,051,109–43,051,116, hatched) is absent from JHOS-4 coverage, confirming the
cryptic +8 acceptor usage predicted by EVEE. JHOS-4 carries the variant at near-
homozygous VAF (Broad WES: 0.962; Sanger WES: 1.000), consistent with germline
heterozygosity followed by somatic loss of heterozygosity — a Knudson two-hit
pattern expected for a BRCA1 pathogenic variant in ovarian cancer. Exon coordinates
from GENCODE v49, transcript ENST00000357654.9.

---

## Figure: EVEE splice + branchpoint variant validation — VAF-corrected disruption
**File:** `figure_splice_vaf_psi.png`

Each point is one splice or branchpoint variant × carrier cell line observation from the
Sanger Cell Model Passports (Sanger CMP), restricted to EVEE pathogenicity ≥ 0.99,
ClinVar ≥ 2 stars, ≥ 5 carrier junction reads, and no tissue mismatch (n = 66 observations:
20 Hom/LOH, 46 Het). X-axis: VAF (allele fraction in carrier, from Sanger CMP WES).
Y-axis: 1 − PSI(canonical), where PSI(canonical) is the fraction of junction reads
supporting canonical exon inclusion; 0 = no splice disruption, 1 = complete splice loss.
Red squares: homozygous/LOH carriers (VAF ≥ 0.80); blue circles: heterozygous carriers
(VAF 0.40–0.80). Dashed diagonal: expected 1 − PSI under the model that EVEE predicts
complete disruption on the mutant allele (PSI_mutant = 0), so 1 − PSI_obs = VAF.
Spearman ρ = 0.60 (p = 1.3 × 10⁻⁷) across all 66 observations. Hom/LOH carriers
cluster near 1 − PSI ≈ 1 (complete splice loss) as expected; heterozygous carriers span
a range consistent with partial disruption scaled by allele fraction. The eight
observations with 1 − PSI ≈ 0 (no detected disruption) all have ≥ 15 carrier reads,
indicating genuine splicing-neutral events in the tested cell lines rather than coverage
artefacts.

---

## Supplementary Figure: EVEE RNA-seq validation headline summary
**File:** `supplement_combined_vaf_headline.png`

Combined validation summary across splice (Tier 1: canonical splice acceptor/donor and
branchpoint variants) and NMD (Tier 3: nonsense and frameshift variants). Bar chart shows
the fraction of variants with strong mutant-allele disruption in each stratum, among
testable variants (VAF ≥ 0.40, ≥ 5 carrier reads). Splice validation threshold:
PSI_mutant < 0.30 (PSI_mutant = (PSI_obs − (1−VAF)) / VAF). NMD validation threshold:
expr_mutant < 0.50 (expr_mutant = carrier junction reads / CCLE median, VAF-corrected).
Results: Splice Hom/LOH 14/14 = 100%; Splice Het 16/33 = 48%; Splice All 30/47 = 64%;
NMD All 14/32 = 44%; All Combined 44/79 = 56%. NMD signal in cancer cell lines is
attenuated due to frequent NMD pathway suppression in tumors and the modest (~25%)
expected expression reduction in heterozygous carriers; these rates are conservative
lower bounds.

---

## Methods

### EVEE variant scoring
Variants were scored using EVEE (Evolutionary and Epigenomic Variant Effect predictor),
which combines Evo2 language model embeddings with annotation head probes trained to
predict 366 structural and functional genomic features. Pathogenicity scores and
per-position annotation disruption profiles (Δ = var_head − ref_head) were retrieved
from the EVEE variant database (variants.duckdb, build v5). All variants in the
validation set had EVEE pathogenicity ≥ 0.99 and ClinVar classification of pathogenic
or likely pathogenic with ≥ 1 star.

### Cell line genotype data
Variant–cell line pairs were identified using the Sanger Cell Model Passports (CMP)
public mutation catalogue (`mutations_all_latest.csv.gz`, downloaded May 2026 from
`cog.sanger.ac.uk/cmp/download/`). VAF values are from Sanger CMP WES pipelines
(Broad and/or Sanger). Variants were matched to EVEE candidates by genomic coordinates
(chromosome, position, reference allele, alternative allele; hg38). Only variants with
effect in `{ess_splice, splice_region, nonsense, frameshift}` were included. Zygosity
was defined as: homozygous/LOH (VAF ≥ 0.80), heterozygous (VAF 0.40–0.80), subclonal
(VAF < 0.40).

### RNA-seq splice junction quantification
Junction read counts were retrieved from the Snaptron web service
(`snaptron.cs.jhu.edu`) querying the SRAv3 (srav3h) and CCLE compilations of the
recount3 resource. For each variant, the chromosomal region spanning ±50 kb around
the affected exon was queried. Canonical junction PSI was computed as:
PSI(canonical) = canonical_reads / (canonical_reads + aberrant_reads), where aberrant
reads include the predicted cryptic acceptor/donor junction and full exon-skip junction.
Background PSI was estimated from the CCLE compilation (cell lines without the variant).
A validation was called "strong" when: PSI(canonical) < 0.20 AND aberrant_reads ≥ 3 AND
≥ 1 replicate sample.

### VAF-corrected PSI
To estimate the disruption specifically on the mutant allele, PSI on the mutant allele
(PSI_mutant) was estimated as:

    PSI_mutant = (PSI_obs − (1 − VAF)) / VAF

This model assumes the wild-type allele always contributes canonical splice junctions
(PSI_WT = 1.0). PSI_mutant is clipped to [0, 1]. A threshold of PSI_mutant < 0.30 was
used to call splice validation.

### BRCA1 rs80358099 case study
JHOS-4 (CVCL_4649, DepMap ACH-000584) was identified as carrying BRCA1 c.5278-1G>A
(rs80358099) at near-homozygous VAF (Broad WES: 0.962; Sanger WES: 1.000) from the
Sanger CMP database (model_id SIDM00303), independently confirmed as `ess_splice` and
`cancer_predisposition_variant: TRUE`. RNA-seq data from 5 independent SRA runs
(1 from CCLE 2019, PRJNA523380; 4 from Watanabe et al. 2016, PAX8 cistrome study) all
show PSI(canonical exon 22 inclusion) = 0, with reads split between cryptic +8 acceptor
and full exon-22 skip junctions. Control cell lines JHOS-2 (Sanger sister line, BRCA1
wild-type by CMP) and A2780 (BRCA1 wild-type ovarian cancer line) show canonical
splicing. EVEE annotation disruption tracks were rendered using per-position Δ profiles
from the EVEE token scores database, with exon coordinates from GENCODE v49
(ENST00000357654.9). Introns are compressed to 28 px for display. Junction arcs are
scaled by square root of read count (max arc width = 15 px). The figure was generated
using the `figure_brca1_browser_v8.py` script in the EVEE validation repository.

### NMD validation (Supplementary)
For nonsense and frameshift variants (Tier 3), gene-level expression in carrier cell
lines was compared to the CCLE background distribution using two metrics: (1) junction
read sum from Snaptron as a proxy for transcript abundance, and (2) log2(TPM+1) from
the DepMap 23Q4 public expression matrix (OmicsExpressionProteinCodingGenesTPMLogp1.csv,
figshare:43347204). Robust Z-scores were computed as (carrier_value − median) /
(1.4826 × MAD) across all cell lines with detectable expression. Neither metric showed
a significant dosage-response between VAF and expression depletion (Pearson r < 0.2),
consistent with frequent NMD pathway suppression in cancer cell lines and the modest
expected expression reduction (~25%) in heterozygous carriers.
