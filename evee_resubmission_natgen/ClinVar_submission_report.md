# ClinVar Resubmission Report — FinnGen R12 × EVEE Pipeline Candidates

**Project:** EVEE (Goodfire AI) × FinnGen R12 PheWAS rare-variant validation  
**Prepared:** 2026-05-27  
**Classification framework:** ACMG/AMP 2015 + ClinGen SVI 2020 (PM2) + SVI 2022 (PP3) + Tavtigian 2018 point system  
**Point thresholds:** Pathogenic ≥ 10 · Likely Pathogenic 6–9 · LP-lean 4–5 · VUS 2–3  
**External review:** Each variant independently reviewed and documented in `Clinvar_reclass.zip` (generated via structured clinical review, 2026-05-22)  

---

## Submission Strategy

| Batch | Variants | Rationale |
|-------|----------|-----------|
| **Batch 1** | MEPE, F2 p.Leu389Val, MYH14, RP1 | Strongest/cleanest cases; establish submitter credibility |
| **Batch 2** | F2 chr11:46726097, TTN | Hold pending Batch 1 landing; require case-count verification (F2) and credibility as established submitter (TTN) |

---

## Variant 1 — MEPE p.Lys70IlefsTer26 · LEAD SUBMISSION

### Identification

| Field | Value |
|-------|-------|
| Gene | *MEPE* (Matrix extracellular phosphoglycoprotein; HGNC:13361) |
| Genomic (GRCh38) | chr4:87845065:GGAAA:G (0-based VCF) |
| HGVS coding | NM_005166.4:c.209_212del *(confirm with VEP against MANE Select)* |
| HGVS protein | p.Lys70IlefsTer26 *(predicted; verify with VEP)* |
| Consequence | Frameshift → premature stop at residue 95/525 |
| dbSNP | rs753138805 |
| ClinVar existing | VCV003358411 · **0★ VUS** · single submitter (Illumina 2018) · "MEPE-related disorder" |

### Disease

| Field | Value |
|-------|-------|
| Condition | Otosclerosis (OMIM:166800); also associated with fracture susceptibility |
| Inheritance | Autosomal dominant with incomplete/age-dependent penetrance |
| Gene-disease validity | Strong/Definitive — established Schrauwen 2018 (PMID:30245510) + replicated Ramo 2023 (PMID:36670102) |
| Mechanism | MEPE encodes bone matrix phosphoglycoprotein; frameshift at residue 70 abolishes the ASARM peptide (C-terminal mineralization domain) required for PHEX binding and phosphate homeostasis → dysregulated bone remodeling → otosclerosis + decreased bone mineral density |

### ACMG/AMP Evidence

| Criterion | Strength | Points | Basis |
|-----------|----------|--------|-------|
| PVS1 | Very Strong | 8 | Frameshift in AD LoF gene; NMD predicted (verify: >50 nt upstream of last exon-exon junction); LoF mechanism established by Schrauwen 2018 |
| PS4 | Moderate | 2 | FinnGen R12: OR = 3.47, p = 8.22 × 10⁻¹⁷, otosclerosis endpoint (H8_OTOSCLE). *(Meta-GWAS Ramo 2023 reports OR = 21.5, 95% CI 9.6–48.4 across FinnGen + EstBB + UKB — cite for PS4_Strong upgrade to +4 if using published OR)* |
| PP3 | Moderate | 1 | EVEE pathogenicity = 0.970 (> 0.5 threshold; SVI 2022 PP3_Moderate) |
| PM2 | Not applied | 0 | Finnish AF = 0.32% — above 0.1% threshold; Finnish founder enrichment documented (cases AF 1.1% vs controls AF 0.3%) |
| **Total** | | **11** | **PATHOGENIC** (threshold ≥ 10) |

> **Note on PS4:** Using FinnGen-only OR = 3.47 gives PS4_Moderate (+2). The published meta-GWAS (Ramo 2023 Nat Commun, PMID:36670102) reports OR = 21.5 across 3,504 cases and 861,198 controls — this supports PS4_Strong (+4), pushing total to 13 pts. Pull Supplementary Table from PMID:36670102 for the exact OR/CI for rs753138805 before submission.

### FinnGen R12 Association

| Endpoint | OR | p-value | Finnish AF |
|----------|----|---------|------------|
| H8_OTOSCLE (otosclerosis) | 3.47 | 8.22 × 10⁻¹⁷ | 0.32% |
| H8_HL_CON_NAS (conductive hearing loss) | — | — | — |
| ST19_FRACT_FOREA (forearm fracture) | — | — | — |
| ST19_FRACT_LOWER_LEG (lower leg fracture) | — | — | — |

Multi-phenotype association (n = 4 endpoints) is consistent with MEPE's role in bone mineralization.

### Key Literature

- **PMID:30245510** — Schrauwen et al. 2018, *Genetics in Medicine*: MEPE LoF → familial otosclerosis; establishes AD gene-disease relationship
- **PMID:36670102** — Ramo et al. 2023, *Nature Communications*: Meta-GWAS across FinnGen/EstBB/UKB confirming MEPE-otosclerosis association; OR = 21.5

### ClinVar Submission

- **Recommended classification:** Likely Pathogenic (conservative) or Pathogenic (if PVS1 verified at Very Strong)
- **Condition:** Otosclerosis — OMIM:166800 *(NOT "MEPE-related disorder" — specificity is the upgrade value)*
- **Inheritance:** Autosomal dominant
- **Value add:** Replaces 2018 automated pre-literature VUS with structured ACMG/AMP classification citing landmark 2018 + 2023 papers

### Pre-Submission Checklist

- [ ] Run VEP on chr4:87845065:GGAAA:G against MANE Select NM_005166.4 — confirm HGVS and exon position
- [ ] Verify NMD prediction: is frameshift > 50 nt upstream of last exon-exon junction? Determines PVS1_VeryStrong vs PVS1_Strong
- [ ] Map truncation relative to ASARM motif (C-terminal, ~aa 500–525) — early truncation at aa 70 vs late truncation has different mechanism weight
- [ ] Pull Ramo 2023 Supplementary Table for rs753138805 exact OR/CI to confirm PS4_Strong eligibility
- [ ] Confirm NM_005166.4 is MANE Select for MEPE (versus NM_020203.4 — the external review used a different RefSeq accession)

---

## Variant 2 — F2 p.Leu389Val · NOVEL SUBMISSION

### Identification

| Field | Value |
|-------|-------|
| Gene | *F2* (Coagulation factor II / Prothrombin; HGNC:3535) |
| Genomic (GRCh38) | chr11:46728029:C:G |
| HGVS coding | NM_000506.5:c.1165C>G |
| HGVS protein | p.Leu389Val |
| Consequence | Missense — leucine to valine in catalytic serine protease domain |
| ClinVar existing | **NOT IN CLINVAR** — confirmed by exhaustive search (gene + protein + coordinates + HGVS all return zero results) |

### Disease

| Field | Value |
|-------|-------|
| Condition | Thrombophilia due to thrombin defect (OMIM:188050) / Prothrombin deficiency (OMIM:613679) |
| Inheritance | Autosomal dominant |
| Gene-disease validity | Definitive — F2/prothrombin is canonical AD thrombophilia gene |
| Mechanism | Leu389 is in the serine protease catalytic domain; near known pathogenic residues R382C (OMIM:176930.0001) and G558V (OMIM:176930.0005) |

### ACMG/AMP Evidence

| Criterion | Strength | Points | Basis |
|-----------|----------|--------|-------|
| PS4 | Moderate | 2 | FinnGen R12: OR = 2.76, p = 1.9 × 10⁻⁸, phlebitis/thrombophlebitis/DVT (I9_PHLETHROMBDVTLOW) |
| PP3 | Moderate | 1 | EVEE pathogenicity = 0.908 (SVI 2022 PP3_Moderate) |
| PM2 | Supporting | 1 | Finnish AF = 0.12%; gnomAD overall AF = 3.29 × 10⁻⁵ (ultra-rare outside Finland) |
| PM1 | Supporting | +1 (optional) | Catalytic domain location; near known pathogenic residues — apply if structural domain mapping confirms |
| **Total** | | **4 (–5 with PM1)** | **LP-lean** (threshold ≥ 4) |

### FinnGen R12 Association

| Endpoint | OR | p-value | Finnish AF |
|----------|----|---------|------------|
| I9_PHLETHROMBDVTLOW (phlebitis/DVT lower extremity) | 2.76 | 1.9 × 10⁻⁸ | 0.12% |

### Key Context

This is a **completely novel ClinVar entry** — no other laboratory has deposited this variant. F2 is a canonical AD thrombophilia gene with well-established gene-disease validity. The FinnGen evidence provides case-control data that future curators encountering this variant in clinical settings will benefit from.

### ClinVar Submission

- **Recommended classification:** Likely Pathogenic (lean) or VUS-with-evidence
- **Condition:** Thrombophilia due to thrombin defect — OMIM:188050
- **Inheritance:** Autosomal dominant
- **Value add:** Novel addition to ClinVar; deposits population-scale Finnish evidence for a variant currently invisible to clinical labs

### Pre-Submission Checklist

- [ ] Confirm HGVS against MANE Select NM_000506.5
- [ ] Map Leu389 to F2 protein domain (UniProt P00734 feature table) — confirm catalytic domain for PM1
- [ ] Verify gnomAD v4 popmax AF excluding FIN population
- [ ] Confirm absolute case count in FinnGen for I9_PHLETHROMBDVTLOW (OR = 2.76 at p = 1.9 × 10⁻⁸ implies reasonable N but verify)
- [ ] Cite nearby pathogenic residues R382C and G558V (OMIM allelic variants) for PM1 context

---

## Variant 3 — MYH14 p.Glu1214Lys · RECLASSIFICATION (1★ VUS → LP-lean)

### Identification

| Field | Value |
|-------|-------|
| Gene | *MYH14* (Myosin heavy chain 14; HGNC:23212) |
| Genomic (GRCh38) | chr19:50276162:G:A |
| HGVS coding | NM_024729.4:c.3640G>A *(confirm with VEP)* |
| HGVS protein | p.Glu1214Lys — glutamate to lysine, charge reversal |
| Consequence | Missense in myosin motor domain |
| ClinVar existing | VCV003297404 · **1★ VUS** · single submitter (Ambry Genetics) · "Inborn genetic diseases (hearing loss)" |

### Disease

| Field | Value |
|-------|-------|
| Condition | Deafness, autosomal dominant 4A (DFNA4A; OMIM:600652) |
| Inheritance | Autosomal dominant |
| Gene-disease validity | Definitive — MYH14-DFNA4 well-established |
| VCEP coverage | **Not covered** by ClinGen Hearing Loss VCEP (covers CDH23, COCH, GJB2, KCNQ4, MYO6, MYO7A, SLC26A4, TECTA, USH2A, OTOF, MYO15A — not MYH14); submit under general ACMG/AMP 2015 + SVI updates |

### ACMG/AMP Evidence

| Criterion | Strength | Points | Basis |
|-----------|----------|--------|-------|
| PS4 | Moderate | 2 | FinnGen R12: OR = 4.4, p = 5.64 × 10⁻²⁸, sensorineural hearing loss (H8_HL_SEN_NAS) |
| PP3 | Moderate | 1 | EVEE/Evo2 pathogenicity = 0.838 (SVI 2022 PP3_Moderate) |
| PM2 | Supporting | 1 | Finnish AF = 0.044%; gnomAD overall = not detected (ultra-rare) |
| PM1 | Supporting | +1 (optional) | Myosin motor domain location — apply if UniProt Q7Z406 mapping confirms |
| **Total** | | **4 (–5 with PM1)** | **LP-lean** |

### FinnGen R12 Association

| Endpoint | OR | p-value | Finnish AF |
|----------|----|---------|------------|
| H8_HL_SEN_NAS (sensorineural hearing loss, non-age-related) | 4.4 | 5.64 × 10⁻²⁸ | 0.044% |

The exceptional statistical significance (p = 5.64 × 10⁻²⁸) with a biologically coherent endpoint (sensorineural hearing loss matching DFNA4A) provides strong PS4_Moderate justification.

### ClinVar Submission

- **Recommended classification:** Likely Pathogenic (lean) for DFNA4A
- **Condition:** Deafness, autosomal dominant 4A — OMIM:600652 *(more specific than Ambry's "inborn genetic diseases")*
- **Inheritance:** Autosomal dominant
- **Value add:** Upgrades 1★ single-submitter VUS to 2★ multi-submitter aggregate; specific OMIM-coded condition adds clinical utility

### Pre-Submission Checklist

- [ ] Confirm HGVS against MANE Select NM_024729.4
- [ ] Map Glu1214 to myosin motor domain (UniProt Q7Z406 feature table) — confirm domain for PM1
- [ ] Review existing Ambry Genetics VUS reasoning in VCV003297404 — address their evidence in submission narrative
- [ ] Cite Donaudy 2004 (PMID:14684075) and other MYH14-DFNA4A establishing papers
- [ ] Confirm H8_HL_SEN_NAS is specifically sensorineural (non-age-related) in FinnGen endpoint definitions — strengthens PP4 if specific phenotype match

---

## Variant 4 — RP1 p.Leu172Arg · CONFLICT RESOLUTION (2★ conflicting → LP for AR-RP1)

### Identification

| Field | Value |
|-------|-------|
| Gene | *RP1* (Retinitis pigmentosa 1; HGNC:10263) |
| Genomic (GRCh38) | chr8:54621480:T:G |
| HGVS coding | NM_006269.2:c.515T>G |
| HGVS protein | p.Leu172Arg — leucine to arginine in DCX2 domain (aa 154–233) |
| Consequence | Missense in doublecortin domain 2 (DCX2) |
| dbSNP | rs180729424 |
| ClinVar existing | VCV000195261 · **2★ conflicting** · 4 submitters: 2× Pathogenic (Invitae Feb 2026, Blueprint 2019), 2× VUS (Illumina 2018, Eurofins 2015) |

### Disease

| Field | Value |
|-------|-------|
| Condition | Retinitis Pigmentosa 1, autosomal recessive (arRP1; OMIM:180100) |
| Inheritance | **Autosomal recessive** — critical framing (see warning below) |
| Gene-disease validity | Definitive (ClinGen) |
| Regional partition | Per Zou 2021 (PMID:33681214): N-terminal variants (aa 1–613, including DCX2 at 154–233) cause AR-RP when biallelic; heterozygous N-terminal variants are typically NOT pathogenic for AD-RP |

### ⚠️ Critical Warning

**This variant MUST be submitted as LP/P for autosomal recessive RP1, NOT autosomal dominant RP1.** Position 172 falls in the N-terminal DCX2 domain (aa 154–233). Per Zou 2021, this region is AR-causing. Invitae's Feb 2026 update (SCV001533777.6) found the variant in trans with another pathogenic variant in an affected individual — the gold-standard AR confirmation. Submitting as AD would be clinically misleading and inconsistent with current expert consensus.

### ACMG/AMP Evidence

| Criterion | Strength | Points | Basis |
|-----------|----------|--------|-------|
| PS4 | Moderate | 2 | FinnGen R12: OR = 17.0, p = 2.99 × 10⁻¹⁰, hereditary retinal dystrophy (H7_HEREDRETINADYST). Downgraded from Strong: heterozygous FinnGen association for AR disease may reflect cryptic biallelic cases or founder haplotype enrichment |
| PM3 | Moderate | 2 | Invitae (SCV001533777.6, Feb 2026): variant detected in trans with pathogenic variant in affected individual — direct AR confirmation |
| PM1 | Moderate | 2 | DCX2 domain location (aa 154–233; PMID:8197273) — critical microtubule-binding domain |
| PP3 | Moderate | 1 | EVEE pathogenicity = 0.906; PolyPhen-2 disruptive per Invitae |
| PM2 | Supporting | 1 | Finnish AF = 0.064%; gnomAD overall AF = 6.3 × 10⁻⁵ |
| **Total** | | **8** | **Likely Pathogenic** (threshold ≥ 6) |

> **Note:** Our pipeline TSV computed 6 pts (PS4_strong + PP3 + PM2 only). The external review added PM3_Moderate (+2) from the Invitae in-trans observation and PM1_Moderate (+2) for DCX2 domain, raising the score to 8–10 pts. The 6-point pipeline score is conservative; the 8-point external review score is more complete.

### Current ClinVar Submitters

| Submitter | Date | Classification | Notes |
|-----------|------|----------------|-------|
| Labcorp/Invitae | Feb 2026 | **Pathogenic** | In-trans with pathogenic variant in affected AR-IRD individual; most current, most rigorous |
| Blueprint Genetics | 2019 | **Pathogenic** | My Retina Tracker patient observation |
| Illumina | 2018 | VUS | Automated screening; pre-AR characterization (2018 predates Zou 2021) |
| Eurofins NTD | 2015 | VUS | 2 heterozygous observations; pre-AR characterization (2015 predates Zou 2021) |

### FinnGen R12 Association

| Endpoint | OR | p-value | Finnish AF |
|----------|----|---------|------------|
| H7_HEREDRETINADYST (hereditary retinal dystrophy) | 17.0 | 2.99 × 10⁻¹⁰ | 0.064% |

### ClinVar Submission

- **Recommended classification:** Likely Pathogenic for autosomal recessive Retinitis Pigmentosa 1
- **Condition:** Retinitis Pigmentosa 1, autosomal recessive — OMIM:180100
- **Inheritance:** **Autosomal recessive** (explicit)
- **Value add:** Joins Invitae/Blueprint Pathogenic consensus; adds Finnish biobank population evidence; explicitly documents AR framing to resolve the 2★ conflict caused by pre-2021 VUS submissions

### Pre-Submission Checklist

- [ ] **Confirm AR framing** throughout submission — every field must say "autosomal recessive"
- [ ] Confirm DCX2 domain boundaries (aa 154–233) via PMID:8197273 for PM1 documentation
- [ ] Cite Zou 2021 (PMID:33681214) for N-terminal AR partition rationale — this is the key literature anchor
- [ ] Cite Audo 2012 (PMID:22334370) for AR-RP1 mechanism in N-terminal region
- [ ] Investigate FinnGen case zygosity if accessible — homozygotes or compound hets would strengthen PS4 framing for AR disease
- [ ] Pull PMID:28041643, 29068140, 32565670 cited by Invitae — verify variant context in these papers

---

## Variant 5 — F2 chr11:46726097 · WEAK RECORD UPGRADE

**Batch 2 — hold until F2 p.Leu389Val (Variant 2) is accepted first.**

### Identification

| Field | Value |
|-------|-------|
| Gene | *F2* (Coagulation factor II / Prothrombin; HGNC:3535) |
| Genomic (GRCh38) | chr11:46726097:G:T |
| HGVS coding | NM_000506.5:c.? *(missense — confirm with VEP; position not yet annotated)* |
| HGVS protein | p.? *(confirm with VEP)* |
| Consequence | Missense |
| ClinVar existing | VCV002582746 · **0★ VUS** · single submitter · no assertion criteria · "Thrombophilia due to thrombin defect" |

### Disease

Same as Variant 2: Thrombophilia / Prothrombin deficiency, AD, Definitive gene-disease validity.

### ACMG/AMP Evidence

| Criterion | Strength | Points | Basis |
|-----------|----------|--------|-------|
| PS4 | Strong | 4 | FinnGen R12: OR = 24.7, p = 1.44 × 10⁻⁶, coagulation disorders (D3_COAGOTHER) |
| PP3 | Moderate | 1 | Evo2 pathogenicity = 0.933 (SVI 2022 PP3_Moderate) |
| PM2 | Supporting | 1 | Finnish AF = 6 × 10⁻⁵; gnomAD = not detected |
| **Total** | | **6** | **Likely Pathogenic** |

> **Caveat on PS4:** OR = 24.7 with p = 1.44 × 10⁻⁶ suggests a very small absolute case count (Finnish AF = 6 × 10⁻⁵ means extremely few carriers). High OR from sparse data may reflect statistical inflation. Verify absolute case count before claiming PS4_Strong; may need downgrade to PS4_Moderate (+2) if N_cases < 10.

### FinnGen R12 Association

| Endpoint | OR | p-value | Finnish AF |
|----------|----|---------|------------|
| D3_COAGOTHER (other coagulation disorders) | 24.7 | 1.44 × 10⁻⁶ | 6 × 10⁻⁵ |

### ClinVar Submission

- **Recommended classification:** Likely Pathogenic (pending case-count verification)
- **Condition:** Thrombophilia due to thrombin defect — OMIM:188050
- **Inheritance:** Autosomal dominant
- **Value add:** Upgrades 0★ no-criteria VUS with structured ACMG/AMP evidence; completes the F2 picture alongside Variant 2

### Pre-Submission Checklist

- [ ] **Confirm HGVS** — run VEP on chr11:46726097:G:T against NM_000506.5 to get c.? and p.?
- [ ] Map protein position to F2 domain structure (UniProt P00734)
- [ ] Verify absolute case count in FinnGen D3_COAGOTHER endpoint — if very small, downgrade PS4_Strong → PS4_Moderate
- [ ] Confirm D3_COAGOTHER endpoint captures inherited coagulopathies (not primarily acquired) in FinnGen phenotype definitions
- [ ] **Wait for Variant 2 (F2 p.Leu389Val) to be accepted** before submitting — avoid "gene padding" optics in opening batch

---

## Variant 6 — TTN p.Val35195Glu · VUS-WITH-EVIDENCE DEPOSIT

**Batch 2 — submit ONLY as VUS-with-evidence, NOT as Likely Pathogenic.**

### Identification

| Field | Value |
|-------|-------|
| Gene | *TTN* (Titin; HGNC:12403) |
| Genomic (GRCh38) | chr2:178527204:A:T |
| HGVS coding | NM_001267550.2:c.105584A>T |
| HGVS protein | p.Val35195Glu |
| Consequence | Missense — valine to glutamate in M-band/A-band region (~aa 32000–35991) |
| ClinVar existing | VCV000281698 · **2★ multi-submitter, no conflicts** · **all submitters call VUS** · 6 conditions listed |

### Disease

| Field | Value |
|-------|-------|
| Condition candidates | Limb-girdle muscular dystrophy 2J (OMIM:608807) / Dilated cardiomyopathy 1G (OMIM:604145) / HMERF (OMIM:603689) |
| FinnGen phenotype | G6_MUSDYST (muscular dystrophy) |
| Inheritance | Complex: AD for cardiomyopathies; AR for most titinopathies (LGMD2J); AD for HMERF (exon 344) |
| VCEP coverage | No active TTN VCEP — submit under general ACMG/AMP |

### ⚠️ Critical Warning

**Do NOT submit as Likely Pathogenic.** The pipeline mechanical calculation yields 6 pts (LP threshold), but:
1. **BP1 applies** (−1 pt): TTN missense default per Di Feo 2025 — "missense variant in gene where primarily truncating variants cause disease"
2. **2★ multi-submitter VUS consensus exists**: multiple expert clinical labs have independently examined this variant and agreed on VUS; a new submitter claiming LP against unanimous VUS consensus will face immediate pushback and damage credibility
3. **AR-pattern complication**: OR = 119.6 in heterozygotes for a primarily-AR disease requires explanation (cryptic biallelic, founder haplotype, or novel AD mechanism — none confirmed)

The **correct framing is VUS-with-evidence** to deposit the extraordinary FinnGen statistical data (OR = 119.6, p = 5.65 × 10⁻⁸⁶) into the ClinVar aggregate without asserting a reclassification that cannot currently be defended.

### ACMG/AMP Evidence

| Criterion | Strength | Points | Basis |
|-----------|----------|--------|-------|
| PS4 | Strong | 4 | FinnGen R12: OR = 119.6, p = 5.65 × 10⁻⁸⁶, muscular dystrophy (G6_MUSDYST) — statistically extraordinary |
| PP3 | Moderate | 1 | EVEE pathogenicity = 0.923; AlphaMissense calibrated threshold for TTN = 0.792 per Di Feo 2025 |
| PM2 | Supporting | 1 | Finnish AF = 0.09%; gnomAD = not detected |
| BP1 | Supporting | −1 | Default TTN missense penalty per Di Feo 2025: TTN disease primarily caused by truncating variants |
| **Pipeline total** | | **6** | **LP** (pipeline mechanical) |
| **Recommended submit** | | — | **VUS-with-evidence** |

### FinnGen R12 Association

| Endpoint | OR | p-value | Finnish AF |
|----------|----|---------|------------|
| G6_MUSDYST (muscular dystrophy) | **119.6** | **5.65 × 10⁻⁸⁶** | 0.09% |

The OR = 119.6 and p = 5.65 × 10⁻⁸⁶ represent one of the strongest variant-phenotype associations in the entire FinnGen R12 dataset. This data is the primary value being deposited.

### Existing ClinVar Submitters (all VUS)

Six records under VCV000281698, all classifying VUS, multiple conditions. The existing VUS consensus reflects legitimate expert uncertainty about TTN missense variants in the absence of functional data.

### ClinVar Submission

- **Recommended classification:** VUS-with-evidence *(NOT LP)*
- **Condition:** Choose based on protein region verification — if M-band: Dilated cardiomyopathy 1G (OMIM:604145); if A-band: consider broader "TTN-related titinopathy"
- **Submission framing:** "Evidence deposit to inform future reclassification" — explicitly acknowledge existing multi-submitter VUS consensus; present FinnGen data as new evidence, not a reclassification assertion
- **Value add:** Adds OR = 119.6, p = 5.65 × 10⁻⁸⁶ — extraordinary population-scale evidence that future curators with functional or segregation data can act on

### Pre-Submission Checklist

- [ ] **Read all 6 existing RCVs in VCV000281698** — understand each submitter's VUS reasoning before framing narrative
- [ ] Map chr2:178527204 to exact TTN exon and determine PSI (Percent Spliced In) value — constitutively expressed exons carry more weight
- [ ] Determine sarcomere region (Z-disc / I-band / A-band / M-band) for appropriate condition selection
- [ ] Check for Ig domain or Fn3 domain at position 35195 per Di Feo 2025 — missense in beta-sheets of Ig domains has additional folding impact
- [ ] Verify FinnGen case zygosity in G6_MUSDYST — any homozygotes or compound hets would transform the AR interpretation
- [ ] **Wait for Batch 1 to establish submitter credibility** before depositing this variant

---

## Summary Table

| # | Gene | HGVS | Classification | ClinVar | Batch | Key action |
|---|------|------|---------------|---------|-------|------------|
| 1 | MEPE | p.Lys70IlefsTer26 | **PATHOGENIC** (11 pts) | 0★ VUS | 1 | VEP verify; cite Ramo 2023 for PS4_Strong |
| 2 | F2 | p.Leu389Val | **LP-lean** (4 pts) | NOT IN CLINVAR | 1 | Novel entry; verify HGVS; map catalytic domain |
| 3 | MYH14 | p.Glu1214Lys | **LP-lean** (4 pts) | 1★ VUS (Ambry) | 1 | Specific OMIM condition; motor domain mapping |
| 4 | RP1 | p.Leu172Arg | **LP** (6–8 pts) | 2★ conflicting | 1 | **AR framing mandatory**; cite Zou 2021 + Invitae in-trans |
| 5 | F2 | chr11:46726097 | **LP** (6 pts) | 0★ VUS no criteria | 2 | VEP HGVS first; verify case count before PS4_Strong |
| 6 | TTN | p.Val35195Glu | **VUS-with-evidence** | 2★ VUS consensus | 2 | **Do NOT submit as LP**; frame as evidence deposit |

---

*Report generated from: `/mnt/home/ryo/finngen_r12_actionable.tsv` + external review `Clinvar_reclass.zip` (2026-05-22)*  
*Figure: `figures/supplement/supfig_finngen_acmg.png` · Artifact: `artifacts/finngen_resubmission.feather`*
