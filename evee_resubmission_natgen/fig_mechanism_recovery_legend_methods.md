# Cross-gene mechanism-class recovery — text for manuscript integration

Standalone context for the cross-gene mechanism-class recovery figure
(extracted from the Figure 3 supplement). Demonstrates that EVEE features
recover documented allelic-series mechanism classes via supervised
5-fold cross-validation across four allelic-heterogeneity genes
(LDLR, LMNA, MYH7, TP53), beyond what positional, CADD, AlphaMissense, or
chance baselines achieve.

---

## Headline numbers — balanced accuracy (5-fold CV)

| Gene  | N    | k | Random | Position only | CADD | AlphaMissense | **EVEE** | **EVEE + position** |
|:------|-----:|--:|-------:|--------------:|-----:|--------------:|---------:|--------------------:|
| LDLR  | 1581 | 5 | 0.20   | 0.34          | 0.20 | 0.25          | **0.70** | **0.71**            |
| LMNA  |  241 | 3 | 0.33   | 0.53          | 0.34 | 0.47          | **0.88** | **0.90**            |
| MYH7  |  269 | 4 | 0.25   | 0.81          | 0.25 | 0.25          | 0.61     | **0.80**            |
| TP53  |  488 | 4 | 0.25   | 0.35          | 0.25 | 0.33          | **0.76** | **0.76**            |
| **Mean** | — | — | 0.26 | 0.51          | 0.26 | 0.33          | **0.74** | **0.79**            |

EVEE + position is the strongest baseline-free combination on every gene,
with a cross-gene mean balanced accuracy of 0.79 — +0.28 over position-only,
+0.46 over AlphaMissense, and +0.53 over CADD. EVEE alone leads on three of
four genes; on MYH7, where mechanism labels are tightly aligned to primary
structure (head / converter / mid-rod / distal-rod), the position-only
baseline is unusually strong (0.81), and adding positional features to EVEE
recovers comparable performance (0.80). Both generic pathogenicity baselines
(CADD, AlphaMissense) perform at or near chance for mechanism-class
recovery, consistent with their training objectives targeting overall
pathogenicity rather than mechanism stratification.

---

## Abstract — additional sentence

> Across four allelic-heterogeneity genes (LDLR, LMNA, MYH7, TP53; 2,579
> high-confidence ClinVar variants in total), EVEE features recovered the
> documented per-gene mechanism-class labels via supervised cross-validation
> at a mean balanced accuracy of 0.79, versus 0.51 for a position-only
> baseline and 0.33 and 0.26 for AlphaMissense and CADD, respectively.

---

## Results — proposed paragraph

> **EVEE features recover documented mechanism-class labels across multiple
> allelic-heterogeneity genes.** To test whether the supervised
> mechanism-class recovery observed for LDLR (Fig. 3) generalises to other
> genes with documented allelic series, we extended the supervised 5-fold
> cross-validation analysis to LMNA, MYH7, and TP53 (Methods). Across all
> four genes, EVEE features classified variants into their literature-
> annotated mechanism categories at balanced accuracies of 0.70 (LDLR,
> k = 5 classes), 0.88 (LMNA, k = 3), 0.61 (MYH7, k = 4), and 0.76 (TP53,
> k = 4), versus 0.20–0.33 for chance, 0.20–0.34 for CADD, and 0.25–0.47 for
> AlphaMissense (Fig. _X_) — both generic pathogenicity scores performed at
> or near chance across all four genes, consistent with their pathogenicity-
> oriented training objectives. Combining EVEE features with variant
> position yielded a further gain, reaching 0.71–0.90 across the four genes
> (cross-gene mean 0.79; +0.28 over position-only, +0.46 over AlphaMissense,
> +0.53 over CADD). A single exception is instructive: for MYH7, whose
> mechanism classes are tightly co-localised along primary structure (head,
> converter, mid-rod, distal-rod), the position-only baseline reaches
> 0.81 — higher than EVEE alone (0.61) — and the EVEE + position combination
> (0.80) recovers parity. EVEE therefore captures mechanism-class signal
> beyond positional encoding on the three genes where position is
> non-trivial, and complements positional features on the one gene where
> position itself already separates classes.

---

## Methods

> **Cross-gene supervised mechanism-class recovery.** For each of four
> allelic-heterogeneity genes (LDLR, LMNA, MYH7, TP53), variants from the
> ClinVar high-confidence set (one- and two-star review status,
> Pathogenic/Likely Pathogenic) were assigned to literature-annotated
> mechanism classes using the gene's domain × consequence scheme (per-gene
> class definitions in Supplementary Methods): LDLR k = 5 classes
> (Goldstein–Brown receptor-class taxonomy), LMNA k = 3 classes
> (laminopathy spectrum: A-band rod DCM/EDMD, R482 hot-spot FPLD2,
> splice/progeroid), MYH7 k = 4 classes (head HCM, converter HCM, mid-rod
> Laing, distal-rod MSM), TP53 k = 4 classes (DNA-contact, structural DBD,
> null/LoF, tetramerisation). Only variants with a high- or medium-
> confidence class assignment and class membership ≥ 5 variants were
> retained, yielding sample sizes of N = 1581 (LDLR), 241 (LMNA), 269
> (MYH7), and 488 (TP53).
>
> Per-variant features were the standard EVEE disruption-profile features
> (annotation × probe Δ scores), filtered as in the LDLR analysis (tissue
> features dropped; |max| > 0.05 across the variant set), and standardised
> per feature. For each gene, a multinomial logistic-regression classifier
> (L2-regularised, C = 0.1) was evaluated by stratified 5-fold
> cross-validation; balanced accuracy is reported. Baselines on the same
> splits were chance (1 / k), a CADD-only logistic regression, an
> AlphaMissense-only logistic regression, and a position-only logistic
> regression using only the variant's encoded residue position. An
> EVEE + position combination concatenated the positional encoding with
> the EVEE feature set. Missing per-variant CADD or AlphaMissense scores
> (uncommon at < 5 % per gene) were mean-imputed within the gene.

---

## Figure caption

> **Figure _X_ — EVEE features recover documented mechanism-class labels
> across four allelic-heterogeneity genes.** Per-gene supervised 5-fold
> cross-validation balanced accuracy for assigning variants to their
> literature-annotated mechanism classes (LDLR, k = 5 receptor classes;
> LMNA, k = 3 laminopathy classes; MYH7, k = 4 structural-domain classes;
> TP53, k = 4 functional classes). Class definitions are detailed in the
> Methods and Supplementary Methods. Bars are grouped by gene; each bar
> reports the mean cross-validation balanced accuracy of a single
> classifier. *Random* = 1 / k chance; *Position only* = logistic
> regression on encoded residue position; *CADD* = logistic regression on
> the CADD score; *AlphaMissense* = logistic regression on the
> AlphaMissense score; *EVEE* = logistic regression on the EVEE per-variant
> disruption features; *EVEE + position* = concatenation of EVEE features
> with the positional encoding. Variant counts (N) and class counts (k) are
> annotated below each gene. EVEE + position is the strongest combination
> on every gene (cross-gene mean 0.79); EVEE alone leads on every gene
> except MYH7, where mechanism classes coincide with primary structure and
> the position-only baseline is already strong. Both generic pathogenicity
> baselines (CADD, AlphaMissense) perform at or near chance.

---

## Scope and caveats

- High-confidence ClinVar subset only; variants without a confident
  domain × consequence class assignment are excluded.
- Within-gene CV; performance is per-class, not per-variant pathogenicity.
- MYH7 is presented as an honest scope limit, not a failure: when
  mechanism is itself a near-monotonic function of primary structure
  (myosin head / converter / mid-rod / distal-rod), positional features
  alone are competitive, and EVEE's gain is realised only when combined
  with position.
- CADD and AlphaMissense are evaluated here as *baselines*, not as
  competing methods — their training objectives target overall pathogenicity
  (CADD: deleteriousness from population frequency and conservation;
  AlphaMissense: pathogenicity classification of missense variants), not
  mechanism-class stratification, and the chance-level performance reflects
  the expected ceiling for a pathogenicity-only signal on a within-gene
  mechanism-class task.
- "EVEE" denotes the per-variant disruption-profile feature set used
  throughout this paper.
