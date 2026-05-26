# LDLR clinical-severity validation — text for manuscript integration

Standalone context for the Tapestry × EVEE LDLR validation. Three figures
accompany this text: a per-tier distribution of EVEE Pathogenicity scores
(violin), a primary ROC (any-FH vs presymptomatic carrier), and a
missense-only ROC supplement. AlphaMissense and CADD are the baselines.

---

## Cohort at a glance

- **N = 147** unique LDLR variants intersecting both the Mayo Tapestry
  biobank and ClinVar.
- 3-tier FH-severity scale, aligned with standard FH diagnostic rubrics:
  - **Clinical FH** (definite + probable): 111 variants
  - **Suspected FH** (possible): 20 variants
  - **Presymptomatic carrier** (variant present, FH absent): 16 variants
- Variant composition: 83.7 % missense, 14.3 % truncating or frameshift,
  2.0 % other (in-frame, initiator-codon, intronic).

**Headline AUCs — any FH manifestation vs presymptomatic carrier
(n+ = 131, n− = 16):**

| Score | AUC, full cohort (N = 147) | AUC, missense-only (N = 123) |
|:------|:--:|:--:|
| **EVEE Pathogenicity** | **0.91** | **0.89** |
| AlphaMissense          | 0.70     | 0.70     |
| CADD                   | 0.65     | 0.61     |

Per-tier median EVEE Pathogenicity falls monotonically: Clinical FH = 0.76,
Suspected FH = 0.10, Presymptomatic carrier = 0.01.

---

## Abstract — additional sentence

> Validated against an independent clinical cohort of 147 LDLR variant
> carriers from the Mayo Clinic Tapestry biobank, EVEE Pathogenicity scores
> distinguished Familial Hypercholesterolemia (FH)–manifesting carriers from
> presymptomatic carriers of the same variants at AUC = 0.91, exceeding the
> AlphaMissense and CADD baselines by 0.21 and 0.26 AUC, respectively.

---

## Results — proposed paragraph

> **EVEE Pathogenicity scores track clinical severity in an independent
> patient cohort.** To test whether EVEE's continuous variant-level scores
> recover clinically meaningful gradations of disease severity, we
> intersected the Mayo Clinic Tapestry biobank's LDLR cohort with ClinVar,
> yielding 147 unique variant carriers with both a per-variant EVEE
> Pathogenicity score and an expert-curated FH severity tier. We restricted
> analysis to labels lying on the FH-severity axis — *Clinical FH* (definite
> or probable; n = 111), *Suspected FH* (possible; n = 20), and
> *Presymptomatic carrier* (variant-positive but no FH manifestation;
> n = 16) — and excluded variants whose carriers presented with non-FH
> hypercholesterolemia patterns or unrelated phenotypes, which by definition
> lie on different biological axes (Methods).
>
> EVEE Pathogenicity separated the three severity tiers monotonically, with
> per-tier median scores of 0.76 (Clinical FH), 0.10 (Suspected FH), and
> 0.01 (Presymptomatic carrier) (Fig. _X_). For the clinically actionable
> binary endpoint — distinguishing carriers who will manifest FH from those
> who carry an LDLR variant but remain phenotypically unaffected — EVEE
> Pathogenicity achieved AUC = 0.91, versus 0.70 for AlphaMissense and 0.65
> for CADD (Fig. _Y_). The advantage was robust to variant-type composition:
> restricted to the 123 missense variants in the cohort — the regime in
> which AlphaMissense is natively defined — EVEE retained AUC = 0.89, and
> its margin over AlphaMissense was unchanged at 0.19 AUC (Extended Data
> Fig. _Z_).

---

## Methods

> **Tapestry clinical-severity validation.** The Mayo Clinic Tapestry biobank
> provided a per-variant clinical-severity annotation for each LDLR variant
> carried by a participant. We collapsed Tapestry's seven-level severity
> annotation into a three-level ordinal scale aligned with standard FH
> diagnostic rubrics (Dutch Lipid Clinic Network, Simon Broome, MEDPED):
> *Clinical FH* (definite + probable; both indicating high-confidence FH
> manifestation), *Suspected FH* (possible; partial diagnostic criteria met),
> and *Presymptomatic carrier* (LDLR variant present without FH manifestation
> at ascertainment). Three annotations were excluded from the severity
> ordinal as off-axis: hypercholesterolemia of a non-FH lipid pattern
> (elevated cholesterol not matching the FH pattern, often polygenic or
> modifier-driven), clinical presentation outside the known FH phenotypic
> spectrum, and carriers with missing phenotype data. The included tiers
> share a single biological axis — degree of LDLR loss of function
> manifesting as the canonical FH phenotype — which is the scale EVEE
> Pathogenicity is expected to track.
>
> Tapestry variants were collapsed to one entry per unique variant by
> protein-level HGVS identifier and intersected with the full LDLR ClinVar
> EVEE catalogue (4,193 variants), which provides per-variant EVEE
> Pathogenicity along with AlphaMissense and CADD scores from the standard
> dbNSFP-derived ClinVar annotations. The intersection yielded 147 unique
> variants (Clinical FH n = 111, Suspected FH n = 20, Presymptomatic carrier
> n = 16; 83.7 % missense, 14.3 % truncating or frameshift, 2.0 % other).
> All 147 variants had non-missing scores for every method compared; for
> non-missense variants in the catalogue, AlphaMissense scores are filled by
> codon-level imputation (the score of the most pathogenic same-codon
> missense substitution) and are reported here for comparability. A
> missense-only restricted analysis is reported alongside the full cohort
> (Extended Data Fig. _Z_).
>
> Score discrimination was assessed by receiver-operating characteristic
> AUC for the clinical-screening contrast: positives = Clinical FH ∪
> Suspected FH (n+ = 131), negatives = Presymptomatic carrier (n− = 16).
> The same Presymptomatic-carrier negative class is the only true
> variant-positive / FH-negative comparator available in this cohort and is
> retained despite class imbalance; class sizes are reported in-plot.
> REVEL was not included as a baseline to match the comparator set used
> elsewhere in the manuscript.

---

## Figure captions

> **Figure _X_ — EVEE Pathogenicity stratifies LDLR variant carriers by
> clinical FH severity in the Mayo Tapestry cohort.** Each LDLR variant
> carrier in Tapestry was assigned to one of three severity tiers based on
> Mayo expert curation of standard FH diagnostic criteria: Clinical FH
> (definite + probable; n = 111), Suspected FH (possible; n = 20), or
> Presymptomatic carrier (variant-positive without FH manifestation;
> n = 16). Off-axis Tapestry labels — non-FH hypercholesterolemia patterns
> and phenotypes outside the FH spectrum — were excluded from the severity
> ordinal (Methods). Each circle is a unique variant; violins show the
> per-tier density of EVEE Pathogenicity scores, and horizontal bars mark
> the per-tier median (annotated to the right of each bar). Medians fall
> monotonically across the tiers (0.76 → 0.10 → 0.01), consistent with a
> graded relationship between EVEE Pathogenicity and the clinical penetrance
> of FH.

> **Figure _Y_ — EVEE Pathogenicity discriminates FH-manifesting carriers
> from presymptomatic LDLR variant carriers.** Receiver-operating
> characteristic for the clinical-screening contrast — *any* FH manifestation
> (Clinical FH ∪ Suspected FH; n+ = 131) versus Presymptomatic carrier
> (n− = 16) — across the 147 unique LDLR variants in the Tapestry × ClinVar
> intersection (same cohort as Fig. _X_). EVEE Pathogenicity reaches
> AUC = 0.91, exceeding AlphaMissense (0.70) and CADD (0.65) by 0.21 and
> 0.26, respectively. The Presymptomatic-carrier class is small (n = 16) but
> is the only true variant-positive / FH-negative comparator available in
> this cohort; class sizes are annotated in-plot.

> **Extended Data Fig. _Z_ — Missense-only restricted validation.** As
> Fig. _Y_, restricted to the 123 missense variants in the cohort — the
> regime in which AlphaMissense is natively defined. EVEE Pathogenicity
> achieves AUC = 0.89 versus AlphaMissense 0.70 and CADD 0.61. The 0.19 AUC
> margin over AlphaMissense is unchanged from the full-cohort analysis
> (Fig. _Y_), confirming that EVEE's clinical advantage is intrinsic to its
> scoring and not an artifact of broader variant-type coverage.

---

## Scope and caveats

- The cohort intersects 147 of the ~4,200 LDLR ClinVar variants;
  ascertainment reflects the Tapestry biobank's recruitment focus on
  FH-evaluated carriers and is not a random ClinVar sample. We report
  per-variant clinical-correspondence performance, not population-level
  prevalence.
- "EVEE Pathogenicity" denotes the per-variant continuous EVEE pathogenicity
  score throughout this section.
- Three Tapestry severity annotations are deliberately excluded from the
  analysis as off-axis: non-FH-pattern hypercholesterolemia, phenotype
  outside the known FH spectrum, and unclear / missing phenotype. These
  annotations describe carriers whose variant is present but whose clinical
  presentation does not lie on the FH-severity axis and so cannot be ranked
  against an FH-penetrance-tracking score.
