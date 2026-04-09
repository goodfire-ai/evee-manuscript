#!/usr/bin/env python3
"""
Prepare Figure 2B panel data — binary annotation head AUROC by category.

Reads eval.json from the clinvar-evo2-probe-v1 run and heads.json from
variant-viewer to assign viewer groups, then maps to broad categories.

Input:  clinvar-evo2-probe-v1/eval.json, variant-viewer/heads.json
Output: data/panels/fig2b.csv
"""
import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "fig2b.feather"

PROBE_DIR = Path(
    "/mnt/polished-lake/artifacts/fellows-shared/life-sciences/genomics/mendelian/website_probes/v4/token"
)
EVAL_JSON = PROBE_DIR / "eval.json"
HEADS_JSON = Path("/mnt/polished-lake/home/ryamamoto/variant-viewer/heads.json")


def assign_viewer_group(head: str, viewer_heads_map: dict) -> str:
    if head in viewer_heads_map:
        return viewer_heads_map[head]
    if head.startswith("atacseq_"):
        return "ATAC-seq"
    if head.startswith("chipseq_"):
        return "ChIP-seq"
    if head.startswith("chromhmm_") or head.startswith("fstack_"):
        return "Chromatin"
    if head.startswith("ccre_"):
        return "Chromatin"
    if head.startswith("interpro_"):
        return "InterPro"
    if head.startswith("pfam_") and "_x_path" in head:
        return "Pfam × Path"
    if head.startswith("pfam_"):
        return "Pfam"
    if head.startswith("amino_acid_"):
        return "Substitution"
    if head.startswith("elm_"):
        return "ELM Motif"
    if head.startswith("ptm_"):
        return "PTM"
    if head.startswith("spliceai_"):
        return "Splice"
    if head.startswith("gtex_"):
        return "Expression"
    if head.startswith("gnomad_"):
        return "Population"
    if head.startswith("dna_shape_"):
        return "Sequence"
    if head.startswith("secondary_structure_"):
        return "Protein"
    if head.startswith("region_"):
        return "Region"
    if head.startswith("is_splice") or head.startswith("is_exon_to") or head.startswith("is_intron_to") or head in [
        "is_branchpoint_region", "is_polypyrimidine_tract"
    ]:
        return "Splice"
    if head in ["plddt", "phi", "psi", "sasa", "rsa", "n_contacts", "is_buried", "is_low_plddt"]:
        return "Protein"
    if head in [
        "cadd_c", "cadd_wg_c", "revel_c", "clinpred_c", "bayesdel_c", "vest4_c",
        "alphamissense_c", "primateai_c", "sift_c", "polyphen_c", "mcap_c",
        "mpc_c", "deogen2_c", "metalr_c", "mutpred_c", "mvp_c", "eve_c",
    ]:
        return "Clinical"
    if head in ["gerp_c", "remm_c", "regulomedb_c", "phastcons_100way", "phylop_100way"]:
        return "Conservation"
    if head in [
        "blosum62_c", "grantham_c", "hydrophobicity_c", "volume_c", "mw_c",
        "charge_altering", "aa_swap",
    ]:
        return "Substitution"
    if head in ["pathogenic", "csq_x_path", "consequence_label", "impact", "splice_disrupting"]:
        return "Variant Effect"
    if head in ["loeuf_c"]:
        return "Constraint"
    if head in ["cpg_density", "is_cpg_island", "gc_content", "trinuc_mutation_rate", "recomb_rate"]:
        return "Sequence"
    if head.startswith("in_") or head.startswith("is_"):
        return "Structure"
    return "Other"


def broad_group(vg: str) -> str:
    if vg in ["ATAC-seq", "ChIP-seq", "Chromatin"]:
        return "Regulatory"
    if vg == "InterPro":
        return "InterPro"
    if vg in ["Pfam", "Pfam × Path"]:
        return "Pfam"
    if vg in ["Protein", "Structure"]:
        return "Protein & Structure"
    if vg in ["Substitution", "ELM Motif", "PTM"]:
        return "Substitution & Motif"
    if vg == "Clinical":
        return "Clinical"
    if vg in ["Splice", "Variant Effect"]:
        return "Splicing"
    if vg in ["Conservation", "Constraint"]:
        return "Conservation"
    if vg in ["Sequence", "Region"]:
        return "Sequence & Region"
    if vg in ["Expression", "Population"]:
        return "Expression"
    return vg


def main():
    with open(EVAL_JSON) as f:
        eval_data = json.load(f)
    with open(HEADS_JSON) as f:
        viewer_heads_map = {h: cfg["group"] for h, cfg in json.load(f)["heads"].items()}

    rows = []
    for head, m in eval_data.items():
        if m.get("kind") != "binary":
            continue
        auroc = m.get("auc")
        if auroc is None:
            continue
        vg = assign_viewer_group(head, viewer_heads_map)
        bg = broad_group(vg)
        if bg == "Pathogenicity":
            continue
        rows.append({"head": head, "broad_group": bg, "auroc": auroc})

    df = pl.DataFrame(rows)
    df = df.sort("broad_group", "head")
    df.write_csv(OUT)
    print(f"Saved {len(df)} rows to {OUT}")


if __name__ == "__main__":
    main()
