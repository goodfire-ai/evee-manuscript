# %% [markdown]
# # End-to-End Probe Inference Demo
#
# This notebook walks through the Evee pathogenicity prediction pipeline:
# 1. Load pre-extracted Evo2 activations for 8 ClinVar variants
# 2. Inspect the activation tensor layout
# 3. Compute the variant−reference diff
# 4. Run the bilinear covariance probe → pathogenicity scores
# 5. Compare predictions to ground truth
#
# **No GPU required.** All artifacts are pre-computed and included in the repo.
# Only standard libraries: `torch`, `safetensors`, `polars`.

# %%
from pathlib import Path

import polars as pl
import safetensors.torch
import torch

SAMPLES = Path(__file__).resolve().parents[1] / "artifacts" / "samples"

# %% [markdown]
# ## 1. Load Sample Variants
#
# 8 curated ClinVar variants from the gene-holdout test set:
# 4 pathogenic + 4 benign, across missense, splice, intronic, and synonymous.

# %%
metadata = pl.read_ipc(SAMPLES / "metadata.feather")
print(metadata)

# %% [markdown]
# ## 2. Load Activations
#
# Each variant has its own safetensors file containing a `[2, 2, 256, 4096]`
# tensor:
#
# | Dim | Size | Meaning |
# |-----|------|---------|
# | 0   | 2    | Direction: forward, backward (reverse complement) |
# | 1   | 2    | View: variant sequence, reference sequence |
# | 2   | 256  | Positions selected by cosine divergence |
# | 3   | 4096 | Evo2-7B block 27 hidden dimension |
#
# These are the positions where the model's internal representation differs
# most between the variant and reference genome — where it "notices" the
# mutation.

# %%
variant_ids = metadata["variant_id"].to_list()
activations = []
for vid in variant_ids:
    safe_name = vid.replace(":", "_")
    tensors = safetensors.torch.load_file(str(SAMPLES / f"{safe_name}.safetensors"))
    activations.append(tensors["activations"])

activations = torch.stack(activations).float()  # [8, 2, 2, 256, 4096]
print(f"Loaded {activations.shape[0]} variants: {activations.shape}")

# %%
# Inspect the first variant (COL1A1 missense pathogenic)
var_fwd = activations[0, 0, 0]  # variant, forward direction
ref_fwd = activations[0, 0, 1]  # reference, forward direction
diff = var_fwd - ref_fwd

print(f"Variant: {metadata[0, 'variant_id']} ({metadata[0, 'gene_name']}, {metadata[0, 'consequence']})")
print(f"  Variant activation norm:  {var_fwd.norm(dim=-1).mean():.2f}")
print(f"  Reference activation norm: {ref_fwd.norm(dim=-1).mean():.2f}")
print(f"  Difference norm:           {diff.norm(dim=-1).mean():.4f}")

# %% [markdown]
# ## 3. Bilinear Covariance Probe
#
# The probe has three stages:
#
# 1. **Projection**: Two learned projections (left, right) per direction
#    map each position from 4096-d to 64-d
# 2. **Covariance pooling**: `left.T @ right / K` captures second-order
#    interactions across all 256 positions → a 64×64 matrix
# 3. **Bilinear readout**: A learned bilinear form on the covariance
#    matrix produces pathogenicity logits
#
# All 203 annotation heads share the same covariance representation.
# Only the readout weights differ per head. Here we use just the
# pathogenicity head.

# %%
probe = safetensors.torch.load_file(str(SAMPLES / "probe_pathogenicity.safetensors"))
for name, tensor in probe.items():
    print(f"  {name}: {tensor.shape}")


# %%
def predict_pathogenicity(
    activations: torch.Tensor,
    probe: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Run the bilinear covariance probe on pre-extracted activations.

    Parameters
    ----------
    activations : [B, 2, 2, K, d] — directions × views × positions × features
    probe : dict of probe weight tensors

    Returns
    -------
    scores : [B] — P(pathogenic) per variant
    """
    # Variant − reference diff per direction
    diff_fwd = activations[:, 0, 0] - activations[:, 0, 1]  # [B, K, 4096]
    diff_bwd = activations[:, 1, 0] - activations[:, 1, 1]  # [B, K, 4096]

    # Project to 64-d (separate left/right per direction)
    left_fwd = diff_fwd @ probe["proj_left_fwd"].T + probe["proj_left_fwd_bias"]
    right_fwd = diff_fwd @ probe["proj_right_fwd"].T + probe["proj_right_fwd_bias"]
    left_bwd = diff_bwd @ probe["proj_left_bwd"].T + probe["proj_left_bwd_bias"]
    right_bwd = diff_bwd @ probe["proj_right_bwd"].T + probe["proj_right_bwd_bias"]

    # Covariance pooling: left.T @ right / K → [B, 64, 64]
    K = left_fwd.shape[1]
    cov = (
        torch.bmm(left_fwd.transpose(1, 2), right_fwd) / K
        + torch.bmm(left_bwd.transpose(1, 2), right_bwd) / K
        + probe["eye"].unsqueeze(0)  # residual
    )

    # Bilinear head readout
    feat = torch.einsum("ij,bjk,ik->bi", probe["head_left"], cov, probe["head_right"])
    logits = feat @ probe["out_weight"].T + probe["out_bias"]
    return torch.softmax(logits, dim=-1)[:, 1]


# %% [markdown]
# ## 4. Run Inference

# %%
with torch.no_grad():
    scores = predict_pathogenicity(activations, probe)

for vid, score in zip(variant_ids, scores.tolist()):
    print(f"  {vid}: {score:.4f}")

# %% [markdown]
# ## 5. Compare to Ground Truth

# %%
results = metadata.with_columns(
    pl.Series("score", [round(s, 4) for s in scores.tolist()]),
    pl.Series("prediction", ["pathogenic" if s > 0.5 else "benign" for s in scores.tolist()]),
).with_columns(
    (pl.col("prediction") == pl.col("label")).alias("correct"),
)
print(results.select("variant_id", "gene_name", "consequence", "label", "score", "prediction", "correct"))

n_correct = results["correct"].sum()
print(f"\nAccuracy: {n_correct}/{results.height} ({100 * n_correct / results.height:.0f}%)")

# %% [markdown]
# ## Summary
#
# The pipeline extracts Evo2-7B block 27 activations at 256 divergent
# positions per variant, computes the variant−reference diff from both
# reading directions, and feeds the result through a bilinear covariance
# probe that captures second-order position interactions.
#
# The full model achieves **0.97 AUROC** on 34K held-out variants
# (gene-level split), outperforming CADD, AlphaMissense, and other
# computational predictors across all consequence types.
