# %% [markdown]
# # End-to-End Probe Inference Demo
#
# This notebook walks through the Evee pathogenicity prediction pipeline:
# 1. Load pre-extracted Evo2 activations for 8 ClinVar variants
# 2. Inspect the activation tensor layout
# 3. Apply the unified-diff transform (var − ref, concat fwd+bwd)
# 4. Load the trained covariance probe
# 5. Run inference → pathogenicity scores
# 6. Compare predictions to ground truth
#
# **No GPU required** — all artifacts are pre-computed.

# %%
from pathlib import Path

import polars as pl
import safetensors.torch
import torch

ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path(".")
ARTIFACTS = ROOT / "artifacts"

# %% [markdown]
# ## 1. Load Sample Data
#
# We have 8 curated ClinVar variants from the gene-holdout test set:
# - 4 pathogenic, 4 benign
# - Missense, splice donor, intronic, synonymous consequences

# %%
metadata = pl.read_ipc(ARTIFACTS / "sample_metadata.feather")
print(metadata)

# %%
tensors = safetensors.torch.load_file(str(ARTIFACTS / "sample_activations.safetensors"))
activations = tensors["activations"].float()  # [8, 2, 3, 256, 4096]
print(f"Activations shape: {activations.shape}")
print(f"  dim 0: {activations.shape[0]} variants")
print(f"  dim 1: {activations.shape[1]} directions (forward, backward)")
print(f"  dim 2: {activations.shape[2]} views (var_same, ref_same, ref_cross)")
print(f"  dim 3: {activations.shape[3]} positions (top-256 by divergence)")
print(f"  dim 4: {activations.shape[4]} features (Evo2-7B block 27 hidden dim)")

# %% [markdown]
# ## 2. Activation Layout
#
# For each variant, Evo2 processes two genomic windows (forward and reverse
# complement). At each of 256 selected positions, we store three activation
# views:
#
# | View | Description |
# |------|-------------|
# | `var_same` | Variant sequence, read in the selecting direction |
# | `ref_same` | Reference sequence, same direction |
# | `ref_cross` | Reference sequence, opposite direction (remapped) |
#
# The 256 positions are selected by cosine divergence between variant and
# reference activations — the positions where the model "notices" the mutation
# most.

# %%
# Show activation magnitude for the first variant (COL1A1 missense pathogenic)
variant_acts = activations[0]  # [2, 3, 256, 4096]
var_same_fwd = variant_acts[0, 0]   # [256, 4096]
ref_same_fwd = variant_acts[0, 1]   # [256, 4096]

print(f"Variant: {metadata[0, 'variant_id']} ({metadata[0, 'gene_name']}, {metadata[0, 'consequence']})")
print(f"  var_same_fwd norm: {var_same_fwd.norm(dim=-1).mean():.2f}")
print(f"  ref_same_fwd norm: {ref_same_fwd.norm(dim=-1).mean():.2f}")
print(f"  diff norm:         {(var_same_fwd - ref_same_fwd).norm(dim=-1).mean():.4f}")

# %% [markdown]
# ## 3. Unified-Diff Transform
#
# The probe doesn't see raw activations. Instead, we compute:
#
# ```
# diff = var_same - ref_same           # what changed at each position
# unified = concat(diff_fwd, diff_bwd) # both directions → [256, 8192]
# ```
#
# This captures the mutation's effect on the model's internal representation,
# from both reading directions.

# %%
def unified_diff(acts: torch.Tensor) -> torch.Tensor:
    """[B, 2, 3, K, d] → [B, K, 2*d]: var-ref diff, concat fwd+bwd."""
    diff = acts[:, :, 0] - acts[:, :, 1]  # var_same - ref_same → [B, 2, K, d]
    return torch.cat([diff[:, 0], diff[:, 1]], dim=-1)  # → [B, K, 2*d]


probe_input = unified_diff(activations)
print(f"Probe input shape: {probe_input.shape}")
print(f"  {probe_input.shape[0]} variants × {probe_input.shape[1]} positions × {probe_input.shape[2]} features")

# %% [markdown]
# ## 4. Load Trained Probe
#
# The sequence covariance probe learns a low-rank projection of the
# activation diffs, then pools across positions using covariance statistics.
# This captures second-order interactions between positions — not just what
# changed, but how changes at different positions relate to each other.

# %%
checkpoint = torch.load(ARTIFACTS / "probe_weights.pt", map_location="cpu", weights_only=False)
config = checkpoint["model_config"]
state_dict = checkpoint["state_dict"]

n_binary = sum(1 for h in config["heads"].values() if h["kind"] == "binary")
n_cat = sum(1 for h in config["heads"].values() if h["kind"] == "categorical")
n_cont = sum(1 for h in config["heads"].values() if h["kind"] == "continuous")
print(f"Probe config:")
print(f"  d_model: {config['d_model']}")
print(f"  heads: {len(config['heads'])} ({n_binary} binary, {n_cat} categorical, {n_cont} continuous)")
print(f"\nShared projection layers:")
for name, param in state_dict.items():
    if name.startswith("proj_"):
        print(f"  {name}: {param.shape}")
print(f"\nPathogenicity head:")
for name, param in state_dict.items():
    if "pathogenic" in name:
        print(f"  {name}: {param.shape}")

# %% [markdown]
# ## 5. Run Inference
#
# The probe architecture has three stages:
#
# 1. **Bilinear projection**: Two separate projections (left, right) per
#    direction map each position from 4096-d to 64-d
# 2. **Covariance pooling**: `left.T @ right / K` captures second-order
#    interactions across all 256 positions → a 64×64 matrix per direction
# 3. **Head readout**: A bilinear form on the covariance matrix produces
#    the final logits
#
# The full probe has 203 heads, but they all share the same covariance
# representation. Only the readout weights differ per head.

# %%
# Extract the weights we need
proj_left_fwd = state_dict["proj_left_fwd.weight"]    # [64, 4096]
proj_right_fwd = state_dict["proj_right_fwd.weight"]   # [64, 4096]
proj_left_bwd = state_dict["proj_left_bwd.weight"]     # [64, 4096]
proj_right_bwd = state_dict["proj_right_bwd.weight"]   # [64, 4096]
bias_lf = state_dict["proj_left_fwd.bias"]
bias_rf = state_dict["proj_right_fwd.bias"]
bias_lb = state_dict["proj_left_bwd.bias"]
bias_rb = state_dict["proj_right_bwd.bias"]
eye = state_dict["_eye"]  # [64, 64] identity for residual

head_left = state_dict["head_modules.pathogenic.head_left.weight"]   # [128, 64]
head_right = state_dict["head_modules.pathogenic.head_right.weight"] # [128, 64]
out_w = state_dict["head_modules.pathogenic.out.weight"]             # [2, 128]
out_b = state_dict["head_modules.pathogenic.out.bias"]               # [2]

d_hidden = proj_left_fwd.shape[0]
d_head = head_left.shape[0]
print(f"Projection: 4096 → {d_hidden} (per direction, left+right)")
print(f"Covariance: {d_hidden}×{d_hidden} matrix per direction")
print(f"Head readout: bilinear {d_head}-d → 2 classes")

# %%
# Step-by-step inference
with torch.no_grad():
    diff_fwd = activations[:, 0, 0] - activations[:, 0, 1]  # [B, K, 4096]
    diff_bwd = activations[:, 1, 0] - activations[:, 1, 1]  # [B, K, 4096]

    # 1. Project to low-dim (separate left/right per direction)
    left_fwd = diff_fwd @ proj_left_fwd.T + bias_lf      # [B, K, 64]
    right_fwd = diff_fwd @ proj_right_fwd.T + bias_rf     # [B, K, 64]
    left_bwd = diff_bwd @ proj_left_bwd.T + bias_lb       # [B, K, 64]
    right_bwd = diff_bwd @ proj_right_bwd.T + bias_rb     # [B, K, 64]
    print(f"After projection: left_fwd {left_fwd.shape}, right_fwd {right_fwd.shape}")

    # 2. Covariance pooling: left.T @ right / K → [B, 64, 64]
    K = left_fwd.shape[1]
    cov_fwd = torch.bmm(left_fwd.transpose(1, 2), right_fwd) / K
    cov_bwd = torch.bmm(left_bwd.transpose(1, 2), right_bwd) / K
    cov = cov_fwd + cov_bwd + eye.unsqueeze(0)  # residual connection
    print(f"Covariance matrix: {cov.shape}")

    # 3. Bilinear head readout: feat_i = sum_{j,k} head_left[i,j] * cov[j,k] * head_right[i,k]
    feat = torch.einsum("ij,bjk,ik->bi", head_left, cov, head_right)  # [B, 128]
    print(f"Head features: {feat.shape}")

    # 4. Final classification
    logits = feat @ out_w.T + out_b  # [B, 2]
    probs = torch.softmax(logits, dim=-1)
    scores = probs[:, 1]  # P(pathogenic)

print(f"\nPathogenicity scores: {[round(s, 4) for s in scores.tolist()]}")

# %% [markdown]
# ## 6. Compare to Ground Truth

# %%
results = metadata.with_columns(
    pl.Series("score", [round(s, 4) for s in scores.tolist()]),
    pl.Series("prediction", ["pathogenic" if s > 0.5 else "benign" for s in scores.tolist()]),
)
results = results.with_columns(
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
# reading directions, and feeds the result through a covariance probe
# that captures second-order position interactions.
#
# The full model achieves **0.97 AUROC** on 34K held-out variants
# (gene-level split), outperforming CADD, AlphaMissense, and other
# computational predictors across all consequence types.
