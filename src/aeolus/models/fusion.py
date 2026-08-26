"""Consensus fusion layer.

Scope v2.1 §6.1. Learns to weight the deterministic models' predictions into the
AEOLUS best guess, conditioned on the synoptic situation -- a GNN that is strong
on inner-core intensification and a transformer that is strong on steering
should not receive the same weight in every regime.

The learned weights are exposed rather than hidden inside the forward pass:
§10.1 requires the fusion layer to down-weight a divergent model, and that is
only auditable if the weights can be read out per cycle.
"""

from __future__ import annotations

from .base import DEFAULT_LEADS, ModelSpec, require_torch


def build_fusion(
    n_models: int = 5,
    context_dim: int = 16,
    hidden_dim: int = 64,
    lead_hours: tuple[int, ...] = DEFAULT_LEADS,
    weight_floor: float = 0.02,
):
    """Context-conditioned weighted consensus.

    Returns ``(module, spec)``. Weights are per-model and per-lead-time, softmax
    normalised, with a floor so a model is never fully zeroed on a short noisy
    validation record (see :func:`aeolus.inference.cycle.fusion_weights` for the
    same reasoning in the non-learned path).
    """
    torch = require_torch()
    nn = torch.nn

    n_leads = len(lead_hours)
    spec = ModelSpec(
        name="fusion", input_dim=context_dim, latent_dim=hidden_dim, lead_hours=lead_hours
    )

    class ConsensusFusion(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.context = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, n_models * n_leads),
            )
            self.weight_floor = weight_floor

        def weights(self, context):
            logits = self.context(context).view(-1, n_leads, n_models)
            w = torch.softmax(logits, dim=-1)
            w = w.clamp_min(self.weight_floor)
            return w / w.sum(dim=-1, keepdim=True)

        def forward(self, predictions, context):
            """``predictions`` is (batch, n_models, n_leads, 3)."""
            if predictions.shape[1] != n_models:
                raise ValueError(
                    f"expected {n_models} model predictions, got {predictions.shape[1]}"
                )
            w = self.weights(context).permute(0, 2, 1).unsqueeze(-1)
            return (predictions * w).sum(dim=1)

    return ConsensusFusion(), spec
