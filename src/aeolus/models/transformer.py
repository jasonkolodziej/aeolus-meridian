"""Spatiotemporal transformer backbone.

Scope v2.1 §3.2. Global context and long-range steering over gridded fields.

The v2.1-critical detail is not in the architecture but in what it is fed: this
model is pretrained on ERA5 (Stage A) and fine-tuned on GDAS analyses (Stage B),
and only the Stage B weights are registrable. See :mod:`aeolus.training.curriculum`.
"""

from __future__ import annotations

from .base import DEFAULT_LEADS, ModelSpec, require_torch


def build_transformer(
    n_variables: int = 8,
    patch_size: int = 4,
    grid_size: tuple[int, int] = (40, 40),
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 6,
    dropout: float = 0.1,
    lead_hours: tuple[int, ...] = DEFAULT_LEADS,
):
    """Patch-embedding transformer over storm-centred atmospheric fields.

    Returns ``(module, spec)``. A learned CLS token carries the pooled state; it
    is the latent handed to MERIDIAN, so it must summarise the synoptic regime
    rather than any single grid point.
    """
    torch = require_torch()
    nn = torch.nn

    nlat, nlon = grid_size
    if nlat % patch_size or nlon % patch_size:
        raise ValueError(f"grid {grid_size} is not divisible by patch_size {patch_size}")
    n_patches = (nlat // patch_size) * (nlon // patch_size)

    spec = ModelSpec(
        name="transformer", input_dim=n_variables, latent_dim=d_model, lead_hours=lead_hours
    )

    class FieldTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch = nn.Conv2d(n_variables, d_model, patch_size, stride=patch_size)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
            nn.init.trunc_normal_(self.pos, std=0.02)
            nn.init.trunc_normal_(self.cls, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, spec.output_dim)
            # Synoptic regime classification head (§3.2 outputs).
            self.regime = nn.Linear(d_model, 4)

        def encode(self, fields):
            tokens = self.patch(fields).flatten(2).transpose(1, 2)
            cls = self.cls.expand(tokens.shape[0], -1, -1)
            x = torch.cat([cls, tokens], dim=1) + self.pos
            return self.norm(self.encoder(x)[:, 0])

        def forward(self, fields):
            latent = self.encode(fields)
            track = self.head(latent).view(-1, len(lead_hours), spec.outputs_per_lead)
            return track, self.regime(latent)

    return FieldTransformer(), spec
