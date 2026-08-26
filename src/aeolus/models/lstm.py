"""LSTM / GRU track and intensity baseline.

Scope v2.1 §3.1. The fast model: it runs every cycle, warm-starts the ensemble
and provides the low-latency first guess (30 s target, §6.2.2).

Inputs are storm-history sequences built from *working*-quality fixes -- see
:mod:`aeolus.data.besttrack` for why that distinction is load-bearing.
"""

from __future__ import annotations

from .base import DEFAULT_LEADS, ModelSpec, require_torch


def build_lstm(
    input_dim: int = 14,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.1,
    lead_hours: tuple[int, ...] = DEFAULT_LEADS,
    cell: str = "lstm",
):
    """Construct the recurrent baseline.

    Returns ``(module, spec)``. The head predicts *displacements* from the
    current position rather than absolute lat/lon: absolute coordinates make the
    model memorise basin climatology, which looks excellent in training and
    fails on any storm outside the usual corridor.
    """
    torch = require_torch()
    nn = torch.nn

    if cell not in {"lstm", "gru"}:
        raise ValueError(f"cell must be 'lstm' or 'gru', got {cell!r}")

    spec = ModelSpec(
        name="lstm", input_dim=input_dim, latent_dim=hidden_dim, lead_hours=lead_hours
    )

    class TrackRNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
            self.rnn = rnn_cls(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.norm = nn.LayerNorm(hidden_dim)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, spec.output_dim),
            )

        def encode(self, x):
            """Return the latent used by MERIDIAN and the fusion model (§5.7)."""
            out, _ = self.rnn(x)
            return self.norm(out[:, -1, :])

        def forward(self, x):
            latent = self.encode(x)
            return self.head(latent).view(-1, len(lead_hours), spec.outputs_per_lead)

    return TrackRNN(), spec
