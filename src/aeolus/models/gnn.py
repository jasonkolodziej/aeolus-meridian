"""Graph neural network over an irregular mesh.

Scope v2.1 §3.3. Message passing between geographically distant but physically
coupled nodes -- station networks, buoy arrays, icosahedral mesh points. Called
out in the scope as the critical component for rapid intensification, where the
inner-core processes are localised and non-Euclidean.
"""

from __future__ import annotations

from .base import DEFAULT_LEADS, ModelSpec, require_torch


def build_gnn(
    node_features: int = 12,
    edge_features: int = 3,
    hidden_dim: int = 128,
    n_layers: int = 6,
    lead_hours: tuple[int, ...] = DEFAULT_LEADS,
):
    """Interaction-network style GNN with edge conditioning.

    Returns ``(module, spec)``. Implemented with ``index_add_`` scatter rather
    than a PyG dependency, so the reference implementation installs with plain
    torch; the message/update decomposition is the standard one and swapping in
    PyG later is a local change.
    """
    torch = require_torch()
    nn = torch.nn

    spec = ModelSpec(
        name="gnn", input_dim=node_features, latent_dim=hidden_dim, lead_hours=lead_hours
    )

    class MessagePassing(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.message = nn.Sequential(
                nn.Linear(2 * hidden_dim + edge_features, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.update = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, h, edge_index, edge_attr):
            src, dst = edge_index[0], edge_index[1]
            msg = self.message(torch.cat([h[src], h[dst], edge_attr], dim=-1))
            agg = torch.zeros_like(h).index_add_(0, dst, msg)
            return self.norm(h + self.update(torch.cat([h, agg], dim=-1)))

    class MeshGNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(node_features, hidden_dim)
            self.layers = nn.ModuleList([MessagePassing() for _ in range(n_layers)])
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, spec.output_dim),
            )

        def encode(self, x, edge_index, edge_attr, center_index: int = 0):
            h = self.embed(x)
            for layer in self.layers:
                h = layer(h, edge_index, edge_attr)
            return h[center_index]

        def forward(self, x, edge_index, edge_attr, center_index: int = 0):
            latent = self.encode(x, edge_index, edge_attr, center_index)
            return self.head(latent).view(-1, len(lead_hours), spec.outputs_per_lead)

    return MeshGNN(), spec
