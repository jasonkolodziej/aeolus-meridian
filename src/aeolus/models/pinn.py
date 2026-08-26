"""Physics-informed corrector.

Scope v2.1 §3.6. Note the correction made in review: the governing equations are
*loss constraints*, not model inputs -- v2's §3.6 listed them under "Input",
which would be a category error in any implementation.

The corrector takes a candidate track and returns an adjusted one; the physics
enters through :func:`physics_residuals`, which penalises implausible motion.
"""

from __future__ import annotations

from .base import DEFAULT_LEADS, ModelSpec, require_torch

#: Earth rotation rate, for the Coriolis parameter.
OMEGA = 7.2921e-5
#: Storms faster than this over open water are not physically credible.
MAX_TRANSLATION_KT = 45.0


def build_pinn(
    input_dim: int = 14,
    hidden_dim: int = 128,
    n_layers: int = 4,
    lead_hours: tuple[int, ...] = DEFAULT_LEADS,
):
    """Residual corrector applied on top of a candidate forecast.

    Returns ``(module, spec)``. The network outputs a *correction*, initialised
    near zero, so an untrained corrector is the identity: a physics layer should
    never make a good forecast worse before it has learned anything.
    """
    torch = require_torch()
    nn = torch.nn

    spec = ModelSpec(
        name="pinn", input_dim=input_dim, latent_dim=hidden_dim, lead_hours=lead_hours
    )

    class PhysicsCorrector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = [nn.Linear(input_dim + spec.output_dim, hidden_dim), nn.Tanh()]
            for _ in range(n_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
            self.trunk = nn.Sequential(*layers)
            self.correction = nn.Linear(hidden_dim, spec.output_dim)
            nn.init.zeros_(self.correction.weight)
            nn.init.zeros_(self.correction.bias)

        def encode(self, environment, candidate):
            flat = candidate.flatten(1)
            return self.trunk(torch.cat([environment, flat], dim=-1))

        def forward(self, environment, candidate):
            latent = self.encode(environment, candidate)
            delta = self.correction(latent).view_as(candidate)
            return candidate + delta

    return PhysicsCorrector(), spec


def coriolis_parameter(lat_deg):
    """f = 2 * omega * sin(latitude)."""
    torch = require_torch()
    return 2.0 * OMEGA * torch.sin(torch.deg2rad(lat_deg))


def physics_residuals(track, dt_hours: float = 6.0):
    """Penalty terms for physically implausible track behaviour.

    ``track`` has shape (batch, n_leads, 3) as (lat, lon, wind). Three residuals:

    * **speed** -- translation faster than :data:`MAX_TRANSLATION_KT`
    * **curvature** -- implausibly sharp heading changes between steps
    * **hemisphere** -- poleward-left drift in the northern hemisphere, i.e.
      motion inconsistent with the sign of the Coriolis parameter

    Returned separately rather than summed so the loss weights are visible and
    tunable, and so a failure can be attributed to a specific constraint.
    """
    torch = require_torch()
    lat, lon = track[..., 0], track[..., 1]
    dlat = lat[:, 1:] - lat[:, :-1]
    dlon = (lon[:, 1:] - lon[:, :-1]) * torch.cos(torch.deg2rad(lat[:, :-1]))

    speed_kt = torch.sqrt(dlat**2 + dlon**2) * 60.0 / dt_hours
    speed_residual = torch.relu(speed_kt - MAX_TRANSLATION_KT).mean()

    heading = torch.atan2(dlon, dlat)
    curvature = torch.abs(heading[:, 1:] - heading[:, :-1])
    curvature = torch.minimum(curvature, 2 * torch.pi - curvature)
    curvature_residual = torch.relu(curvature - 0.6).mean()

    f = coriolis_parameter(lat[:, :-1])
    hemisphere_residual = torch.relu(-f * dlat).mean()

    return {
        "speed": speed_residual,
        "curvature": curvature_residual,
        "hemisphere": hemisphere_residual,
    }
