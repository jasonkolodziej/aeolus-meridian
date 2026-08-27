"""MERIDIAN: conditional diffusion ensemble generator.

Scope v2.1 §3.5. Denoises a track/intensity trajectory conditioned on AEOLUS
latents to produce 20-50 structurally diverse ensemble members.

A caveat worth keeping in the code rather than only in the review: conditioning
purely on the deterministic latent anchors the ensemble to the deterministic
guess and tends toward underdispersion, exactly when it matters most (bimodal
recurvature). ``extra_conditioning_dim`` exists so raw environmental fields or
GEFS/EPS perturbations can be concatenated to the conditioning vector, and
:func:`aeolus.metrics.probabilistic.spread_skill` is the check on whether that
was enough.
"""

from __future__ import annotations

from .base import DEFAULT_LEADS, ModelSpec, require_torch


def build_diffusion(
    latent_dim: int = 256,
    extra_conditioning_dim: int = 0,
    hidden_dim: int = 256,
    n_layers: int = 6,
    n_timesteps: int = 200,
    lead_hours: tuple[int, ...] = DEFAULT_LEADS,
):
    """Conditional denoising model over trajectories.

    Returns ``(module, spec)``. Uses a cosine noise schedule, which degrades
    less gracelessly at low step counts than the linear schedule -- and step
    count is a hard constraint here, since the whole ensemble must be generated
    inside the 13-minute §6.2.2 budget.

    If that budget binds in practice, the better answer is not the scheduler's
    load-shedding path -- which drops ensemble members, trading tail resolution
    for punctuality with no middle setting. Consistency models (Song et al. 2023,
    arXiv:2303.01469) support one-step generation by design while still allowing
    multistep sampling to trade compute for quality, giving a graded response to
    time pressure instead of a binary drop. They can be **distilled from an
    already-trained diffusion model**, so this is not an architectural commitment
    to make up front: train here, distil for the operational path if needed.
    SWIFT (Stock et al. 2025, arXiv:2509.25631) is prior art for weather.
    """
    torch = require_torch()
    nn = torch.nn

    n_leads = len(lead_hours)
    traj_dim = n_leads * 3
    cond_dim = latent_dim + extra_conditioning_dim

    spec = ModelSpec(
        name="diffusion", input_dim=cond_dim, latent_dim=hidden_dim, lead_hours=lead_hours
    )

    betas = _cosine_schedule(torch, n_timesteps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    class TrajectoryDenoiser(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("betas", betas)
            self.register_buffer("alphas", alphas)
            self.register_buffer("alpha_bars", alpha_bars)
            self.time_embed = nn.Sequential(
                nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
            )
            self.cond_embed = nn.Sequential(
                nn.Linear(cond_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
            )
            self.input_proj = nn.Linear(traj_dim, hidden_dim)
            self.blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.SiLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                    for _ in range(n_layers)
                ]
            )
            self.out = nn.Linear(hidden_dim, traj_dim)

        def forward(self, noisy_traj, t, conditioning):
            h = self.input_proj(noisy_traj.flatten(1))
            h = h + self.time_embed(t.float().unsqueeze(-1) / n_timesteps)
            h = h + self.cond_embed(conditioning)
            for block in self.blocks:
                h = h + block(h)
            return self.out(h).view(-1, n_leads, 3)

        @torch.no_grad()
        def sample(self, conditioning, n_members: int = 20, generator=None):
            """Ancestral sampling of ``n_members`` trajectories.

            ``conditioning`` is (1, cond_dim) or (n_members, cond_dim); a single
            row is broadcast so every member shares the AEOLUS guess and differs
            only through the noise path.
            """
            if conditioning.shape[0] == 1:
                conditioning = conditioning.expand(n_members, -1)
            elif conditioning.shape[0] != n_members:
                raise ValueError("conditioning batch must be 1 or n_members")

            x = torch.randn(n_members, n_leads, 3, generator=generator,
                            device=conditioning.device)
            for step in reversed(range(n_timesteps)):
                t = torch.full((n_members,), step, device=conditioning.device, dtype=torch.long)
                predicted_noise = self(x, t, conditioning)
                alpha = self.alphas[step]
                alpha_bar = self.alpha_bars[step]
                coef = (1.0 - alpha) / torch.sqrt(1.0 - alpha_bar)
                mean = (x - coef * predicted_noise) / torch.sqrt(alpha)
                if step > 0:
                    noise = torch.randn(x.shape, generator=generator, device=x.device)
                    x = mean + torch.sqrt(self.betas[step]) * noise
                else:
                    x = mean
            return x

    return TrajectoryDenoiser(), spec


def _cosine_schedule(torch, n_timesteps: int, s: float = 0.008):
    steps = torch.arange(n_timesteps + 1, dtype=torch.float32) / n_timesteps
    f = torch.cos((steps + s) / (1 + s) * torch.pi * 0.5) ** 2
    alpha_bars = f / f[0]
    betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
    return torch.clip(betas, 1e-4, 0.999)
