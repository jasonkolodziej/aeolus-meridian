"""Model architectures (Scope v2.1 §3). Requires the optional torch extra."""

import pytest

from aeolus.models.base import ModelSpec, require_torch, torch_available

pytestmark = pytest.mark.torch

LEADS = (12, 24, 48, 72, 120)


def test_spec_output_dim_is_leads_times_outputs():
    spec = ModelSpec("x", input_dim=4, latent_dim=8, lead_hours=LEADS)
    assert spec.output_dim == len(LEADS) * 3


def test_require_torch_message_names_the_extra():
    if torch_available():
        pytest.skip("torch installed; the failure path cannot be exercised")
    with pytest.raises(ModuleNotFoundError, match="uv sync --extra torch"):
        require_torch()


def test_lstm_shapes_and_latent():
    torch = require_torch()
    from aeolus.models.lstm import build_lstm

    model, spec = build_lstm(input_dim=14, hidden_dim=32, lead_hours=LEADS)
    x = torch.randn(4, 8, 14)
    assert model(x).shape == (4, len(LEADS), 3)
    assert model.encode(x).shape == (4, spec.latent_dim)


def test_gru_variant_builds():
    torch = require_torch()
    from aeolus.models.lstm import build_lstm

    model, _ = build_lstm(input_dim=6, hidden_dim=16, lead_hours=LEADS, cell="gru")
    assert model(torch.randn(2, 5, 6)).shape == (2, len(LEADS), 3)


def test_invalid_cell_is_rejected():
    from aeolus.models.lstm import build_lstm

    require_torch()
    with pytest.raises(ValueError, match="lstm|gru"):
        build_lstm(cell="rnn")


def test_cnn_is_resolution_agnostic():
    torch = require_torch()
    from aeolus.models.cnn import build_cnn

    model, spec = build_cnn(in_channels=5, base_width=8, depth=3, latent_dim=32,
                            lead_hours=LEADS)
    assert model.encode(torch.randn(2, 5, 64, 64)).shape == (2, 32)
    assert model.encode(torch.randn(2, 5, 128, 128)).shape == (2, 32)
    assert model(torch.randn(2, 5, 64, 64)).shape == (2, len(LEADS), 3)
    assert spec.latent_dim == 32


def test_transformer_emits_track_and_regime():
    torch = require_torch()
    from aeolus.models.transformer import build_transformer

    model, spec = build_transformer(n_variables=8, patch_size=4, grid_size=(32, 32),
                                    d_model=64, n_heads=4, n_layers=2, lead_hours=LEADS)
    track, regime = model(torch.randn(2, 8, 32, 32))
    assert track.shape == (2, len(LEADS), 3)
    assert regime.shape == (2, 4)
    assert model.encode(torch.randn(2, 8, 32, 32)).shape == (2, spec.latent_dim)


def test_transformer_rejects_an_indivisible_grid():
    from aeolus.models.transformer import build_transformer

    require_torch()
    with pytest.raises(ValueError, match="divisible"):
        build_transformer(grid_size=(30, 30), patch_size=4)


def test_gnn_message_passing_runs_over_a_small_graph():
    torch = require_torch()
    from aeolus.models.gnn import build_gnn

    model, _ = build_gnn(node_features=12, edge_features=3, hidden_dim=32, n_layers=2,
                         lead_hours=LEADS)
    x = torch.randn(10, 12)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    edge_attr = torch.randn(4, 3)
    assert model(x, edge_index, edge_attr).shape == (1, len(LEADS), 3)


def test_untrained_pinn_is_the_identity():
    """A physics corrector must not degrade a forecast before it has learned."""
    torch = require_torch()
    from aeolus.models.pinn import build_pinn

    model, _ = build_pinn(input_dim=10, hidden_dim=16, lead_hours=LEADS)
    candidate = torch.randn(3, len(LEADS), 3)
    corrected = model(torch.randn(3, 10), candidate)
    assert torch.allclose(corrected, candidate, atol=1e-6)


def test_physics_residuals_penalise_impossible_motion():
    torch = require_torch()
    from aeolus.models.pinn import physics_residuals

    slow = torch.zeros(1, 5, 3)
    slow[0, :, 0] = torch.tensor([20.0, 20.5, 21.0, 21.5, 22.0])
    fast = torch.zeros(1, 5, 3)
    fast[0, :, 0] = torch.tensor([20.0, 30.0, 40.0, 50.0, 60.0])
    assert physics_residuals(fast)["speed"] > physics_residuals(slow)["speed"]


def test_diffusion_sampling_produces_distinct_members():
    torch = require_torch()
    from aeolus.models.diffusion import build_diffusion

    model, _ = build_diffusion(latent_dim=16, hidden_dim=32, n_layers=2,
                               n_timesteps=10, lead_hours=LEADS)
    members = model.sample(torch.randn(1, 16), n_members=6)
    assert members.shape == (6, len(LEADS), 3)
    assert not torch.allclose(members[0], members[1])


def test_diffusion_accepts_extra_conditioning():
    torch = require_torch()
    from aeolus.models.diffusion import build_diffusion

    model, spec = build_diffusion(latent_dim=16, extra_conditioning_dim=8, hidden_dim=32,
                                  n_layers=2, n_timesteps=5, lead_hours=LEADS)
    assert spec.input_dim == 24
    assert model.sample(torch.randn(1, 24), n_members=3).shape == (3, len(LEADS), 3)


def test_fusion_weights_are_normalised_and_floored():
    torch = require_torch()
    from aeolus.models.fusion import build_fusion

    model, _ = build_fusion(n_models=5, context_dim=16, lead_hours=LEADS, weight_floor=0.05)
    w = model.weights(torch.randn(2, 16))
    assert w.shape == (2, len(LEADS), 5)
    assert torch.allclose(w.sum(-1), torch.ones(2, len(LEADS)), atol=1e-5)
    assert float(w.min()) >= 0.04


def test_fusion_rejects_the_wrong_number_of_models():
    torch = require_torch()
    from aeolus.models.fusion import build_fusion

    model, _ = build_fusion(n_models=5, context_dim=16, lead_hours=LEADS)
    with pytest.raises(ValueError, match="expected 5"):
        model(torch.randn(2, 3, len(LEADS), 3), torch.randn(2, 16))
