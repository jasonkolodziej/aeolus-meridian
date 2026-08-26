"""Cone, intensity PDF, landfall and RI products (Scope v2.1 §6.1)."""

import numpy as np
import pytest

from aeolus.inference.postprocess import (
    EnsembleMember,
    build_cone,
    build_products,
    intensity_pdf,
    landfall_probability,
    rapid_intensification_probability,
)

LEADS = (12, 24, 48, 72, 120)


def members(n=20, spread=1.2, wind_gain=0.0, seed=0):
    """Ensemble whose spread grows with lead time, as a real one does."""
    rng = np.random.default_rng(seed)
    steps = np.arange(len(LEADS), dtype=float)
    growth = 1.0 + steps
    out = []
    for i in range(n):
        out.append(
            EnsembleMember(
                member_id=i,
                lead_hours=LEADS,
                lats=25.0 + steps + rng.normal(0, spread, len(LEADS)) * growth,
                lons=-70.0 - steps + rng.normal(0, spread, len(LEADS)) * growth,
                winds_kt=80.0 + wind_gain * steps + rng.normal(0, 3, len(LEADS)),
            )
        )
    return out


def test_cone_has_one_segment_per_lead_time():
    cone, _ = build_cone(members())
    assert len(cone) == len(LEADS)
    assert [c.lead_hours for c in cone] == list(LEADS)


def test_cone_radius_grows_with_lead_time():
    cone, _ = build_cone(members())
    radii = [c.radius_nm for c in cone]
    assert radii[-1] > radii[0]


def test_small_ensemble_falls_back_to_climatological_radii():
    cone, notes = build_cone(members(n=5))
    assert all(c.basis == "climatology" for c in cone)
    assert any("minimum" in n for n in notes)


def test_empty_ensemble_is_refused():
    with pytest.raises(ValueError, match="empty ensemble"):
        build_cone([])


def test_intensity_pdf_percentiles_are_ordered():
    pdf = intensity_pdf(members())
    for lead, percentiles in pdf.items():
        values = [percentiles[q] for q in sorted(percentiles)]
        assert values == sorted(values)


def test_cone_is_ensemble_based_when_spread_is_credible():
    cone, notes = build_cone(members())
    assert all(c.basis == "ensemble" for c in cone)
    assert not notes


def test_underdispersed_ensemble_falls_back_to_climatology():
    """The characteristic diffusion failure: spread far tighter than climatology."""
    cone, notes = build_cone(members(spread=0.05))
    assert all(c.basis == "climatology" for c in cone)
    assert any("climatology" in n for n in notes)


def test_landfall_probability_is_one_when_all_members_hit():
    prob = landfall_probability(members(spread=0.05), 26.0, -71.0, threshold_nm=200.0)
    assert prob == 1.0


def test_landfall_probability_is_zero_for_a_distant_point():
    assert landfall_probability(members(), 5.0, -140.0, threshold_nm=50.0) == 0.0


def test_rapid_intensification_is_detected_when_winds_surge():
    prob = rapid_intensification_probability(members(wind_gain=20.0, spread=0.05))
    assert prob > 0.8


def test_no_rapid_intensification_for_a_steady_storm():
    assert rapid_intensification_probability(members(wind_gain=0.0)) < 0.2


def test_products_flag_rapid_intensification_above_the_threshold():
    products = build_products(members(wind_gain=20.0))
    assert products.rapid_intensification
    assert products.ri_probability > 0.3


def test_products_include_landfall_when_a_coastline_is_supplied():
    products = build_products(members(), coastline=(26.0, -71.0))
    assert products.landfall_probability is not None


def test_products_omit_landfall_without_a_coastline():
    assert build_products(members()).landfall_probability is None
