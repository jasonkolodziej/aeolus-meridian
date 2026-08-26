"""Dual-flavor features and storm-wise splits (Scope v2.1 §4.3, §4.4, §4.6.1)."""

from datetime import UTC, datetime

import numpy as np
import pytest

from aeolus.data.features import (
    FEATURE_NAMES,
    FlavorMismatchError,
    Normalizer,
    assert_flavor,
    compute_environment_features,
    deep_layer_shear,
)
from aeolus.data.sources import Flavor
from aeolus.data.splits import (
    LeakageError,
    Split,
    assert_no_leakage,
    assign_splits,
    filter_tracks,
)
from aeolus.data.synthetic import generate_archive, generate_fields, paired_fields

T = datetime(2026, 8, 6, 6, tzinfo=UTC)


def test_features_carry_their_flavor():
    fields = generate_fields(T, Flavor.GDAS_FINETUNE, seed=0)
    assert compute_environment_features(fields).flavor is Flavor.GDAS_FINETUNE


def test_feature_vector_length_matches_the_names():
    fs = compute_environment_features(generate_fields(T, Flavor.ERA5_PRETRAIN, seed=0))
    assert fs.values.size == len(FEATURE_NAMES)
    assert set(fs.as_dict()) == set(FEATURE_NAMES)


def test_flavor_guard_rejects_a_pretrain_feature_on_the_operational_path():
    fs = compute_environment_features(generate_fields(T, Flavor.ERA5_PRETRAIN, seed=0))
    with pytest.raises(FlavorMismatchError, match="must not mix"):
        assert_flavor(fs, Flavor.GDAS_FINETUNE)


def test_the_same_code_path_computes_both_flavors():
    """Any ERA5/GDAS difference must come from the data, not the arithmetic."""
    era5, gdas = paired_fields(T, seed=7)
    a = compute_environment_features(era5)
    b = compute_environment_features(gdas)
    assert a.values.shape == b.values.shape
    assert not np.allclose(a.values, b.values)


def test_synthetic_gdas_carries_a_measurable_offset_from_era5():
    """Without this the skew audit would have nothing to detect."""
    deltas = []
    for i in range(20):
        era5, gdas = paired_fields(T, seed=i)
        deltas.append(
            compute_environment_features(gdas)["rh700_pct"]
            - compute_environment_features(era5)["rh700_pct"]
        )
    assert abs(float(np.mean(deltas))) > 0.5


def test_shear_increases_when_the_upper_flow_strengthens():
    low = deep_layer_shear(generate_fields(T, Flavor.ERA5_PRETRAIN, seed=0, shear_kt=5.0))[0]
    high = deep_layer_shear(generate_fields(T, Flavor.ERA5_PRETRAIN, seed=0, shear_kt=40.0))[0]
    assert high > low


def test_potential_intensity_falls_as_shear_rises():
    warm = compute_environment_features(
        generate_fields(T, Flavor.ERA5_PRETRAIN, seed=0, shear_kt=5.0)
    )
    sheared = compute_environment_features(
        generate_fields(T, Flavor.ERA5_PRETRAIN, seed=0, shear_kt=45.0)
    )
    assert sheared["potential_intensity_kt"] < warm["potential_intensity_kt"]


def test_normalizer_refuses_to_fit_across_flavors():
    era5, gdas = paired_fields(T, seed=1)
    with pytest.raises(FlavorMismatchError, match="mixed flavors"):
        Normalizer.fit(
            [compute_environment_features(era5), compute_environment_features(gdas)]
        )


def test_normalizer_rejects_the_wrong_flavor_at_apply_time():
    sets = [
        compute_environment_features(generate_fields(T, Flavor.GDAS_FINETUNE, seed=i))
        for i in range(10)
    ]
    norm = Normalizer.fit(sets)
    era5 = compute_environment_features(generate_fields(T, Flavor.ERA5_PRETRAIN, seed=0))
    with pytest.raises(FlavorMismatchError):
        norm.apply(era5)


# --- splits ----------------------------------------------------------------


def test_splits_follow_the_scope_season_boundaries():
    tracks = generate_archive(2018, 2026, storms_per_season=4, seed=1)
    assignment = assign_splits(tracks)
    for track in tracks:
        split = assignment.of(track.storm_id)
        if track.season <= 2019:
            assert split is Split.TRAIN
        elif track.season <= 2022:
            assert split is Split.VAL
        elif track.season <= 2025:
            assert split is Split.TEST
        else:
            assert split is Split.OPERATIONAL


def test_no_storm_appears_in_two_splits():
    tracks = generate_archive(2015, 2026, storms_per_season=5, seed=2)
    assignment = assign_splits(tracks)
    assert_no_leakage(assignment, tracks)
    ids = [set(assignment.ids(s)) for s in Split]
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            assert not (a & b)


def test_splits_are_also_chronologically_ordered():
    tracks = generate_archive(2015, 2026, storms_per_season=5, seed=3)
    assignment = assign_splits(tracks)
    train = filter_tracks(tracks, assignment, Split.TRAIN)
    test = filter_tracks(tracks, assignment, Split.TEST)
    assert max(t.season for t in train) < min(t.season for t in test)


def test_overlapping_boundaries_are_rejected():
    boundaries = {
        Split.TRAIN: (1980, 2020),
        Split.VAL: (2020, 2022),
        Split.TEST: (2023, 2025),
        Split.OPERATIONAL: (2026, 2100),
    }
    with pytest.raises(ValueError, match="overlapping"):
        assign_splits(generate_archive(2019, 2021, 3, seed=4), boundaries)


def test_leakage_is_detected_when_a_storm_id_repeats_across_seasons():
    tracks = generate_archive(2019, 2021, storms_per_season=3, seed=5)
    duplicate = tracks[0]
    later = [t for t in tracks if t.season >= 2021][0]
    forged = type(duplicate)(
        storm_id=duplicate.storm_id,
        fixes=tuple(
            type(f)(
                storm_id=duplicate.storm_id,
                valid_time=f.valid_time,
                lat=f.lat,
                lon=f.lon,
                max_wind_kt=f.max_wind_kt,
                min_pressure_mb=f.min_pressure_mb,
                quality=f.quality,
            )
            for f in later.fixes
        ),
    )
    with pytest.raises(LeakageError):
        assign_splits([duplicate, forged])
