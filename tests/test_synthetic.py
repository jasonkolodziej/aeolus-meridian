"""Synthetic data generator sanity checks."""

from datetime import UTC, datetime

import numpy as np

from aeolus.data.besttrack import TrackQuality
from aeolus.data.sources import Flavor
from aeolus.data.synthetic import generate_archive, generate_fields, generate_season

T = datetime(2026, 8, 6, 6, tzinfo=UTC)


def test_generated_tracks_are_final_quality():
    """Synthetic archives stand in for HURDAT2, which is post-season reanalysis."""
    for track in generate_season(2020, n_storms=3, seed=1):
        assert track.quality is TrackQuality.FINAL


def test_generation_is_reproducible():
    a = generate_season(2020, n_storms=3, seed=42)
    b = generate_season(2020, n_storms=3, seed=42)
    assert [t.as_array().tolist() for t in a] == [t.as_array().tolist() for t in b]


def test_storms_fall_within_the_requested_season():
    for track in generate_season(2021, n_storms=5, seed=2):
        assert track.season == 2021


def test_intensity_rises_and_falls():
    track = generate_season(2020, n_storms=1, seed=3)[0]
    winds = track.as_array()[:, 2]
    assert winds.argmax() not in (0, len(winds) - 1)


def test_fixes_are_six_hourly_and_synoptic():
    track = generate_season(2020, n_storms=1, seed=4)[0]
    times = [f.valid_time for f in track.fixes]
    for earlier, later in zip(times, times[1:]):
        assert (later - earlier).total_seconds() == 6 * 3600
        assert later.hour in (0, 6, 12, 18)


def test_archive_spans_the_requested_seasons():
    tracks = generate_archive(2018, 2022, storms_per_season=4, seed=5)
    seasons = {t.season for t in tracks}
    assert seasons == {2018, 2019, 2020, 2021, 2022}


def test_storm_ids_are_unique_within_the_archive():
    tracks = generate_archive(2018, 2022, storms_per_season=6, seed=6)
    ids = [t.storm_id for t in tracks]
    assert len(ids) == len(set(ids))


def test_fields_have_matching_shapes_and_carry_a_flavor():
    fields = generate_fields(T, Flavor.GDAS_FINETUNE, shape=(21, 21), seed=0)
    assert fields.shape == (21, 21)
    assert fields.flavor is Flavor.GDAS_FINETUNE


def test_gdas_fields_are_noisier_than_era5():
    era5 = generate_fields(T, Flavor.ERA5_PRETRAIN, seed=0)
    gdas = generate_fields(T, Flavor.GDAS_FINETUNE, seed=0)
    assert float(np.std(gdas.u850)) > float(np.std(era5.u850))
