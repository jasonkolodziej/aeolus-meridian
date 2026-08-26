"""Train / validation / test splits.

Scope v2.1 §4.4. Two properties are enforced mechanically because both were
ambiguous in v2:

1. **Storm-wise disjointness.** No storm contributes fixes to more than one
   split. Splitting on individual synoptic times would let a model see a
   storm's 24h fix in training and its 36h fix in test, which is leakage of
   almost the entire signal.
2. **Temporal ordering.** The split boundaries are also chronological, so the
   test set is strictly in the future of the training set. v2's text described
   the split as "storm-wise, not time-wise" while defining it by year ranges;
   it is in fact both, and both properties are worth keeping -- so both are
   asserted rather than argued about.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .besttrack import Track


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    OPERATIONAL = "operational"


#: Season boundaries from §4.4, inclusive.
DEFAULT_BOUNDARIES: dict[Split, tuple[int, int]] = {
    Split.TRAIN: (1980, 2019),
    Split.VAL: (2020, 2022),
    Split.TEST: (2023, 2025),
    Split.OPERATIONAL: (2026, 2100),
}


class LeakageError(RuntimeError):
    """A storm appears in more than one split, or split ordering is violated."""


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    """Which storms belong to which split."""

    storms: dict[str, Split]
    boundaries: dict[Split, tuple[int, int]]

    def of(self, storm_id: str) -> Split:
        try:
            return self.storms[storm_id]
        except KeyError as exc:
            raise KeyError(f"storm {storm_id!r} is not in any split") from exc

    def ids(self, split: Split) -> tuple[str, ...]:
        return tuple(sorted(s for s, v in self.storms.items() if v is split))

    def counts(self) -> dict[Split, int]:
        return {s: len(self.ids(s)) for s in Split}


def assign_splits(
    tracks: list[Track],
    boundaries: dict[Split, tuple[int, int]] | None = None,
) -> SplitAssignment:
    """Assign whole storms to splits by season.

    A storm that straddles a year boundary (a late-December system whose track
    runs into January) is assigned by the season of its *first* fix, so it lands
    wholly in one split.
    """
    boundaries = boundaries or DEFAULT_BOUNDARIES
    _validate_boundaries(boundaries)

    assignment: dict[str, Split] = {}
    for track in tracks:
        season = track.season
        split = _split_for_season(season, boundaries)
        if split is None:
            continue
        prior = assignment.get(track.storm_id)
        if prior is not None and prior is not split:
            raise LeakageError(
                f"storm {track.storm_id} assigned to both {prior.value} and {split.value}"
            )
        assignment[track.storm_id] = split
    return SplitAssignment(storms=assignment, boundaries=boundaries)


def _split_for_season(season: int, boundaries: dict[Split, tuple[int, int]]) -> Split | None:
    for split, (lo, hi) in boundaries.items():
        if lo <= season <= hi:
            return split
    return None


def _validate_boundaries(boundaries: dict[Split, tuple[int, int]]) -> None:
    missing = set(Split) - set(boundaries)
    if missing:
        raise ValueError(f"missing boundaries for {sorted(s.value for s in missing)}")
    spans = sorted(boundaries.items(), key=lambda kv: kv[1][0])
    for (_, (lo, hi)), (nxt_split, (nxt_lo, _)) in zip(spans, spans[1:], strict=False):
        if hi >= nxt_lo:
            raise ValueError(f"overlapping split boundaries at {nxt_split.value}")
        if lo > hi:
            raise ValueError("split boundary start exceeds end")


def assert_no_leakage(assignment: SplitAssignment, tracks: list[Track]) -> None:
    """Verify storm-wise disjointness and chronological ordering."""
    seen: dict[str, Split] = {}
    for track in tracks:
        if track.storm_id not in assignment.storms:
            continue
        split = assignment.of(track.storm_id)
        prior = seen.setdefault(track.storm_id, split)
        if prior is not split:
            raise LeakageError(f"storm {track.storm_id} spans splits")

    latest_train = _latest_season(assignment, tracks, Split.TRAIN)
    earliest_val = _earliest_season(assignment, tracks, Split.VAL)
    earliest_test = _earliest_season(assignment, tracks, Split.TEST)
    if latest_train is not None and earliest_val is not None and latest_train >= earliest_val:
        raise LeakageError(
            f"train season {latest_train} is not strictly before val season {earliest_val}"
        )
    if earliest_val is not None and earliest_test is not None:
        latest_val = _latest_season(assignment, tracks, Split.VAL)
        if latest_val is not None and latest_val >= earliest_test:
            raise LeakageError(
                f"val season {latest_val} is not strictly before test season {earliest_test}"
            )


def _seasons(assignment: SplitAssignment, tracks: list[Track], split: Split) -> list[int]:
    ids = set(assignment.ids(split))
    return [t.season for t in tracks if t.storm_id in ids]


def _latest_season(assignment, tracks, split) -> int | None:
    seasons = _seasons(assignment, tracks, split)
    return max(seasons) if seasons else None


def _earliest_season(assignment, tracks, split) -> int | None:
    seasons = _seasons(assignment, tracks, split)
    return min(seasons) if seasons else None


def filter_tracks(
    tracks: list[Track], assignment: SplitAssignment, split: Split
) -> list[Track]:
    ids = set(assignment.ids(split))
    return [t for t in tracks if t.storm_id in ids]
