"""MERIDIAN ensemble verification.

Scope v2.1 §7.3. CRPS, Brier score, spread-skill and rank histograms. The
spread-skill ratio is the one to watch: §10 flags underdispersion as the
characteristic failure of an ensemble conditioned on a deterministic guess, and
a ratio well under 1 is what that failure looks like numerically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def crps_ensemble(members: np.ndarray, observation: float) -> float:
    """Continuous Ranked Probability Score for one ensemble forecast.

    Uses the kernel form ``CRPS = E|X - y| - 0.5 * E|X - X'|`` given by Gneiting
    and Raftery (2007, doi:10.1198/016214506000001437); the score itself is due
    to Matheson and Winkler (1976, doi:10.1287/mnsc.22.10.1087).

    CRPS is strictly proper, so it is minimised only by reporting the true
    predictive distribution -- it cannot be gamed by hedging the spread. Lower is
    better, and for a single member it reduces to absolute error.
    """
    x = np.asarray(members, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("empty ensemble")
    term1 = np.abs(x - observation).mean()
    term2 = np.abs(x[:, None] - x[None, :]).mean()
    return float(term1 - 0.5 * term2)


def crps_series(members: np.ndarray, observations: np.ndarray) -> float:
    """Mean CRPS over many cases. ``members`` has shape (n_cases, n_members)."""
    members = np.asarray(members, dtype=float)
    observations = np.asarray(observations, dtype=float)
    if members.ndim != 2:
        raise ValueError("members must be 2-D (n_cases, n_members)")
    if members.shape[0] != observations.shape[0]:
        raise ValueError("case count mismatch between members and observations")
    return float(np.mean([crps_ensemble(m, o) for m, o in zip(members, observations, strict=True)]))


def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Brier score for binary events such as landfall (§7.3)."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if p.shape != y.shape:
        raise ValueError("shape mismatch")
    if p.size == 0:
        raise ValueError("empty input")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("outcomes must be 0 or 1")
    return float(np.mean((p - y) ** 2))


@dataclass(frozen=True, slots=True)
class SpreadSkill:
    spread: float
    skill: float

    @property
    def ratio(self) -> float:
        """Spread / RMSE. A well-calibrated ensemble sits near 1."""
        return float(self.spread / self.skill) if self.skill > 0 else float("inf")

    @property
    def underdispersed(self) -> bool:
        """True when the ensemble is over-confident (ratio below 0.8)."""
        return self.ratio < 0.8

    @property
    def overdispersed(self) -> bool:
        return self.ratio > 1.25


def spread_skill(members: np.ndarray, observations: np.ndarray) -> SpreadSkill:
    """Ensemble spread against ensemble-mean RMSE.

    Spread uses the ddof=1 standard deviation across members, averaged in
    variance space; comparing it to the ensemble-mean RMSE is the standard
    calibration diagnostic.

    This requires an observation, so it is a **post-hoc verification tool only**.
    It cannot gate a product at forecast time -- see
    :func:`aeolus.inference.postprocess.build_cone` for the real-time proxy used
    there instead.
    """
    members = np.asarray(members, dtype=float)
    observations = np.asarray(observations, dtype=float)
    if members.ndim != 2:
        raise ValueError("members must be 2-D (n_cases, n_members)")
    if members.shape[1] < 2:
        raise ValueError("spread needs at least 2 members")
    variance = members.var(axis=1, ddof=1).mean()
    rmse = np.sqrt(((members.mean(axis=1) - observations) ** 2).mean())
    return SpreadSkill(spread=float(np.sqrt(variance)), skill=float(rmse))


def rank_histogram(members: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """Rank (Talagrand) histogram counts, length ``n_members + 1``.

    A flat histogram is consistent with calibration; a U shape is consistent with
    underdispersion.

    Read it with Hamill (2001, doi:10.1175/1520-0493(2001)129<0550:IORHFV>2.0.CO;2)
    in hand: a U shape does **not** uniquely indicate underdispersion. Observation
    error and conditional biases produce the same signature. Since underdispersion
    is the failure mode MERIDIAN is most likely to exhibit, the temptation to read
    every U as confirmation is exactly the error to avoid -- pair it with the
    spread-skill ratio and with per-regime stratification before concluding.
    """
    members = np.asarray(members, dtype=float)
    observations = np.asarray(observations, dtype=float)
    if members.ndim != 2:
        raise ValueError("members must be 2-D")
    n_members = members.shape[1]
    counts = np.zeros(n_members + 1, dtype=int)
    for row, obs in zip(members, observations, strict=True):
        counts[int(np.sum(np.sort(row) < obs))] += 1
    return counts


def reliability(probabilities: np.ndarray, outcomes: np.ndarray, bins: int = 10):
    """Reliability curve: (bin centres, observed frequency, counts)."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    freq = np.full(bins, np.nan)
    counts = np.zeros(bins, dtype=int)
    for b in range(bins):
        sel = idx == b
        counts[b] = int(sel.sum())
        if counts[b]:
            freq[b] = float(y[sel].mean())
    return centres, freq, counts


def ensemble_percentiles(members: np.ndarray, percentiles=(10, 50, 90)) -> dict[int, float]:
    x = np.asarray(members, dtype=float).ravel()
    return {int(q): float(np.percentile(x, q)) for q in percentiles}
