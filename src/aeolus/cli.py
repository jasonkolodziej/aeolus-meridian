"""Command-line interface.

``aeolus schedule`` is the one to reach for first: it prints the real cycle
timeline for a given day, which is the fastest way to see the v2.1 timing change
against v2's assumed t+0:45 delivery.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta

from . import __version__
from .data.availability import LatencyOracle
from .data.besttrack import TrackQuality
from .data.sources import REGISTRY, Role
from .data.splits import assign_splits
from .data.synthetic import generate_archive
from .inference.cycle import climatological_ensemble, run_cycle
from .inference.scheduler import plan_cycle, plan_day
from .time_utils import parse_cycle_label


def _parse_day(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)


def cmd_schedule(args: argparse.Namespace) -> int:
    oracle = LatencyOracle(use_max_latency=args.worst_case)
    plans = plan_day(_parse_day(args.date), oracle)
    if not plans:
        print("no cycles could be planned for that day")
        return 1
    for plan in plans:
        print(plan.describe())
        status = "OK" if plan.meets_advisory_deadline else "MISSES ADVISORY DEADLINE"
        print(f"  advisory margin check: {status}\n")
    return 0


def cmd_sources(args: argparse.Namespace) -> int:
    rows = sorted(REGISTRY.values(), key=lambda s: (s.role.value, s.key))
    width = max(len(s.key) for s in rows)
    for src in rows:
        marker = " " if src.role is Role.OPERATIONAL else "*"
        hours = src.typical_latency.total_seconds() / 3600.0
        print(f"{marker} {src.key:<{width}}  {src.role.value:<14} ~{hours:>6.1f}h  {src.provider}")
    print("\n* not readable during a forecast cycle (Scope v2.1 §4.6)")
    return 0


def cmd_cycle(args: argparse.Namespace) -> int:
    import numpy as np

    from .data.besttrack import Fix
    from .inference.cycle import DeterministicForecast

    target = parse_cycle_label(args.cycle)
    oracle = LatencyOracle(use_max_latency=args.worst_case)
    plan = plan_cycle(target, oracle)

    fix = Fix(
        storm_id=args.storm,
        valid_time=target,
        lat=args.lat,
        lon=args.lon,
        max_wind_kt=args.wind,
        min_pressure_mb=990.0,
        quality=TrackQuality.ESTIMATED if plan.vitals_estimated else TrackQuality.WORKING,
    )

    leads = (12, 24, 36, 48, 72, 96, 120)

    def deterministic(plan_, fix_):
        steps = np.arange(1, len(leads) + 1, dtype=float)
        return DeterministicForecast(
            target_time=plan_.target_time,
            lead_hours=leads,
            lats=fix_.lat + 0.9 * steps,
            lons=fix_.lon - 1.1 * steps,
            winds_kt=np.clip(fix_.max_wind_kt + 4.0 * steps - 0.4 * steps**2, 15.0, None),
            contributors={"lstm": 0.2, "transformer": 0.35, "gnn": 0.25, "cnn": 0.2},
        )

    output = run_cycle(
        plan,
        fix,
        deterministic,
        lambda det: climatological_ensemble(det, n_members=args.members, seed=1),
    )
    print(json.dumps(output.payload(), indent=2))
    return 0


def cmd_splits(args: argparse.Namespace) -> int:
    tracks = generate_archive(args.start, args.end, seed=args.seed)
    assignment = assign_splits(tracks)
    for split, count in assignment.counts().items():
        lo, hi = assignment.boundaries[split]
        print(f"{split.value:<12} {count:>5} storms   seasons {lo}-{hi}")
    print(f"\n{len(tracks)} synthetic storms generated")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aeolus", description=__doc__)
    parser.add_argument("--version", action="version", version=f"aeolus {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("schedule", help="print the cycle timeline for a day")
    p.add_argument("date", help="UTC date, YYYY-MM-DD")
    p.add_argument("--worst-case", action="store_true", help="use max latencies")
    p.set_defaults(func=cmd_schedule)

    p = sub.add_parser("sources", help="list data sources and their roles")
    p.set_defaults(func=cmd_sources)

    p = sub.add_parser("cycle", help="run one demo forecast cycle")
    p.add_argument("cycle", help="cycle label, e.g. 20260806_06Z")
    p.add_argument("--storm", default="AL092026")
    p.add_argument("--lat", type=float, default=22.4)
    p.add_argument("--lon", type=float, default=-72.1)
    p.add_argument("--wind", type=float, default=85.0)
    p.add_argument("--members", type=int, default=20)
    p.add_argument("--worst-case", action="store_true")
    p.set_defaults(func=cmd_cycle)

    p = sub.add_parser("splits", help="summarise synthetic archive splits")
    p.add_argument("--start", type=int, default=2015)
    p.add_argument("--end", type=int, default=2026)
    p.add_argument("--seed", type=int, default=20260806)
    p.set_defaults(func=cmd_splits)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


_ = timedelta  # re-exported for downstream scripts that import it from here
