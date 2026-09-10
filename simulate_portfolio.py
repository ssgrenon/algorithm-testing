"""Sweeps entry count (1..N) and reports P(at least one entry survives the
full season) plus a secondary "how deep did your best entry usually get"
metric, using analysis/portfolio_simulator.py's Monte Carlo simulation.

This is a standalone what-if analysis tool, separate from the live
weekly-pick app -- it uses a synthetic season generator (see
analysis/synthetic_season.py) rather than live ESPN data, since this
question ("how many entries should I buy before the season starts") is
about season-long statistics no live API can answer, and this sandboxed
dev environment can't reach ESPN's API at all regardless.

Usage:
    python simulate_portfolio.py [--max-entries 10] [--trials-per-season 1000]
                                  [--num-seasons 100] [--seed 42] [--weeks 18]
                                  [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from analysis.portfolio_simulator import run_sweep

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "portfolio_sweep_results.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep entry count 1..N for the survivor-pool portfolio analysis.")
    parser.add_argument("--max-entries", type=int, default=10, help="sweep entry counts 1..this (default 10)")
    parser.add_argument("--trials-per-season", type=int, default=1000)
    parser.add_argument("--num-seasons", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weeks", type=int, default=18)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    entry_counts = list(range(1, args.max_entries + 1))

    start = time.time()
    results = run_sweep(
        entry_counts=entry_counts,
        trials_per_season=args.trials_per_season,
        num_seasons=args.num_seasons,
        seed=args.seed,
        weeks=args.weeks,
    )
    elapsed = time.time() - start

    print(
        f"{sum(r.trials for r in results.values())} total trials across "
        f"{len(entry_counts)} entry counts in {elapsed:.1f}s\n"
    )
    print(f"{'N':>3}  {'P(survive full season)':>24}  {'+/- SE':>8}  {'avg weeks (best entry)':>23}")
    for n in entry_counts:
        r = results[n]
        print(
            f"{n:>3}  {r.probability * 100:>22.3f}%  "
            f"{r.standard_error * 100:>7.3f}%  {r.avg_best_entry_weeks_survived:>23.2f}"
        )

    payload = {
        "config": {
            "max_entries": args.max_entries,
            "trials_per_season": args.trials_per_season,
            "num_seasons": args.num_seasons,
            "seed": args.seed,
            "weeks": args.weeks,
        },
        "results": [
            {
                "entry_count": n,
                "trials": results[n].trials,
                "probability": results[n].probability,
                "standard_error": results[n].standard_error,
                "avg_best_entry_weeks_survived": results[n].avg_best_entry_weeks_survived,
            }
            for n in entry_counts
        ],
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
