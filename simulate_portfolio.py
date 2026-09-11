"""Sweeps entry count (1..N) and reports P(at least one entry survives the
full season) plus a secondary "how deep did your best entry usually get"
metric, using analysis/portfolio_simulator.py's Monte Carlo simulation.

This is a standalone what-if analysis tool, separate from the live
weekly-pick app. Two data sources for the underlying seasons:

  --source synthetic (default): a randomized season generator (see
    analysis/synthetic_season.py), run many times for statistical power.
    Doesn't depend on any external data.

  --source real: real NFL seasons and closing spreads from
    nflverse/nfldata's public games.csv (analysis/nflverse_games.py),
    backtesting the exact same strategy against what actually happened.
    Each requested year that's in the data contributes one season; there's
    no synthetic-style repetition of a single season, so precision comes
    from --trials-per-season (repeated stochastic replays of each real
    season's real win probabilities) and from how many years you include.

Usage:
    python simulate_portfolio.py --source synthetic [--max-entries 10]
                                  [--trials-per-season 1000] [--num-seasons 100]
                                  [--seed 42] [--weeks 18] [--out results.json]

    python simulate_portfolio.py --source real --years 2016-2025
                                  [--trials-per-season 1000] [--games-csv path.csv]
                                  [--seed 42] [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

from analysis.fetch_historical_season import parse_years
from analysis.nflverse_games import fetch_and_load_seasons, load_seasons_from_file
from analysis.portfolio_simulator import SweepResult, run_sweep, run_sweep_on_seasons
from analysis.synthetic_season import MatchupProb

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "portfolio_sweep_results.json"


def _print_and_write(
    entry_counts: List[int],
    results: Dict[int, SweepResult],
    elapsed: float,
    config: dict,
    out: Path,
) -> None:
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
        "config": config,
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
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep entry count 1..N for the survivor-pool portfolio analysis.")
    parser.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--max-entries", type=int, default=10, help="sweep entry counts 1..this (default 10)")
    parser.add_argument("--trials-per-season", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)

    # --source synthetic only
    parser.add_argument("--num-seasons", type=int, default=100)
    parser.add_argument("--weeks", type=int, default=18)

    # --source real only
    parser.add_argument("--years", default="2016-2025", help='e.g. "2016-2025" or "2020,2022,2024"')
    parser.add_argument("--games-csv", type=Path, default=None, help="local games.csv; omit to fetch fresh")

    args = parser.parse_args()
    entry_counts = list(range(1, args.max_entries + 1))

    start = time.time()
    if args.source == "synthetic":
        results = run_sweep(
            entry_counts=entry_counts,
            trials_per_season=args.trials_per_season,
            num_seasons=args.num_seasons,
            seed=args.seed,
            weeks=args.weeks,
        )
        config = {
            "source": "synthetic",
            "max_entries": args.max_entries,
            "trials_per_season": args.trials_per_season,
            "num_seasons": args.num_seasons,
            "seed": args.seed,
            "weeks": args.weeks,
        }
    else:
        years = parse_years(args.years)
        seasons_by_year = (
            load_seasons_from_file(args.games_csv, years)
            if args.games_csv
            else fetch_and_load_seasons(years)
        )
        missing = sorted(set(str(y) for y in years) - set(seasons_by_year.keys()))
        if missing:
            print(f"Note: no data found for year(s) {', '.join(missing)}; skipping")

        seasons: List[List[Dict[str, MatchupProb]]] = [seasons_by_year[y] for y in sorted(seasons_by_year)]
        if not seasons:
            print("No seasons available -- nothing to sweep.")
            return

        results = run_sweep_on_seasons(
            entry_counts=entry_counts,
            seasons=seasons,
            trials_per_season=args.trials_per_season,
            seed=args.seed,
        )
        config = {
            "source": "real",
            "max_entries": args.max_entries,
            "trials_per_season": args.trials_per_season,
            "seed": args.seed,
            "years_requested": years,
            "years_used": sorted(seasons_by_year.keys()),
        }

    elapsed = time.time() - start
    _print_and_write(entry_counts, results, elapsed, config, args.out)


if __name__ == "__main__":
    main()
