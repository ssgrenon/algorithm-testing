"""Fetches real, completed NFL seasons from ESPN and converts them into
the per-team, per-week win-probability shape analysis/synthetic_season.py
produces -- so analysis/portfolio_simulator.py can run its Monte Carlo
sweep against real backtested seasons instead of synthetic ones.

Reuses the exact same pipeline the rest of this app uses for live
picks -- data/espn_client.py for fetching and models/win_prob.py's
resolve_team_win_probability for blending ESPN's own win-probability
field with a spread-derived fallback -- rather than a separate model.

This must run somewhere that can reach ESPN's API. This project's dev
sandbox is blocked from doing so by organization policy (confirmed: a
403 policy denial on the CONNECT tunnel to site.api.espn.com, not a
transient failure), so this is meant to run inside the
fetch-historical-season GitHub Actions workflow, which has normal
internet access -- the same way the live weekly-report workflow does.

Usage:
    python analysis/fetch_historical_season.py --years 2016-2025 --out historical_seasons.json
    python analysis/fetch_historical_season.py --years 2023,2024 --weeks 18 --out out.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

from config import CACHE_DIR, DEFAULT_SEASON_TYPE
from data.espn_client import ESPNClient
from models.win_prob import resolve_team_win_probability

logger = logging.getLogger(__name__)

# A bulk one-time historical backfill, not a recurring live poll -- a
# smaller-than-default delay between requests is still polite (this is
# not zero) while keeping a 10-season fetch tractable in one CI job.
BULK_FETCH_MIN_REQUEST_INTERVAL = 0.3


def parse_years(spec: str) -> List[int]:
    """Parses "2016-2025" or "2020,2022,2024" into a list of years."""
    years: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            years.extend(range(int(start), int(end) + 1))
        elif part:
            years.append(int(part))
    return years


def fetch_season_matchups(
    client: ESPNClient, year: int, weeks: int, seasontype: int = DEFAULT_SEASON_TYPE
) -> List[Dict[str, Dict[str, object]]]:
    """One season: a list (index 0 = week 1) of
    ``{team_abbreviation: {"opponent": ..., "win_pct": ...}}`` for every
    team with a usable win probability that week. A team missing both
    ESPN's probability field and a spread-derived fallback (bye weeks, or
    genuinely unavailable historical data) is simply absent from that
    week's dict -- same as a bye in the synthetic generator.
    """
    season: List[Dict[str, Dict[str, object]]] = []
    for week in range(1, weeks + 1):
        games = client.get_week_games(week=week, year=year, seasontype=seasontype)
        week_probs: Dict[str, Dict[str, object]] = {}
        missing = 0

        for game in games:
            for team, opponent, is_home in ((game.home, game.away, True), (game.away, game.home, False)):
                if not team.abbreviation:
                    continue
                resolved = resolve_team_win_probability(game, is_home)
                if resolved.win_pct is None:
                    missing += 1
                    continue
                week_probs[team.abbreviation] = {
                    "opponent": opponent.abbreviation,
                    "win_pct": resolved.win_pct,
                }

        logger.info(
            "  %s week %2d: %d games, %d team-weeks with a usable win probability (%d missing)",
            year, week, len(games), len(week_probs), missing,
        )
        season.append(week_probs)
    return season


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch real NFL seasons' win probabilities from ESPN.")
    parser.add_argument("--years", required=True, help='e.g. "2016-2025" or "2020,2022,2024"')
    parser.add_argument("--weeks", type=int, default=18, help="regular-season weeks to fetch per year")
    parser.add_argument("--seasontype", type=int, default=DEFAULT_SEASON_TYPE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stdout)

    years = parse_years(args.years)
    client = ESPNClient(cache_dir=CACHE_DIR, min_request_interval=BULK_FETCH_MIN_REQUEST_INTERVAL)

    seasons: Dict[str, List[Dict[str, Dict[str, object]]]] = {}
    start = time.time()
    for year in years:
        logger.info("Fetching %s season...", year)
        seasons[str(year)] = fetch_season_matchups(client, year, args.weeks, args.seasontype)
    elapsed = time.time() - start

    total_team_weeks = sum(len(week) for season in seasons.values() for week in season)
    print(f"\nFetched {len(years)} season(s) ({total_team_weeks} total team-weeks) in {elapsed / 60:.1f} min")

    args.out.write_text(json.dumps({"years": years, "weeks_per_season": args.weeks, "seasons": seasons}, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
