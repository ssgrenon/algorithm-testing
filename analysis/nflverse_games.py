"""Loads real NFL game results and closing spreads from nflverse/nfldata's
public ``games.csv`` (https://github.com/nflverse/nfldata), converting
them into the same per-week ``{team: MatchupProb}`` shape
analysis/synthetic_season.py produces -- so
analysis/portfolio_simulator.py's ``run_sweep_on_seasons`` runs
identically against this real, backtested data.

Unlike ESPN's site.api.espn.com (blocked by organization policy from this
dev environment -- see analysis/fetch_historical_season.py's docstring),
this file is plain public GitHub content and is directly reachable here;
no GitHub Actions relay is needed for it.

Win probability comes from the closing ``spread_line`` via the same
normal-CDF conversion (spread -> win%) analysis/synthetic_season.py uses
for its own synthetic ratings, applied here to a real number instead of a
generated one. ``spread_line`` is the home team's expected margin
(positive = home favored) -- confirmed against this file's own
home/away moneylines before use (e.g. a game with spread_line=-3 and
home_moneyline=+136 -- a positive, underdog-shaped moneyline -- confirms
negative spread_line means the home team is the underdog), not assumed.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Dict, List

import requests

from analysis.synthetic_season import MatchupProb, spread_to_win_probability

GAMES_CSV_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
DEFAULT_GAME_TYPE = "REG"


def fetch_games_csv(url: str = GAMES_CSV_URL, timeout: int = 30) -> str:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def parse_seasons_from_csv_text(
    csv_text: str, years: List[int], game_type: str = DEFAULT_GAME_TYPE
) -> Dict[str, List[Dict[str, MatchupProb]]]:
    """Returns ``{year_str: season}`` for each requested year found in
    ``csv_text``, where ``season`` is a list (index 0 = week 1) of
    ``{team_abbreviation: MatchupProb}``. A game with no closing spread
    is skipped (both teams excluded from that week, same as a bye
    elsewhere in this project). Season length follows the real schedule
    (17 weeks for 2016-2020, 18 from 2021 on) rather than a fixed number.
    Years requested but absent from the data are simply not in the result.
    """
    wanted_years = {str(y) for y in years}
    weeks_by_year: Dict[str, Dict[int, Dict[str, MatchupProb]]] = {}

    for row in csv.DictReader(io.StringIO(csv_text)):
        if row["season"] not in wanted_years or row["game_type"] != game_type:
            continue
        spread_raw = row.get("spread_line")
        if not spread_raw:
            continue

        week = int(row["week"])
        home, away = row["home_team"], row["away_team"]
        home_spread = float(spread_raw)  # home team's expected margin; positive = home favored
        home_win_pct = spread_to_win_probability(home_spread) * 100.0

        week_probs = weeks_by_year.setdefault(row["season"], {}).setdefault(week, {})
        week_probs[home] = MatchupProb(team=home, opponent=away, win_pct=home_win_pct)
        week_probs[away] = MatchupProb(team=away, opponent=home, win_pct=100.0 - home_win_pct)

    result: Dict[str, List[Dict[str, MatchupProb]]] = {}
    for year, weeks in weeks_by_year.items():
        max_week = max(weeks)
        result[year] = [weeks.get(w, {}) for w in range(1, max_week + 1)]
    return result


def load_seasons_from_file(
    path: Path, years: List[int], game_type: str = DEFAULT_GAME_TYPE
) -> Dict[str, List[Dict[str, MatchupProb]]]:
    return parse_seasons_from_csv_text(Path(path).read_text(), years, game_type)


def fetch_and_load_seasons(
    years: List[int], game_type: str = DEFAULT_GAME_TYPE, url: str = GAMES_CSV_URL
) -> Dict[str, List[Dict[str, MatchupProb]]]:
    return parse_seasons_from_csv_text(fetch_games_csv(url), years, game_type)
