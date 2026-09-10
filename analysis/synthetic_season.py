"""Generates a synthetic NFL-like season for Monte Carlo analysis.

Real ESPN win-probability/odds data only covers the current week and a
short look-ahead window (see data/espn_client.py), and this sandboxed
dev environment can't reach ESPN's API at all -- so a full-season "how
many entries should I buy" analysis needs its own season generator, not
live data.

Each team gets a random power rating; a matchup's win probability comes
from the rating difference via the standard normal-CDF spread-to-win%
conversion sports books use (P(favorite wins) = Phi(spread / sigma), with
sigma the typical standard deviation of an NFL game's margin of victory,
~13.86 points -- a commonly cited approximation). Rating spread is
calibrated so the resulting game probabilities land in a realistic NFL
range (mostly 55-80%, some near-coin-flips, occasional blowouts), not
clustered at 50% or saturated near 0/100.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from data.teams import NFL_TEAMS

DEFAULT_WEEKS = 18
RATING_STDDEV = 5.5  # points, per team; combined matchup diff ~ N(0, 7.8)
MARGIN_STDDEV = 13.86  # approx stddev of NFL game margin of victory, in points
MIN_BYE_WEEK = 4
MAX_BYE_WEEK = 14


@dataclass(frozen=True)
class MatchupProb:
    team: str
    opponent: str
    win_pct: float  # 0-100


def _win_probability(rating_diff: float) -> float:
    """Standard normal-CDF spread-to-win-probability conversion."""
    return 0.5 * (1 + math.erf(rating_diff / (MARGIN_STDDEV * math.sqrt(2))))


def generate_team_ratings(rng: random.Random, teams: List[str] = NFL_TEAMS) -> Dict[str, float]:
    return {team: rng.gauss(0.0, RATING_STDDEV) for team in teams}


def _assign_bye_weeks(rng: random.Random, weeks: int, teams: List[str]) -> Dict[str, int]:
    if weeks < MIN_BYE_WEEK + 2:
        return {team: -1 for team in teams}  # season too short for a bye window
    bye_choices = list(range(MIN_BYE_WEEK, min(MAX_BYE_WEEK, weeks) + 1))
    return {team: rng.choice(bye_choices) for team in teams}


def generate_season_schedule(
    rng: random.Random, weeks: int = DEFAULT_WEEKS, teams: List[str] = NFL_TEAMS
) -> List[List[Tuple[str, str]]]:
    """One randomized schedule: each week, pair up whichever teams aren't on
    a bye that week. Not a literal real-NFL schedule (no division
    structure, home/field balance, etc.) -- just enough structure (one bye
    per team, spread across the season) to matter for this analysis.
    """
    bye_weeks = _assign_bye_weeks(rng, weeks, teams)
    schedule: List[List[Tuple[str, str]]] = []
    for week in range(1, weeks + 1):
        playing = [t for t in teams if bye_weeks[t] != week]
        rng.shuffle(playing)
        pairs = [(playing[i], playing[i + 1]) for i in range(0, len(playing) - 1, 2)]
        schedule.append(pairs)
    return schedule


def generate_season(
    rng: random.Random, weeks: int = DEFAULT_WEEKS, teams: List[str] = NFL_TEAMS
) -> List[Dict[str, MatchupProb]]:
    """One full synthetic season: a list (index 0 = week 1) of
    ``{team_abbreviation: MatchupProb}`` for every team playing that week.
    """
    ratings = generate_team_ratings(rng, teams)
    schedule = generate_season_schedule(rng, weeks, teams)

    season: List[Dict[str, MatchupProb]] = []
    for week_pairs in schedule:
        week_probs: Dict[str, MatchupProb] = {}
        for home, away in week_pairs:
            home_win_pct = _win_probability(ratings[home] - ratings[away]) * 100.0
            week_probs[home] = MatchupProb(team=home, opponent=away, win_pct=home_win_pct)
            week_probs[away] = MatchupProb(team=away, opponent=home, win_pct=100.0 - home_win_pct)
        season.append(week_probs)
    return season
