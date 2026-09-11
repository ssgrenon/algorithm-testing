"""Loads seasons fetched by analysis/fetch_historical_season.py into the
same per-week ``{team: MatchupProb}`` shape analysis/synthetic_season.py
produces, so analysis/portfolio_simulator.py's run_sweep_on_seasons() runs
identically against real or synthetic data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from analysis.synthetic_season import MatchupProb


def load_real_seasons(path: Path) -> Dict[str, List[Dict[str, MatchupProb]]]:
    """Returns ``{year_str: season}`` for every season in the fetched file,
    where ``season`` is a list (index 0 = week 1) of
    ``{team_abbreviation: MatchupProb}``.
    """
    payload = json.loads(Path(path).read_text())
    result: Dict[str, List[Dict[str, MatchupProb]]] = {}

    for year, weeks_data in payload["seasons"].items():
        season: List[Dict[str, MatchupProb]] = []
        for week_data in weeks_data:
            week_probs = {
                team: MatchupProb(team=team, opponent=info["opponent"], win_pct=info["win_pct"])
                for team, info in week_data.items()
            }
            season.append(week_probs)
        result[year] = season

    return result
