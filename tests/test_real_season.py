import json

from analysis.real_season import load_real_seasons
from analysis.synthetic_season import MatchupProb


def test_load_real_seasons_converts_to_matchup_prob(tmp_path):
    payload = {
        "years": [2024],
        "weeks_per_season": 2,
        "seasons": {
            "2024": [
                {"KC": {"opponent": "DEN", "win_pct": 78.0}, "DEN": {"opponent": "KC", "win_pct": 22.0}},
                {"SF": {"opponent": "DAL", "win_pct": 60.0}, "DAL": {"opponent": "SF", "win_pct": 40.0}},
            ]
        },
    }
    path = tmp_path / "seasons.json"
    path.write_text(json.dumps(payload))

    result = load_real_seasons(path)

    assert set(result.keys()) == {"2024"}
    season = result["2024"]
    assert len(season) == 2
    assert season[0]["KC"] == MatchupProb(team="KC", opponent="DEN", win_pct=78.0)
    assert season[1]["DAL"] == MatchupProb(team="DAL", opponent="SF", win_pct=40.0)


def test_load_multiple_seasons(tmp_path):
    payload = {
        "years": [2023, 2024],
        "weeks_per_season": 1,
        "seasons": {
            "2023": [{"KC": {"opponent": "DEN", "win_pct": 70.0}, "DEN": {"opponent": "KC", "win_pct": 30.0}}],
            "2024": [{"SF": {"opponent": "DAL", "win_pct": 55.0}, "DAL": {"opponent": "SF", "win_pct": 45.0}}],
        },
    }
    path = tmp_path / "seasons.json"
    path.write_text(json.dumps(payload))

    result = load_real_seasons(path)
    assert set(result.keys()) == {"2023", "2024"}
    assert result["2023"][0]["KC"].win_pct == 70.0
    assert result["2024"][0]["SF"].win_pct == 55.0
