from data.models import Game, Odds, Team, WinProbability
from analysis.fetch_historical_season import fetch_season_matchups, parse_years


class TestParseYears:
    def test_range(self):
        assert parse_years("2016-2025") == list(range(2016, 2026))

    def test_comma_list(self):
        assert parse_years("2020,2022,2024") == [2020, 2022, 2024]

    def test_mixed(self):
        assert parse_years("2020-2021,2023") == [2020, 2021, 2023]

    def test_single_year(self):
        assert parse_years("2024") == [2024]


def make_game(event_id, week, home, away, home_win_pct=None, away_win_pct=None, spread=None, state="post"):
    probability = None
    if home_win_pct is not None or away_win_pct is not None:
        probability = WinProbability(home_win_pct=home_win_pct, away_win_pct=away_win_pct)
    odds = Odds(spread=spread) if spread is not None else None
    return Game(
        event_id=event_id,
        competition_id=event_id,
        week=week,
        season_year=2024,
        state=state,
        home=Team(abbreviation=home, display_name=home),
        away=Team(abbreviation=away, display_name=away),
        probability=probability,
        odds=odds,
    )


class FakeESPNClient:
    def __init__(self, games_by_week):
        self.games_by_week = games_by_week

    def get_week_games(self, week=None, year=None, seasontype=None):
        return self.games_by_week.get(week, [])


class TestFetchSeasonMatchups:
    def test_extracts_win_pct_from_probability(self):
        client = FakeESPNClient({1: [make_game("1", 1, "KC", "DEN", home_win_pct=0.7, away_win_pct=0.3)]})
        season = fetch_season_matchups(client, year=2024, weeks=1)
        assert season[0]["KC"]["win_pct"] == 70.0
        assert season[0]["KC"]["opponent"] == "DEN"
        assert season[0]["DEN"]["win_pct"] == 30.0

    def test_falls_back_to_spread_when_no_probability(self):
        client = FakeESPNClient({1: [make_game("1", 1, "KC", "DEN", spread=-6.5)]})
        season = fetch_season_matchups(client, year=2024, weeks=1)
        assert season[0]["KC"]["win_pct"] > 50.0

    def test_excludes_team_with_no_probability_or_spread(self):
        client = FakeESPNClient({1: [make_game("1", 1, "KC", "DEN")]})
        season = fetch_season_matchups(client, year=2024, weeks=1)
        assert season[0] == {}

    def test_produces_one_entry_per_requested_week(self):
        client = FakeESPNClient(
            {1: [make_game("1", 1, "KC", "DEN", spread=-3)], 2: [make_game("2", 2, "SF", "DAL", spread=-3)]}
        )
        season = fetch_season_matchups(client, year=2024, weeks=2)
        assert len(season) == 2

    def test_missing_week_produces_empty_dict_not_error(self):
        client = FakeESPNClient({})
        season = fetch_season_matchups(client, year=2024, weeks=3)
        assert season == [{}, {}, {}]
