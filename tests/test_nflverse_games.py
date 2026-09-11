from analysis.nflverse_games import load_seasons_from_file, parse_seasons_from_csv_text
from analysis.synthetic_season import spread_to_win_probability

CSV_HEADER = (
    "game_id,season,game_type,week,away_team,away_score,home_team,home_score,"
    "spread_line,home_moneyline,away_moneyline\n"
)


def make_csv(rows):
    return CSV_HEADER + "\n".join(rows) + "\n"


class TestParseSeasonsFromCsvText:
    def test_home_favored_positive_spread(self):
        # spread_line positive => home favored (confirmed against real data
        # in analysis/nflverse_games.py's docstring: home_ml negative here
        # would mean home is the betting favorite, consistent).
        csv_text = make_csv(["g1,2024,REG,1,TB,24,ATL,27,2.5,-133,120"])
        seasons = parse_seasons_from_csv_text(csv_text, years=[2024])
        atl = seasons["2024"][0]["ATL"]
        assert atl.win_pct == spread_to_win_probability(2.5) * 100.0
        assert atl.win_pct > 50.0

    def test_home_underdog_negative_spread(self):
        csv_text = make_csv(["g1,2016,REG,1,CAR,20,DEN,21,-3,136,-150"])
        seasons = parse_seasons_from_csv_text(csv_text, years=[2016])
        den = seasons["2016"][0]["DEN"]
        assert den.win_pct == spread_to_win_probability(-3) * 100.0
        assert den.win_pct < 50.0

    def test_home_and_away_win_pct_sum_to_100(self):
        csv_text = make_csv(["g1,2024,REG,1,TB,24,ATL,27,2.5,-133,120"])
        seasons = parse_seasons_from_csv_text(csv_text, years=[2024])
        week = seasons["2024"][0]
        assert abs(week["ATL"].win_pct + week["TB"].win_pct - 100.0) < 1e-9

    def test_filters_by_year(self):
        csv_text = make_csv(
            ["g1,2023,REG,1,TB,24,ATL,27,2.5,-133,120", "g2,2024,REG,1,SF,10,DAL,24,-3,120,-140"]
        )
        seasons = parse_seasons_from_csv_text(csv_text, years=[2024])
        assert set(seasons.keys()) == {"2024"}

    def test_filters_by_game_type(self):
        csv_text = make_csv(
            ["g1,2024,REG,1,TB,24,ATL,27,2.5,-133,120", "g2,2024,POST,19,SF,10,DAL,24,-3,120,-140"]
        )
        seasons = parse_seasons_from_csv_text(csv_text, years=[2024])
        # Only the REG game should appear -- season should have 1 week, not 19.
        assert len(seasons["2024"]) == 1

    def test_skips_games_with_no_spread(self):
        csv_text = make_csv(["g1,2024,REG,1,TB,24,ATL,27,,-133,120"])
        seasons = parse_seasons_from_csv_text(csv_text, years=[2024])
        assert seasons == {}

    def test_season_length_follows_max_week_present(self):
        csv_text = make_csv(
            ["g1,2016,REG,1,TB,24,ATL,27,2.5,-133,120", "g2,2016,REG,17,SF,10,DAL,24,-3,120,-140"]
        )
        seasons = parse_seasons_from_csv_text(csv_text, years=[2016])
        assert len(seasons["2016"]) == 17

    def test_requested_year_absent_from_data_is_omitted(self):
        csv_text = make_csv(["g1,2024,REG,1,TB,24,ATL,27,2.5,-133,120"])
        seasons = parse_seasons_from_csv_text(csv_text, years=[1999])
        assert seasons == {}


class TestLoadSeasonsFromFile:
    def test_reads_from_disk(self, tmp_path):
        path = tmp_path / "games.csv"
        path.write_text(make_csv(["g1,2024,REG,1,TB,24,ATL,27,2.5,-133,120"]))
        seasons = load_seasons_from_file(path, years=[2024])
        assert "2024" in seasons
