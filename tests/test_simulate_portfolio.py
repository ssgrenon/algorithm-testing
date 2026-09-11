import sys

CSV_HEADER = (
    "game_id,season,game_type,week,away_team,away_score,home_team,home_score,"
    "spread_line,home_moneyline,away_moneyline\n"
)


def make_small_games_csv(tmp_path):
    # Two teams, two weeks, so there's at least one valid non-colliding pair
    # of picks across two entries.
    rows = [
        "g1,2024,REG,1,DEN,20,KC,27,-6.5,-260,220",
        "g2,2024,REG,1,DAL,10,SF,24,-5,-210,180",
        "g3,2024,REG,2,LV,17,KC,24,-6,-250,210",
        "g4,2024,REG,2,SEA,14,SF,27,-6,-250,210",
    ]
    path = tmp_path / "games.csv"
    path.write_text(CSV_HEADER + "\n".join(rows) + "\n")
    return path


class TestSimulatePortfolioRealSource:
    def test_runs_end_to_end_against_a_local_csv(self, tmp_path, monkeypatch, capsys):
        csv_path = make_small_games_csv(tmp_path)
        out_path = tmp_path / "results.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "simulate_portfolio.py",
                "--source",
                "real",
                "--years",
                "2024",
                "--games-csv",
                str(csv_path),
                "--max-entries",
                "2",
                "--trials-per-season",
                "5",
                "--out",
                str(out_path),
            ],
        )
        import simulate_portfolio

        simulate_portfolio.main()

        assert out_path.exists()
        output = capsys.readouterr().out
        assert "total trials" in output

    def test_missing_year_is_reported_not_crashed(self, tmp_path, monkeypatch, capsys):
        csv_path = make_small_games_csv(tmp_path)
        out_path = tmp_path / "results.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "simulate_portfolio.py",
                "--source",
                "real",
                "--years",
                "1999,2024",
                "--games-csv",
                str(csv_path),
                "--max-entries",
                "1",
                "--trials-per-season",
                "5",
                "--out",
                str(out_path),
            ],
        )
        import simulate_portfolio

        simulate_portfolio.main()
        assert "no data found for year(s) 1999" in capsys.readouterr().out
