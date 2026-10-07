from data.models import Game, Odds, Team, WinProbability
from data.teams import NFL_TEAMS
from models.win_prob import TeamWeekWinProbability, build_win_probability_table
from report import WeeklyReport, compute_held_back_teams, fetch_pipeline_games, remaining_pool, render_html, render_text
from strategy.joint_optimizer import JointRecommendation, TeamOption


def make_game(
    event_id="1",
    week=3,
    state="pre",
    home_abbr="KC",
    away_abbr="DEN",
    home_win_pct=None,
    away_win_pct=None,
    spread=None,
    spread_details=None,
):
    probability = None
    if home_win_pct is not None or away_win_pct is not None:
        probability = WinProbability(home_win_pct=home_win_pct, away_win_pct=away_win_pct)
    odds = Odds(spread=spread, details=spread_details) if spread is not None else None
    return Game(
        event_id=event_id,
        competition_id=event_id,
        week=week,
        season_year=2026,
        state=state,
        home=Team(abbreviation=home_abbr, display_name=home_abbr),
        away=Team(abbreviation=away_abbr, display_name=away_abbr),
        probability=probability,
        odds=odds,
    )


class TestRemainingPool:
    def test_excludes_used_teams(self):
        remaining = remaining_pool(["KC", "BUF"])
        assert "KC" not in remaining
        assert "BUF" not in remaining
        assert len(remaining) == len(NFL_TEAMS) - 2

    def test_no_used_teams_returns_full_list(self):
        assert remaining_pool([]) == NFL_TEAMS

    def test_preserves_nfl_teams_order(self):
        remaining = remaining_pool(["KC"])
        assert remaining == [t for t in NFL_TEAMS if t != "KC"]


class FakeESPNClient:
    """Stub client for testing the fetch-orchestration logic without any
    network I/O -- returns canned games per requested week."""

    def __init__(self, games_by_week):
        self.games_by_week = games_by_week
        self.requested_weeks = []

    def get_week_games(self, week=None, seasontype=None):
        self.requested_weeks.append(week)
        return self.games_by_week.get(week, [])


class TestFetchPipelineGames:
    def test_fetches_current_and_lookahead_weeks(self):
        client = FakeESPNClient(
            {
                None: [make_game(week=5, home_win_pct=0.6, away_win_pct=0.4)],
                6: [make_game(event_id="2", week=6)],
                7: [make_game(event_id="3", week=7)],
            }
        )
        current_week, current_games, all_games = fetch_pipeline_games(client, week=None, lookahead_weeks=2)

        assert current_week == 5
        assert len(current_games) == 1
        assert client.requested_weeks == [None, 6, 7]
        assert len(all_games) == 3

    def test_explicit_week_is_used_as_current_week(self):
        client = FakeESPNClient({4: [make_game(week=4)], 5: [make_game(event_id="2", week=5)]})
        current_week, current_games, all_games = fetch_pipeline_games(client, week=4, lookahead_weeks=1)
        assert current_week == 4
        assert client.requested_weeks == [4, 5]

    def test_no_current_week_games_returns_empty(self):
        client = FakeESPNClient({})
        current_week, current_games, all_games = fetch_pipeline_games(client, week=None, lookahead_weeks=3)
        assert current_week is None
        assert current_games == []
        assert all_games == []

    def test_unknown_week_number_skips_lookahead_fetch(self):
        # ESPN returned games but no week number in the payload.
        client = FakeESPNClient({None: [make_game(week=None)]})
        current_week, current_games, all_games = fetch_pipeline_games(client, week=None, lookahead_weeks=3)
        assert current_week is None
        assert client.requested_weeks == [None]  # no follow-up fetches attempted


def wp(team, week, win_pct):
    return TeamWeekWinProbability(
        team_abbreviation=team,
        week=week,
        season_year=2026,
        opponent_abbreviation="OPP",
        is_home=True,
        win_pct=win_pct,
        source="api",
    )


class TestComputeHeldBackTeams:
    def test_includes_team_with_better_future_matchup(self):
        current_games = [make_game(event_id="1", home_abbr="BUF", away_abbr="NYJ", home_win_pct=0.5, away_win_pct=0.5)]
        win_prob_table = build_win_probability_table(current_games)
        win_prob_table[("BUF", 4)] = wp("BUF", 4, 95.0)

        held_back = compute_held_back_teams(
            current_games, win_prob_table, current_week=3, used_teams_a=[], used_teams_b=[], picked_teams=set()
        )

        assert any(h.team_abbreviation == "BUF" for h in held_back)
        buf = next(h for h in held_back if h.team_abbreviation == "BUF")
        assert buf.best_future_week == 4
        assert buf.future_value > 0

    def test_excludes_picked_teams(self):
        current_games = [make_game(home_abbr="KC", away_abbr="DEN", home_win_pct=0.5, away_win_pct=0.5)]
        win_prob_table = build_win_probability_table(current_games)
        win_prob_table[("KC", 4)] = wp("KC", 4, 99.0)

        held_back = compute_held_back_teams(
            current_games,
            win_prob_table,
            current_week=3,
            used_teams_a=[],
            used_teams_b=[],
            picked_teams={"KC"},
        )
        assert all(h.team_abbreviation != "KC" for h in held_back)

    def test_excludes_teams_used_by_both_entries(self):
        current_games = [make_game(home_abbr="KC", away_abbr="DEN", home_win_pct=0.5, away_win_pct=0.5)]
        win_prob_table = build_win_probability_table(current_games)
        win_prob_table[("KC", 4)] = wp("KC", 4, 99.0)

        held_back = compute_held_back_teams(
            current_games,
            win_prob_table,
            current_week=3,
            used_teams_a=["KC"],
            used_teams_b=["KC"],
            picked_teams=set(),
        )
        assert all(h.team_abbreviation != "KC" for h in held_back)

    def test_excludes_team_with_no_better_future_matchup(self):
        current_games = [make_game(home_abbr="KC", away_abbr="DEN", home_win_pct=0.9, away_win_pct=0.1)]
        win_prob_table = build_win_probability_table(current_games)
        win_prob_table[("KC", 4)] = wp("KC", 4, 40.0)  # much worse than this week

        held_back = compute_held_back_teams(
            current_games, win_prob_table, current_week=3, used_teams_a=[], used_teams_b=[], picked_teams=set()
        )
        assert all(h.team_abbreviation != "KC" for h in held_back)

    def test_sorted_by_future_value_descending(self):
        current_games = [
            make_game(event_id="1", home_abbr="BUF", away_abbr="NYJ", home_win_pct=0.5, away_win_pct=0.5),
            make_game(event_id="2", home_abbr="SF", away_abbr="DAL", home_win_pct=0.5, away_win_pct=0.5),
        ]
        win_prob_table = build_win_probability_table(current_games)
        win_prob_table[("BUF", 4)] = wp("BUF", 4, 70.0)
        win_prob_table[("SF", 4)] = wp("SF", 4, 95.0)

        held_back = compute_held_back_teams(
            current_games, win_prob_table, current_week=3, used_teams_a=[], used_teams_b=[], picked_teams=set()
        )
        held_back_abbrevs = [h.team_abbreviation for h in held_back]
        assert held_back_abbrevs.index("SF") < held_back_abbrevs.index("BUF")


def team_option(team, opponent="OPP", win_pct=80.0):
    return TeamOption(
        team_abbreviation=team,
        opponent_abbreviation=opponent,
        event_id="1",
        win_pct=win_pct,
        win_pct_source="api",
        spread_detail=None,
    )


def joint_rec(pick_a_team, pick_b_team, reasoning="because reasons", field_aware=False):
    pick_a = team_option(pick_a_team) if pick_a_team else None
    pick_b = team_option(pick_b_team) if pick_b_team else None
    return JointRecommendation(
        week=3,
        pick_a=pick_a,
        pick_b=pick_b,
        both_survive_pct=60.0 if pick_a and pick_b else None,
        one_survives_pct=35.0 if pick_a and pick_b else None,
        both_eliminated_pct=5.0 if pick_a and pick_b else None,
        reasoning=reasoning,
        field_aware=field_aware,
    )


def make_report(rec, field_rec=None):
    return WeeklyReport(
        week=3,
        joint_rec=rec,
        used_teams_a=[],
        used_teams_b=[],
        remaining_a=["KC", "SF"],
        remaining_b=["KC", "SF"],
        held_back=[],
        lookahead_weeks=3,
        week_number_known=True,
        joint_rec_field_aware=field_rec,
    )


class TestRenderTextFieldAware:
    def test_no_field_section_when_field_aware_rec_absent(self):
        text = render_text(make_report(joint_rec("KC", "SF")))
        assert "RECOMMENDED PICKS (joint optimizer)" in text
        assert "IGNORING THE FIELD" not in text
        assert "CONSIDERING THE FIELD" not in text

    def test_shows_both_sections_with_same_picks_note_when_picks_match(self):
        rec = joint_rec("KC", "SF")
        field_rec = joint_rec("KC", "SF", reasoning="field-aware reasoning", field_aware=True)
        text = render_text(make_report(rec, field_rec))

        assert "IGNORING THE FIELD" in text
        assert "CONSIDERING THE FIELD" in text
        assert "field-aware reasoning" in text
        assert "Same picks either way" in text

    def test_shows_differs_note_when_picks_diverge(self):
        rec = joint_rec("KC", "SF")
        field_rec = joint_rec("BUF", "SF", field_aware=True)
        text = render_text(make_report(rec, field_rec))
        assert "Differs from the field-blind recommendation above" in text


class TestRenderHtmlFieldAware:
    def test_no_field_section_when_field_aware_rec_absent(self):
        html = render_html(make_report(joint_rec("KC", "SF")))
        assert "considering the field" not in html
        assert "ignoring the field" not in html

    def test_shows_both_sections_with_same_picks_note_when_picks_match(self):
        rec = joint_rec("KC", "SF")
        field_rec = joint_rec("KC", "SF", reasoning="field-aware reasoning", field_aware=True)
        html = render_html(make_report(rec, field_rec))

        assert "ignoring the field" in html
        assert "considering the field" in html
        assert "field-aware reasoning" in html
        assert "Same picks either way" in html

    def test_shows_differs_note_when_picks_diverge(self):
        rec = joint_rec("KC", "SF")
        field_rec = joint_rec("BUF", "SF", field_aware=True)
        html = render_html(make_report(rec, field_rec))
        assert "Differs from the field-blind recommendation above" in html

    def test_none_report_renders_fallback_without_error(self):
        html = render_html(None)
        assert "No game data was available" in html
