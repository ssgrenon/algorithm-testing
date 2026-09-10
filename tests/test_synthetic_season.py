import random

from analysis.synthetic_season import (
    generate_season,
    generate_season_schedule,
    generate_team_ratings,
)
from data.teams import NFL_TEAMS


class TestGenerateTeamRatings:
    def test_one_rating_per_team(self):
        ratings = generate_team_ratings(random.Random(1))
        assert set(ratings.keys()) == set(NFL_TEAMS)

    def test_deterministic_given_seed(self):
        a = generate_team_ratings(random.Random(7))
        b = generate_team_ratings(random.Random(7))
        assert a == b

    def test_different_seeds_differ(self):
        a = generate_team_ratings(random.Random(1))
        b = generate_team_ratings(random.Random(2))
        assert a != b


class TestGenerateSeasonSchedule:
    def test_no_team_plays_itself_or_twice_in_a_week(self):
        schedule = generate_season_schedule(random.Random(3), weeks=18)
        for week_pairs in schedule:
            teams_this_week = [team for pair in week_pairs for team in pair]
            assert len(teams_this_week) == len(set(teams_this_week))
            for home, away in week_pairs:
                assert home != away

    def test_produces_requested_number_of_weeks(self):
        schedule = generate_season_schedule(random.Random(4), weeks=18)
        assert len(schedule) == 18

    def test_every_team_gets_roughly_one_bye(self):
        schedule = generate_season_schedule(random.Random(5), weeks=18)
        games_played = {team: 0 for team in NFL_TEAMS}
        for week_pairs in schedule:
            for home, away in week_pairs:
                games_played[home] += 1
                games_played[away] += 1
        # Every team should play most weeks (17 or 18 games out of 18 weeks).
        for team, count in games_played.items():
            assert 16 <= count <= 18


class TestGenerateSeason:
    def test_win_pct_pairs_sum_to_100(self):
        season = generate_season(random.Random(9), weeks=6)
        for week_probs in season:
            for team, matchup in week_probs.items():
                opponent_matchup = week_probs[matchup.opponent]
                assert abs(matchup.win_pct + opponent_matchup.win_pct - 100.0) < 1e-9

    def test_win_pct_within_bounds(self):
        season = generate_season(random.Random(11), weeks=18)
        for week_probs in season:
            for matchup in week_probs.values():
                assert 0.0 < matchup.win_pct < 100.0

    def test_deterministic_given_seed(self):
        a = generate_season(random.Random(21), weeks=4)
        b = generate_season(random.Random(21), weeks=4)
        assert a == b

    def test_realistic_spread_of_probabilities(self):
        # Sanity-check the calibration: most games shouldn't be coin flips,
        # and blowouts (>90%) should be rare, not the norm.
        season = generate_season(random.Random(123), weeks=18)
        all_probs = [m.win_pct for week in season for m in week.values()]
        near_even = sum(1 for p in all_probs if 48 <= p <= 52)
        blowouts = sum(1 for p in all_probs if p >= 90)
        assert near_even / len(all_probs) < 0.5
        assert blowouts / len(all_probs) < 0.2
