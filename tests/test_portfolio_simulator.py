import random

from analysis.portfolio_simulator import (
    assign_week,
    run_sweep,
    simulate_one_trial,
)
from analysis.synthetic_season import MatchupProb


def matchup(team, opponent, win_pct):
    return MatchupProb(team=team, opponent=opponent, win_pct=win_pct)


def make_week(*matchups):
    week = {}
    for m in matchups:
        week[m.team] = m
    return week


class TestAssignWeek:
    def test_gives_distinct_teams_to_distinct_entries(self):
        week = make_week(
            matchup("KC", "DEN", 90.0), matchup("DEN", "KC", 10.0),
            matchup("SF", "DAL", 75.0), matchup("DAL", "SF", 25.0),
        )
        assignment = assign_week([0, 1], {0: set(), 1: set()}, week)
        assert assignment[0].team != assignment[1].team
        assert assignment[0].team == "KC"  # best available goes to entry 0 first
        assert assignment[1].team == "SF"

    def test_respects_used_teams_per_entry(self):
        week = make_week(matchup("KC", "DEN", 90.0), matchup("DEN", "KC", 10.0))
        assignment = assign_week([0, 1], {0: {"KC"}, 1: set()}, week)
        assert assignment[0].team == "DEN"
        assert assignment[1].team == "KC"

    def test_allows_opposing_sides_of_same_game(self):
        # Deliberately NOT excluded -- see module docstring.
        week = make_week(matchup("KC", "DEN", 90.0), matchup("DEN", "KC", 10.0))
        assignment = assign_week([0, 1], {0: set(), 1: {"KC"}}, week)
        assert assignment[0].team == "KC"
        assert assignment[1].team == "DEN"

    def test_entry_with_no_eligible_team_gets_none(self):
        week = make_week(matchup("KC", "DEN", 90.0), matchup("DEN", "KC", 10.0))
        assignment = assign_week([0], {0: {"KC", "DEN"}}, week)
        assert assignment[0] is None

    def test_lower_entry_id_gets_first_refusal(self):
        week = make_week(
            matchup("KC", "DEN", 90.0), matchup("DEN", "KC", 10.0),
            matchup("SF", "DAL", 75.0), matchup("DAL", "SF", 25.0),
        )
        assignment = assign_week([0, 1, 2], {0: set(), 1: set(), 2: set()}, week)
        assert assignment[0].team == "KC"
        assert assignment[1].team == "SF"
        assert assignment[2].team == "DAL"  # best remaining after KC/SF taken (25% > DEN's 10%)


def make_certain_win_season(weeks, team_prefix="W"):
    """A season where a fresh, guaranteed-winning team is available every
    week (different team each week, since a real entry can't reuse one).
    """
    season = []
    for week in range(weeks):
        favorite, underdog = f"{team_prefix}{week}A", f"{team_prefix}{week}B"
        season.append(make_week(matchup(favorite, underdog, 100.0), matchup(underdog, favorite, 0.0)))
    return season


class TestSimulateOneTrial:
    def test_certain_win_survives_whole_season(self):
        season = make_certain_win_season(3)
        trial = simulate_one_trial(season, entry_count=1, rng=random.Random(1))
        assert trial.weeks_survived == [3]
        assert trial.any_survived_full_season is True

    def test_certain_loss_eliminated_week_one(self):
        # Only a guaranteed-loss team is available -- entry 0 has no choice.
        season = [make_week(matchup("DEN", "KC", 0.0), matchup("KC", "DEN", 100.0)) for _ in range(1)]
        trial = simulate_one_trial(season, entry_count=1, rng=random.Random(1))
        # entry 0 gets first refusal, so it actually gets the 100% team (KC) --
        # use two entries and read entry 1 (which is forced onto the loser).
        trial_two_entries = simulate_one_trial(season, entry_count=2, rng=random.Random(1))
        assert trial_two_entries.weeks_survived[1] == 0

    def test_any_survived_true_if_at_least_one_entry_goes_the_distance(self):
        # Entry 0 always gets first refusal on that week's guaranteed winner.
        season = make_certain_win_season(2)
        trial = simulate_one_trial(season, entry_count=2, rng=random.Random(1))
        assert trial.weeks_survived[0] == 2
        assert trial.any_survived_full_season is True

    def test_adding_an_entry_never_changes_which_team_earlier_entries_get(self):
        # assign_week's actual guarantee: entries 0..N-1 get the identical
        # team assignment whether or not entry N exists (lower ids always
        # get first refusal). This is *not* the same as claiming a full
        # multi-week trial replays identically with more entries added --
        # each extra alive entry consumes one more rng.random() draw per
        # week, which shifts later weeks' RNG stream position even for
        # entries whose assignment didn't change. That's why this checks
        # assign_week directly rather than simulate_one_trial's outcome.
        week = make_week(
            matchup("KC", "DEN", 70.0), matchup("DEN", "KC", 30.0),
            matchup("SF", "DAL", 60.0), matchup("DAL", "SF", 40.0),
            matchup("BUF", "MIA", 55.0), matchup("MIA", "BUF", 45.0),
        )
        assignment_two = assign_week([0, 1], {0: set(), 1: set()}, week)
        assignment_three = assign_week([0, 1, 2], {0: set(), 1: set(), 2: set()}, week)
        assert assignment_two[0].team == assignment_three[0].team
        assert assignment_two[1].team == assignment_three[1].team


class TestRunSweep:
    def test_probability_is_monotonically_non_decreasing_in_entry_count(self):
        results = run_sweep(entry_counts=[1, 2, 3, 5, 8], trials_per_season=20, num_seasons=5, seed=7, weeks=8)
        probs = [results[n].probability for n in [1, 2, 3, 5, 8]]
        assert probs == sorted(probs)

    def test_results_cover_every_requested_entry_count(self):
        results = run_sweep(entry_counts=[1, 4, 10], trials_per_season=5, num_seasons=2, seed=1, weeks=6)
        assert set(results.keys()) == {1, 4, 10}

    def test_probability_within_bounds(self):
        results = run_sweep(entry_counts=[1, 5], trials_per_season=20, num_seasons=3, seed=2, weeks=8)
        for result in results.values():
            assert 0.0 <= result.probability <= 1.0
            assert result.trials == 20 * 3

    def test_same_seed_is_reproducible(self):
        a = run_sweep(entry_counts=[1, 3], trials_per_season=10, num_seasons=2, seed=42, weeks=6)
        b = run_sweep(entry_counts=[1, 3], trials_per_season=10, num_seasons=2, seed=42, weeks=6)
        assert a[1].probability == b[1].probability
        assert a[3].probability == b[3].probability
