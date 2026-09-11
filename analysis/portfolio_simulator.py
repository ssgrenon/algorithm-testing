"""Simulates how many of N entries survive a full season under a single,
simple strategy: each week, every entry still alive gets the best
available team it hasn't used before; on a collision (two entries would
otherwise want the same team), the next-best remaining team goes to the
next entry. No same-game exclusion, no minimum win-probability floor --
this is a direct generalization of strategy/joint_optimizer.py from 2
entries to N, using exactly the algorithm described for this analysis.

Deliberately allows two of your own entries to land on opposite sides of
the same matchup. strategy/joint_optimizer.py avoided that (it was
maximizing a different objective), but for "at least one entry survives
the season," opposing picks within one game *guarantee* one of that pair
wins that week (barring a tie) -- excluding it would be a mistake here.

Assignment picks a specific team for a specific entry, but reassigning
*which* alive entry gets *which* team this week doesn't change this
week's aggregate "at least one alive" probability at all: that's
1 - product(1 - p_i) over currently-alive entries, and multiplication
doesn't care about labels. What the assignment *does* affect is which
teams are left for each entry in future weeks, since (like a real
separate survivor entry) each one tracks its own used-team history.

P(at least one of N entries survives the season) is non-decreasing in N
for a simple reason that holds regardless of assignment scheme: "at
least one of the first N-1 survives" is a subset of "at least one of N
survives," so the latter can't have lower probability. This module's
specific assignment additionally guarantees that entries 0..N-1 get the
exact same weekly team whether or not entry N exists (lower ids always
get first refusal) -- but that's a per-week assignment guarantee, not a
claim that a full multi-week trial replays identically with more
entries added: each extra alive entry consumes one more random draw per
week, which shifts later weeks' RNG stream position for everyone. See
tests/test_portfolio_simulator.py for both properties tested precisely.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from analysis.synthetic_season import MatchupProb, generate_season


def assign_week(
    alive_entry_ids: List[int],
    used_teams_by_entry: Dict[int, Set[str]],
    week_matchups: Dict[str, MatchupProb],
) -> Dict[int, Optional[MatchupProb]]:
    """Greedy assignment for one week. ``alive_entry_ids`` must be sorted --
    the lowest id gets first refusal on the best remaining eligible team,
    which is what gives the module its non-decreasing-in-N guarantee.

    An entry with no eligible team left at all (it would have had to use
    every team playing that week already -- essentially never happens
    within an 18-week season and 32 teams) gets ``None``: treated as
    eliminated, since it can't submit a valid pick.
    """
    candidates = sorted(week_matchups.values(), key=lambda m: -m.win_pct)
    assignment: Dict[int, Optional[MatchupProb]] = {entry_id: None for entry_id in alive_entry_ids}

    remaining_entries = list(alive_entry_ids)
    for matchup in candidates:
        if not remaining_entries:
            break
        for entry_id in remaining_entries:
            if matchup.team not in used_teams_by_entry[entry_id]:
                assignment[entry_id] = matchup
                remaining_entries.remove(entry_id)
                break

    return assignment


@dataclass
class TrialResult:
    entry_count: int
    weeks_survived: List[int]  # per entry (index = entry id), consecutive weeks won
    any_survived_full_season: bool


def simulate_one_trial(season: List[Dict[str, MatchupProb]], entry_count: int, rng: random.Random) -> TrialResult:
    alive: Set[int] = set(range(entry_count))
    used_teams: Dict[int, Set[str]] = {i: set() for i in range(entry_count)}
    weeks_survived: Dict[int, int] = {i: 0 for i in range(entry_count)}

    for week_matchups in season:
        if not alive:
            break
        alive_sorted = sorted(alive)
        assignment = assign_week(alive_sorted, used_teams, week_matchups)

        for entry_id in alive_sorted:
            matchup = assignment[entry_id]
            if matchup is None:
                alive.discard(entry_id)
                continue
            used_teams[entry_id].add(matchup.team)
            if rng.random() < (matchup.win_pct / 100.0):
                weeks_survived[entry_id] += 1
            else:
                alive.discard(entry_id)

    return TrialResult(
        entry_count=entry_count,
        weeks_survived=[weeks_survived[i] for i in range(entry_count)],
        any_survived_full_season=any(weeks_survived[i] == len(season) for i in range(entry_count)),
    )


@dataclass
class SweepResult:
    entry_count: int
    trials: int
    survived_trials: int
    probability: float
    standard_error: float
    avg_best_entry_weeks_survived: float


def _binomial_standard_error(p: float, n: int) -> float:
    if n == 0:
        return 0.0
    return (p * (1 - p) / n) ** 0.5


def run_sweep_on_seasons(
    entry_counts: List[int],
    seasons: List[List[Dict[str, MatchupProb]]],
    trials_per_season: int,
    seed: int = 42,
) -> Dict[int, SweepResult]:
    """Sweeps ``entry_counts`` (e.g. 1..10) against a fixed, already-built list
    of seasons -- real (fetched from ESPN) or synthetic, doesn't matter, this
    function doesn't generate seasons itself. All entry counts are evaluated
    against the *same* seasons (common random numbers), so differences across
    N reflect the strategy, not which seasons happened to be included. Each
    entry count still gets its own (reproducible) stream of game-outcome draws.
    """
    results: Dict[int, SweepResult] = {}
    for entry_count in entry_counts:
        trial_rng = random.Random(seed * 1_000_003 + entry_count)
        survived = 0
        total_trials = 0
        best_weeks_sum = 0

        for season in seasons:
            for _ in range(trials_per_season):
                trial = simulate_one_trial(season, entry_count, trial_rng)
                total_trials += 1
                if trial.any_survived_full_season:
                    survived += 1
                best_weeks_sum += max(trial.weeks_survived)

        probability = survived / total_trials
        results[entry_count] = SweepResult(
            entry_count=entry_count,
            trials=total_trials,
            survived_trials=survived,
            probability=probability,
            standard_error=_binomial_standard_error(probability, total_trials),
            avg_best_entry_weeks_survived=best_weeks_sum / total_trials,
        )

    return results


def run_sweep(
    entry_counts: List[int],
    trials_per_season: int,
    num_seasons: int,
    seed: int = 42,
    weeks: int = 18,
) -> Dict[int, SweepResult]:
    """Sweeps ``entry_counts`` against ``num_seasons`` freshly-generated
    synthetic seasons (common random numbers across entry counts -- see
    run_sweep_on_seasons). For a real, ESPN-backtested equivalent, fetch
    seasons with analysis/fetch_historical_season.py and load them with
    analysis/real_season.py's load_real_season(), then call
    run_sweep_on_seasons() directly.
    """
    season_rng = random.Random(seed)
    seasons = [generate_season(season_rng, weeks=weeks) for _ in range(num_seasons)]
    return run_sweep_on_seasons(entry_counts, seasons, trials_per_season, seed)
