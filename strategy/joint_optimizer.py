"""Joint pick optimizer: choose Entry A's and Entry B's picks together.

Unlike ``entry_a_value.py`` (optimizes Entry A alone) and
``entry_b_hedge.py`` (hedges Entry B against an already-fixed Entry A
pick), this module searches every valid ``(team_a, team_b)`` pair at once
and picks the one that maximizes:

    P(A wins) + P(B wins) - P(A loses AND B loses)

assuming independence between different games. The pair is always required
to come from two different games (see the constraints below), so that
independence assumption holds by construction -- we never need to reason
about two teams whose outcomes are perfectly correlated because they're
playing each other.

A pair is only considered if it:
  * never repeats a team either entry has already used this season
  * never picks the same team for both entries in the same week
  * never puts the two entries on opposing sides of the same game (since a
    game only has two teams, this is equivalent to: never pick two teams
    from the same game at all)
  * keeps Entry B's win probability at or above a configurable floor
    (default 65%), so B doesn't sacrifice safety purely to diversify away
    from A -- unless nothing clears the floor, in which case the floor is
    relaxed and that's called out in the reasoning rather than silently
    leaving Entry B without a pick.

The search space is small (at most ~2x the number of games per entry), so
this is a plain brute-force scan over all pairs, not an ILP/solver.

Optionally field-aware: pass a ``FieldModel`` (data/field_picks.py, built
from other pool participants' prior picks) and the ranking additionally
penalizes picks most of the field can still also make -- since the pot
splits among winners, surviving alongside a huge chunk of the field is
worth less than surviving in a small group. This never changes a pick's
*true* win/survival probabilities (those depend only on the game, not on
who else picks the same team) -- it only changes which pair ranks best.
Omitting ``field_model`` reproduces the original (field-blind) ranking
exactly, so existing callers are unaffected.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional

from data.field_picks import FieldModel
from data.models import Game
from models.win_prob import resolve_team_win_probability
from state.entries_store import load_used_teams_for_entry
from strategy.entry_b_hedge import DEFAULT_MIN_WIN_PROB_FLOOR as DEFAULT_MIN_WIN_PROB_FLOOR_B
from strategy.entry_b_hedge import meets_win_prob_floor

ENTRY_A_NAME = "Entry A"
ENTRY_B_NAME = "Entry B"

# How many objective-score points a pick loses for every 100% of the field
# that still has it available. 0.3 is deliberately mild: at the extremes
# (a team the whole field still has vs. one nobody else has), the gap is
# comparable to swapping from an ~85% favorite to an ~70% one -- noticeable,
# but it won't make the optimizer throw away a true survival edge chasing
# differentiation alone.
DEFAULT_FIELD_WEIGHT = 0.3


@dataclass
class TeamOption:
    team_abbreviation: str
    opponent_abbreviation: Optional[str]
    event_id: Optional[str]
    win_pct: Optional[float]  # 0-100, may be spread-estimated
    win_pct_source: str  # "api" | "spread_estimate" | "unknown"
    spread_detail: Optional[str]


@dataclass
class PairScore:
    pick_a: TeamOption
    pick_b: TeamOption
    both_survive_pct: float
    one_survives_pct: float
    both_eliminated_pct: float
    objective_score: float  # what ranking uses -- equals raw_objective_score unless field-aware
    raw_objective_score: float  # the field-blind P(A)+P(B)-P(both lose) objective, always present
    field_availability_a: Optional[float] = None  # 0-1, fraction of the field that could still pick A's team
    field_availability_b: Optional[float] = None


@dataclass
class JointSearchResult:
    best: Optional[PairScore]
    runner_up: Optional[PairScore]
    floor_relaxed: bool
    pairs_considered: int


@dataclass
class JointRecommendation:
    week: int
    pick_a: Optional[TeamOption]
    pick_b: Optional[TeamOption]
    both_survive_pct: Optional[float]
    one_survives_pct: Optional[float]
    both_eliminated_pct: Optional[float]
    reasoning: str
    floor_relaxed: bool = False
    pairs_considered: int = 0
    field_aware: bool = False
    field_availability_a: Optional[float] = None
    field_availability_b: Optional[float] = None


def build_team_options(current_week_games: List[Game]) -> List[TeamOption]:
    """All teams playing a not-yet-started game this week, with win prob/spread."""
    options: List[TeamOption] = []
    for game in current_week_games:
        if game.state and game.state != "pre":
            continue
        spread_detail = game.odds.details if game.odds else None
        for team, opponent, is_home in ((game.home, game.away, True), (game.away, game.home, False)):
            if not team.abbreviation:
                continue
            resolved = resolve_team_win_probability(game, is_home)
            options.append(
                TeamOption(
                    team_abbreviation=team.abbreviation,
                    opponent_abbreviation=opponent.abbreviation,
                    event_id=game.event_id,
                    win_pct=resolved.win_pct,
                    win_pct_source=resolved.source,
                    spread_detail=spread_detail,
                )
            )
    return options


def _score_pair(a: TeamOption, b: TeamOption) -> PairScore:
    p_a = a.win_pct / 100.0
    p_b = b.win_pct / 100.0
    both_survive = p_a * p_b
    both_eliminated = (1 - p_a) * (1 - p_b)
    one_survives = 1.0 - both_survive - both_eliminated
    objective = p_a + p_b - both_eliminated
    return PairScore(
        pick_a=a,
        pick_b=b,
        both_survive_pct=both_survive * 100.0,
        one_survives_pct=one_survives * 100.0,
        both_eliminated_pct=both_eliminated * 100.0,
        objective_score=objective,
        raw_objective_score=objective,
    )


def _apply_field_awareness(pair: PairScore, field_model: FieldModel, field_weight: float) -> PairScore:
    """Returns a copy of ``pair`` with objective_score penalized by how much
    of the field can still make each pick -- true survival probabilities
    (both_survive_pct etc.) are untouched, since those don't depend on the
    field at all; only which pair ranks best changes.
    """
    availability_a = field_model.availability_fraction(pair.pick_a.team_abbreviation)
    availability_b = field_model.availability_fraction(pair.pick_b.team_abbreviation)
    crowding_penalty = field_weight * (availability_a + availability_b)
    return replace(
        pair,
        objective_score=pair.raw_objective_score - crowding_penalty,
        field_availability_a=availability_a,
        field_availability_b=availability_b,
    )


def find_best_pair(
    current_week_games: List[Game],
    used_teams_a: List[str],
    used_teams_b: List[str],
    min_win_prob_floor_b: float = DEFAULT_MIN_WIN_PROB_FLOOR_B,
    field_model: Optional[FieldModel] = None,
    field_weight: float = DEFAULT_FIELD_WEIGHT,
) -> JointSearchResult:
    """Pure brute-force search over all constraint-satisfying (team_a, team_b) pairs.

    Pass ``field_model`` to additionally penalize picks most of the field
    can still also make (see module docstring) -- omit it for the
    original, field-blind ranking.
    """
    options = build_team_options(current_week_games)

    available_a = [o for o in options if o.team_abbreviation not in used_teams_a and o.win_pct is not None]
    available_b_all = [o for o in options if o.team_abbreviation not in used_teams_b and o.win_pct is not None]
    available_b = [o for o in available_b_all if meets_win_prob_floor(o.win_pct, min_win_prob_floor_b)]

    floor_relaxed = False
    if not available_b and available_b_all:
        available_b = available_b_all
        floor_relaxed = True

    scored: List[PairScore] = []
    for a in available_a:
        for b in available_b:
            if a.team_abbreviation == b.team_abbreviation:
                continue
            if a.event_id is not None and a.event_id == b.event_id:
                continue  # same game -- opposing sides
            pair = _score_pair(a, b)
            if field_model is not None:
                pair = _apply_field_awareness(pair, field_model, field_weight)
            scored.append(pair)

    scored.sort(key=lambda p: (-p.objective_score, p.pick_a.team_abbreviation, p.pick_b.team_abbreviation))

    best = scored[0] if scored else None
    # The same two teams with A/B swapped scores identically (the objective is
    # symmetric) but isn't a meaningfully different alternative -- skip past
    # any such swaps to find a genuinely different runner-up pairing, if one
    # exists.
    runner_up = None
    if best is not None:
        best_team_set = {best.pick_a.team_abbreviation, best.pick_b.team_abbreviation}
        for candidate in scored[1:]:
            if {candidate.pick_a.team_abbreviation, candidate.pick_b.team_abbreviation} != best_team_set:
                runner_up = candidate
                break
    return JointSearchResult(
        best=best, runner_up=runner_up, floor_relaxed=floor_relaxed, pairs_considered=len(scored)
    )


def _describe(option: TeamOption) -> str:
    win_pct = f"{option.win_pct:.1f}%" if option.win_pct is not None else "unknown"
    basis = " (estimated from spread)" if option.win_pct_source == "spread_estimate" else ""
    spread = f", spread {option.spread_detail}" if option.spread_detail else ""
    return f"{option.team_abbreviation} vs {option.opponent_abbreviation or '?'} -- {win_pct} win prob{basis}{spread}"


def _build_reasoning(
    pair: PairScore, floor_relaxed: bool, min_win_prob_floor_b: float, runner_up: Optional[PairScore]
) -> str:
    parts = [
        f"Entry A: {_describe(pair.pick_a)}.",
        f"Entry B: {_describe(pair.pick_b)}.",
    ]

    if floor_relaxed:
        parts.append(
            f"No team available to Entry B cleared the {min_win_prob_floor_b:.0f}% floor this week; "
            f"the floor was relaxed rather than leave Entry B without a pick."
        )
    else:
        parts.append(f"Entry B's pick clears the {min_win_prob_floor_b:.0f}% win-probability floor.")

    parts.append(
        f"The two picks are in different games (A faces {pair.pick_a.opponent_abbreviation or '?'}, "
        f"B faces {pair.pick_b.opponent_abbreviation or '?'}), so this week's outcomes are treated as independent."
    )
    parts.append(
        f"Estimated for this pairing -- both survive: {pair.both_survive_pct:.1f}%, "
        f"exactly one survives: {pair.one_survives_pct:.1f}%, both eliminated: {pair.both_eliminated_pct:.1f}%."
    )

    if runner_up is not None:
        parts.append(
            f"This pairing beat the next-best combination ({runner_up.pick_a.team_abbreviation}/"
            f"{runner_up.pick_b.team_abbreviation}) on the combined objective "
            f"({pair.objective_score:.3f} vs {runner_up.objective_score:.3f})."
        )

    if pair.field_availability_a is not None and pair.field_availability_b is not None:
        parts.append(
            f"Field-aware: {pair.field_availability_a:.0%} of the field could still pick "
            f"{pair.pick_a.team_abbreviation}, {pair.field_availability_b:.0%} could still pick "
            f"{pair.pick_b.team_abbreviation} -- the ranking (objective {pair.objective_score:.3f}, "
            f"vs {pair.raw_objective_score:.3f} ignoring the field) favors scarcer picks over equally "
            f"safe crowded ones, since the pot splits among winners."
        )

    return " ".join(parts)


def recommend(
    current_week_games: List[Game],
    current_week: int,
    used_teams_a: Optional[List[str]] = None,
    used_teams_b: Optional[List[str]] = None,
    min_win_prob_floor_b: float = DEFAULT_MIN_WIN_PROB_FLOOR_B,
    field_model: Optional[FieldModel] = None,
    field_weight: float = DEFAULT_FIELD_WEIGHT,
) -> JointRecommendation:
    """Jointly optimized picks for both entries, with reasoning.

    ``used_teams_a``/``used_teams_b`` default to loading
    ``state/used_teams_a.json`` / ``state/used_teams_b.json``.

    Pass ``field_model`` (data/field_picks.py, built from other pool
    participants' prior picks) to rank pairs with the field-aware
    crowding penalty described in the module docstring; omit it (the
    default) for the original field-blind ranking.
    """
    if used_teams_a is None:
        used_teams_a = load_used_teams_for_entry(ENTRY_A_NAME)
    if used_teams_b is None:
        used_teams_b = load_used_teams_for_entry(ENTRY_B_NAME)

    search = find_best_pair(
        current_week_games, used_teams_a, used_teams_b, min_win_prob_floor_b, field_model, field_weight
    )

    if search.best is None:
        return JointRecommendation(
            week=current_week,
            pick_a=None,
            pick_b=None,
            both_survive_pct=None,
            one_survives_pct=None,
            both_eliminated_pct=None,
            reasoning="No valid pick pair available this week (not enough eligible teams/games for both entries).",
            floor_relaxed=search.floor_relaxed,
            pairs_considered=search.pairs_considered,
            field_aware=field_model is not None,
        )

    best = search.best
    return JointRecommendation(
        week=current_week,
        pick_a=best.pick_a,
        pick_b=best.pick_b,
        both_survive_pct=best.both_survive_pct,
        one_survives_pct=best.one_survives_pct,
        both_eliminated_pct=best.both_eliminated_pct,
        reasoning=_build_reasoning(best, search.floor_relaxed, min_win_prob_floor_b, search.runner_up),
        floor_relaxed=search.floor_relaxed,
        pairs_considered=search.pairs_considered,
        field_aware=field_model is not None,
        field_availability_a=best.field_availability_a,
        field_availability_b=best.field_availability_b,
    )
