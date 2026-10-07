"""Models the rest of the survivor pool from other participants' prior
picks, so weekly recommendations can optionally account for how crowded
or scarce a team is across the field -- not just your own win probability.

This never changes your own survival math: a team's win probability this
week is the same regardless of who else picks it. It only matters if you
care about differentiating from the field, which matters here because
the pool's pot is split among winners -- being right isn't enough if a
huge chunk of the field is right alongside you every week; see
strategy/joint_optimizer.py's optional field-aware scoring, which uses
this module's output.

Input is a CSV of other participants' used teams (e.g. exported from a
shared Google Sheet -- fetch_field_picks_csv can pull directly from a
Google Sheets "export as CSV" URL, or you can download it and pass a
local path). Expected columns (case-insensitive, extra columns ignored,
matched against a few common aliases):

    entry (or participant/name)  -- a participant/entry identifier
    team (or pick)               -- team abbreviation used
    week                         -- optional; the week that pick was used

If ``week`` is absent or unparseable, the pick is still counted toward
that entry's used-teams set (current scarcity), just without week-level
detail (future-week projection).
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import requests

ENTRY_COLUMN_ALIASES = ("entry", "participant", "name", "player", "owner")
TEAM_COLUMN_ALIASES = ("team", "pick", "team_abbreviation", "selection")
WEEK_COLUMN_ALIASES = ("week",)


@dataclass(frozen=True)
class FieldPick:
    entry: str
    team: str
    week: Optional[int]


def _find_column(fieldnames_lower: Dict[str, str], aliases: tuple) -> Optional[str]:
    for alias in aliases:
        if alias in fieldnames_lower:
            return fieldnames_lower[alias]
    return None


def parse_field_picks_from_csv_text(csv_text: str) -> List[FieldPick]:
    reader = csv.DictReader(io.StringIO(csv_text))
    fieldnames_lower = {(name or "").strip().lower(): name for name in (reader.fieldnames or [])}

    entry_col = _find_column(fieldnames_lower, ENTRY_COLUMN_ALIASES)
    team_col = _find_column(fieldnames_lower, TEAM_COLUMN_ALIASES)
    week_col = _find_column(fieldnames_lower, WEEK_COLUMN_ALIASES)

    if entry_col is None or team_col is None:
        raise ValueError(
            f"Couldn't find entry/team columns in CSV header {reader.fieldnames!r}; "
            f"expected a column named one of {ENTRY_COLUMN_ALIASES} and one of {TEAM_COLUMN_ALIASES}"
        )

    picks: List[FieldPick] = []
    for row in reader:
        entry = (row.get(entry_col) or "").strip()
        team = (row.get(team_col) or "").strip().upper()
        if not entry or not team:
            continue

        week: Optional[int] = None
        if week_col and row.get(week_col):
            try:
                week = int(row[week_col])
            except ValueError:
                week = None

        picks.append(FieldPick(entry=entry, team=team, week=week))
    return picks


def load_field_picks_from_file(path: Path) -> List[FieldPick]:
    return parse_field_picks_from_csv_text(Path(path).read_text())


def fetch_field_picks_csv(url: str, timeout: int = 30) -> str:
    """Fetches a CSV from a direct URL -- e.g. a Google Sheets "export as
    CSV" link (https://docs.google.com/spreadsheets/d/<ID>/export?format=csv),
    which must be shared so "anyone with the link" can view it.
    """
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def load_field_picks(source: str) -> List[FieldPick]:
    """``source`` is a local file path, or an http(s) URL to fetch directly."""
    if source.startswith("http://") or source.startswith("https://"):
        return parse_field_picks_from_csv_text(fetch_field_picks_csv(source))
    return load_field_picks_from_file(Path(source))


@dataclass
class FieldModel:
    """Aggregated view of the field: which teams each entry has used."""

    used_teams_by_entry: Dict[str, Set[str]]

    @property
    def entry_count(self) -> int:
        return len(self.used_teams_by_entry)

    def used_count(self, team: str) -> int:
        """How many field entries have already used this team."""
        return sum(1 for used in self.used_teams_by_entry.values() if team in used)

    def available_count(self, team: str) -> int:
        return self.entry_count - self.used_count(team)

    def availability_fraction(self, team: str) -> float:
        """Fraction (0-1) of the field that still has this team available.

        Higher means more of the field *could* still pick it this week (a
        crowded, undifferentiating pick if it's also a strong option for
        most of them); lower means it's already scarce across the field,
        so picking it stands out more. This is a simple proxy -- it doesn't
        model whether a team is each entry's *best* remaining option, just
        whether they still have it at all.
        """
        if self.entry_count == 0:
            return 0.0
        return self.available_count(team) / self.entry_count


def build_field_model(picks: List[FieldPick]) -> FieldModel:
    used_teams_by_entry: Dict[str, Set[str]] = {}
    for pick in picks:
        used_teams_by_entry.setdefault(pick.entry, set()).add(pick.team)
    return FieldModel(used_teams_by_entry=used_teams_by_entry)


def load_field_model(source: str) -> FieldModel:
    return build_field_model(load_field_picks(source))
