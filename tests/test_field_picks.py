import pytest

from data.field_picks import (
    FieldPick,
    build_field_model,
    load_field_picks_from_file,
    parse_field_picks_from_csv_text,
)


class TestParseFieldPicksFromCsvText:
    def test_basic_columns(self):
        csv_text = "entry,team,week\nAlice,KC,1\nBob,SF,1\n"
        picks = parse_field_picks_from_csv_text(csv_text)
        assert picks == [FieldPick("Alice", "KC", 1), FieldPick("Bob", "SF", 1)]

    def test_column_name_aliases(self):
        csv_text = "participant,pick\nAlice,KC\n"
        picks = parse_field_picks_from_csv_text(csv_text)
        assert picks == [FieldPick("Alice", "KC", None)]

    def test_case_insensitive_headers(self):
        csv_text = "Entry,Team\nAlice,KC\n"
        picks = parse_field_picks_from_csv_text(csv_text)
        assert picks == [FieldPick("Alice", "KC", None)]

    def test_team_uppercased(self):
        csv_text = "entry,team\nAlice,kc\n"
        picks = parse_field_picks_from_csv_text(csv_text)
        assert picks[0].team == "KC"

    def test_missing_week_is_none(self):
        csv_text = "entry,team\nAlice,KC\n"
        picks = parse_field_picks_from_csv_text(csv_text)
        assert picks[0].week is None

    def test_unparseable_week_is_none_not_error(self):
        csv_text = "entry,team,week\nAlice,KC,bye\n"
        picks = parse_field_picks_from_csv_text(csv_text)
        assert picks[0].week is None

    def test_blank_rows_skipped(self):
        csv_text = "entry,team\nAlice,KC\n,\nBob,\n"
        picks = parse_field_picks_from_csv_text(csv_text)
        assert len(picks) == 1

    def test_missing_required_columns_raises(self):
        csv_text = "foo,bar\n1,2\n"
        with pytest.raises(ValueError, match="entry/team columns"):
            parse_field_picks_from_csv_text(csv_text)


class TestLoadFieldPicksFromFile:
    def test_reads_from_disk(self, tmp_path):
        path = tmp_path / "field.csv"
        path.write_text("entry,team\nAlice,KC\n")
        picks = load_field_picks_from_file(path)
        assert picks == [FieldPick("Alice", "KC", None)]


class TestFieldModel:
    def test_used_count_and_available_count(self):
        picks = [
            FieldPick("Alice", "KC", 1),
            FieldPick("Bob", "KC", 2),
            FieldPick("Carol", "SF", 1),
        ]
        model = build_field_model(picks)
        assert model.entry_count == 3
        assert model.used_count("KC") == 2
        assert model.available_count("KC") == 1
        assert model.used_count("DAL") == 0
        assert model.available_count("DAL") == 3

    def test_availability_fraction(self):
        picks = [FieldPick("Alice", "KC", 1), FieldPick("Bob", "KC", 1), FieldPick("Carol", "SF", 1)]
        model = build_field_model(picks)
        assert model.availability_fraction("KC") == pytest.approx(1 / 3)
        assert model.availability_fraction("DAL") == 1.0

    def test_empty_field_availability_is_zero_not_div_by_zero(self):
        model = build_field_model([])
        assert model.entry_count == 0
        assert model.availability_fraction("KC") == 0.0

    def test_same_entry_using_team_twice_counts_once(self):
        picks = [FieldPick("Alice", "KC", 1), FieldPick("Alice", "KC", 1)]
        model = build_field_model(picks)
        assert model.used_count("KC") == 1
