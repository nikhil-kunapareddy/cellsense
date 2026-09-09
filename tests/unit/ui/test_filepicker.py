"""Pins cellsense.ui.filepicker: fuzzy matching/ranking, EXCLUDED_DIRS +
dot-file exclusion, the MAX_DEPTH walk cap, and @-mention extraction.
"""

from __future__ import annotations

import pytest

from cellsense.ui.filepicker import (
    EXCLUDED_DIRS,
    MAX_DEPTH,
    FileHit,
    FilePicker,
    _fuzzy_score,
    extract_attachments,
)


class TestFileHitSizeHint:
    @pytest.mark.parametrize(
        "size_bytes,expected",
        [
            (0, "0B"),
            (500, "500B"),
            (2048, "2.0KB"),
            (5 * 1024 * 1024, "5.0MB"),
            (3 * 1024 * 1024 * 1024, "3.0GB"),
        ],
    )
    def test_size_hint_formats_the_right_unit(self, size_bytes, expected) -> None:
        assert FileHit(relpath="a.csv", size_bytes=size_bytes).size_hint() == expected


class TestFuzzyScore:
    def test_empty_query_matches_everything_with_score_zero(self) -> None:
        assert _fuzzy_score("", "anything.csv") == 0

    def test_non_subsequence_does_not_match(self) -> None:
        assert _fuzzy_score("xyz", "sales.csv") is None

    def test_subsequence_in_order_matches(self) -> None:
        assert _fuzzy_score("sls", "sales.csv") is not None

    def test_tighter_earlier_match_scores_higher(self) -> None:
        tight = _fuzzy_score("sales", "sales_q1.csv")
        loose = _fuzzy_score("sales", "old_archive_sales_report.csv")
        assert tight > loose


class TestFilePickerWalk:
    def test_finds_supported_spreadsheet_files(self, tmp_path) -> None:
        (tmp_path / "a.csv").write_text("x")
        (tmp_path / "b.xlsx").write_text("x")
        (tmp_path / "c.txt").write_text("x")
        picker = FilePicker(tmp_path)
        relpaths = {hit.relpath for hit in picker.files()}
        assert relpaths == {"a.csv", "b.xlsx"}

    def test_excluded_directories_are_never_descended_into(self, tmp_path) -> None:
        for excluded in EXCLUDED_DIRS:
            d = tmp_path / excluded
            d.mkdir()
            (d / "hidden.csv").write_text("x")
        (tmp_path / "visible.csv").write_text("x")
        picker = FilePicker(tmp_path)
        relpaths = {hit.relpath for hit in picker.files()}
        assert relpaths == {"visible.csv"}

    def test_dot_prefixed_directories_are_excluded(self, tmp_path) -> None:
        hidden_dir = tmp_path / ".secret"
        hidden_dir.mkdir()
        (hidden_dir / "data.csv").write_text("x")
        picker = FilePicker(tmp_path)
        assert picker.files() == []

    def test_dot_prefixed_files_are_excluded(self, tmp_path) -> None:
        (tmp_path / ".hidden.csv").write_text("x")
        picker = FilePicker(tmp_path)
        assert picker.files() == []

    def test_unsupported_extensions_are_excluded(self, tmp_path) -> None:
        (tmp_path / "notes.txt").write_text("x")
        (tmp_path / "data.json").write_text("x")
        picker = FilePicker(tmp_path)
        assert picker.files() == []

    def test_refresh_forces_a_rescan(self, tmp_path) -> None:
        picker = FilePicker(tmp_path)
        assert picker.files() == []
        (tmp_path / "new.csv").write_text("x")
        assert picker.files() == []  # still cached
        picker.refresh()
        assert [hit.relpath for hit in picker.files()] == ["new.csv"]

    def test_search_ranks_and_limits_results(self, tmp_path) -> None:
        (tmp_path / "sales_q1.csv").write_text("x")
        (tmp_path / "old_archive_sales_report.csv").write_text("x")
        picker = FilePicker(tmp_path)
        results = picker.search("sales", limit=1)
        assert len(results) == 1
        assert results[0].relpath == "sales_q1.csv"

    def test_search_with_empty_query_returns_files_up_to_limit(self, tmp_path) -> None:
        for i in range(5):
            (tmp_path / f"f{i}.csv").write_text("x")
        picker = FilePicker(tmp_path)
        assert len(picker.search("", limit=3)) == 3

    def test_walk_depth_cap(self, tmp_path) -> None:
        """The walk pushes a subdirectory onto its stack only while the
        current directory's own depth tag is < MAX_DEPTH, and the root starts
        at depth 0 -- so the deepest directory whose *files* actually get
        scanned is MAX_DEPTH - 1 levels below the root (its own subdirectory,
        one level deeper still, is never pushed and so is never visited).
        """
        current = tmp_path
        for level in range(MAX_DEPTH + 2):
            current = current / f"d{level}"
            current.mkdir()
            (current / f"marker_{level}.csv").write_text("x")

        picker = FilePicker(tmp_path)
        found_levels = {
            int(hit.relpath.rsplit("marker_", 1)[1].split(".")[0]) for hit in picker.files()
        }
        assert max(found_levels) == MAX_DEPTH - 1
        assert MAX_DEPTH not in found_levels


class TestExtractAttachments:
    def test_no_mentions_returns_line_unchanged(self) -> None:
        text, paths = extract_attachments("what is the total revenue")
        assert text == "what is the total revenue"
        assert paths == []

    def test_single_mention_is_extracted(self) -> None:
        text, paths = extract_attachments("summarize @sales.csv please")
        assert paths == ["sales.csv"]
        assert "@sales.csv" not in text
        assert "summarize" in text and "please" in text

    def test_multiple_mentions_are_all_extracted(self) -> None:
        _text, paths = extract_attachments("join @a.csv and @b.xlsx")
        assert paths == ["a.csv", "b.xlsx"]

    def test_mention_at_start_of_line(self) -> None:
        text, paths = extract_attachments("@sales.csv total revenue?")
        assert paths == ["sales.csv"]
        assert text == "total revenue?"

    def test_collapses_extra_whitespace_left_behind(self) -> None:
        text, _ = extract_attachments("a   @file.csv   b")
        assert text == "a b"

    def test_email_like_at_is_not_treated_as_a_mention_when_preceded_by_non_space(self) -> None:
        # (?<!\S)@ requires the @ to be at the start or after whitespace.
        text, paths = extract_attachments("contact me@example.com for details")
        assert paths == []
        assert text == "contact me@example.com for details"
