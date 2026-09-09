"""Pins cellsense.io.schema: TableRef/FileData/Workspace addressing and
resolution, Citation hashability, merge_citations grouping/sort (no
truncation), format_citations range-collapsing/count-threshold rendering,
and ToolResult.to_text/preview.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cellsense.errors import DataError
from cellsense.io.schema import (
    Citation,
    FileData,
    TableRef,
    ToolResult,
    Workspace,
    _json_safe,
    format_citations,
    merge_citations,
)


def _fd(sheets: dict[str, pd.DataFrame], file_type: str, path: str) -> FileData:
    return FileData(path=Path(path), sheets=sheets, file_type=file_type)


class TestTableRef:
    def test_label_omits_sheet_for_csv(self) -> None:
        assert TableRef(filename="a.csv", sheet=None).label == "a.csv"

    def test_label_includes_sheet_for_excel(self) -> None:
        assert TableRef(filename="a.xlsx", sheet="Q1").label == "a.xlsx[Q1]"

    def test_is_frozen_and_hashable(self) -> None:
        ref = TableRef(filename="a.csv", sheet=None)
        assert hash(ref) == hash(TableRef(filename="a.csv", sheet=None))
        with pytest.raises(AttributeError):
            ref.filename = "b.csv"  # type: ignore[misc]


class TestFileData:
    def test_filename_is_the_path_basename(self) -> None:
        fd = _fd({"default": pd.DataFrame({"a": [1]})}, "csv", "/some/dir/data.csv")
        assert fd.filename == "data.csv"

    def test_total_rows_sums_every_sheet(self) -> None:
        fd = _fd(
            {"S1": pd.DataFrame({"a": [1, 2]}), "S2": pd.DataFrame({"a": [1, 2, 3]})},
            "excel",
            "/b.xlsx",
        )
        assert fd.total_rows == 5


class TestWorkspaceResolve:
    def test_is_empty(self) -> None:
        assert Workspace().is_empty() is True
        assert (
            Workspace(
                files={"a.csv": _fd({"default": pd.DataFrame({"a": [1]})}, "csv", "a.csv")}
            ).is_empty()
            is False
        )

    def test_refs_csv_has_no_sheet(self) -> None:
        ws = Workspace(files={"a.csv": _fd({"default": pd.DataFrame({"a": [1]})}, "csv", "a.csv")})
        assert ws.refs() == [TableRef(filename="a.csv", sheet=None)]

    def test_refs_excel_lists_every_sheet(self) -> None:
        ws = Workspace(
            files={
                "b.xlsx": _fd(
                    {"S1": pd.DataFrame({"a": [1]}), "S2": pd.DataFrame({"a": [1]})},
                    "excel",
                    "b.xlsx",
                )
            }
        )
        assert set(ws.refs()) == {
            TableRef(filename="b.xlsx", sheet="S1"),
            TableRef(filename="b.xlsx", sheet="S2"),
        }

    def test_resolve_defaults_to_the_only_loaded_file(self) -> None:
        ws = Workspace(files={"a.csv": _fd({"default": pd.DataFrame({"a": [1]})}, "csv", "a.csv")})
        ref, df = ws.resolve(None, None)
        assert ref == TableRef(filename="a.csv", sheet=None)
        assert df["a"].iloc[0] == 1

    def test_resolve_with_no_files_loaded_raises(self) -> None:
        with pytest.raises(DataError, match="No files are loaded"):
            Workspace().resolve(None, None)

    def test_resolve_with_multiple_files_and_no_filename_raises(self) -> None:
        ws = Workspace(
            files={
                "a.csv": _fd({"default": pd.DataFrame({"a": [1]})}, "csv", "a.csv"),
                "b.csv": _fd({"default": pd.DataFrame({"a": [1]})}, "csv", "b.csv"),
            }
        )
        with pytest.raises(DataError, match="Multiple files"):
            ws.resolve(None, None)

    def test_resolve_unknown_filename_raises_with_close_match_hint(self) -> None:
        ws = Workspace(
            files={"sales.csv": _fd({"default": pd.DataFrame({"a": [1]})}, "csv", "sales.csv")}
        )
        with pytest.raises(DataError) as exc_info:
            ws.resolve("sales.cvs", None)
        assert "sales.csv" in (exc_info.value.hint or "")

    def test_resolve_csv_with_nondefault_sheet_raises(self) -> None:
        ws = Workspace(files={"a.csv": _fd({"default": pd.DataFrame({"a": [1]})}, "csv", "a.csv")})
        with pytest.raises(DataError, match="no sheet named"):
            ws.resolve("a.csv", "Sheet1")

    def test_resolve_csv_accepts_default_sheet_literal(self) -> None:
        ws = Workspace(files={"a.csv": _fd({"default": pd.DataFrame({"a": [1]})}, "csv", "a.csv")})
        ref, _ = ws.resolve("a.csv", "default")
        assert ref.sheet is None

    def test_resolve_excel_defaults_to_only_sheet(self) -> None:
        ws = Workspace(files={"b.xlsx": _fd({"S1": pd.DataFrame({"a": [1]})}, "excel", "b.xlsx")})
        ref, _ = ws.resolve("b.xlsx", None)
        assert ref.sheet == "S1"

    def test_resolve_excel_multi_sheet_with_no_sheet_argument_raises(self) -> None:
        ws = Workspace(
            files={
                "b.xlsx": _fd(
                    {"S1": pd.DataFrame({"a": [1]}), "S2": pd.DataFrame({"a": [1]})},
                    "excel",
                    "b.xlsx",
                )
            }
        )
        with pytest.raises(DataError, match="multiple sheets"):
            ws.resolve("b.xlsx", None)

    def test_resolve_excel_unknown_sheet_raises_with_close_match_hint(self) -> None:
        ws = Workspace(files={"b.xlsx": _fd({"Q1": pd.DataFrame({"a": [1]})}, "excel", "b.xlsx")})
        with pytest.raises(DataError) as exc_info:
            ws.resolve("b.xlsx", "Q11")
        assert "Q1" in (exc_info.value.hint or "")

    def test_table_is_a_convenience_wrapper_over_resolve(self) -> None:
        ws = Workspace(files={"a.csv": _fd({"default": pd.DataFrame({"a": [7]})}, "csv", "a.csv")})
        assert ws.table("a.csv")["a"].iloc[0] == 7


class TestWorkspaceAdd:
    def test_add_loads_a_new_file_without_touching_existing_ones(self, write_csv) -> None:
        existing = Workspace(
            files={"a.csv": _fd({"default": pd.DataFrame({"a": [1]})}, "csv", "a.csv")}
        )
        new_path = write_csv("b.csv", pd.DataFrame({"z": [9]}))
        existing.add(new_path)
        assert set(existing.files) == {"a.csv", "b.csv"}
        assert existing.files["a.csv"].sheets["default"]["a"].iloc[0] == 1

    def test_add_disambiguates_a_filename_collision(self, write_csv, tmp_path) -> None:
        first = write_csv("dup.csv", pd.DataFrame({"v": [1]}), subdir="north")
        ws = Workspace()
        ws.add(first)
        second = write_csv("dup.csv", pd.DataFrame({"v": [2]}), subdir="south")
        ws.add(second)
        assert "dup.csv" in ws.files
        assert "south/dup.csv" in ws.files
        assert ws.files["dup.csv"].sheets["default"]["v"].iloc[0] == 1
        assert ws.files["south/dup.csv"].sheets["default"]["v"].iloc[0] == 2


class TestCitation:
    def test_is_hashable_and_usable_in_a_set(self) -> None:
        c1 = Citation(filename="a.csv", sheet=None, rows=(1, 2))
        c2 = Citation(filename="a.csv", sheet=None, rows=(1, 2))
        c3 = Citation(filename="a.csv", sheet=None, rows=(1, 3))
        assert hash(c1) == hash(c2)
        assert {c1, c2, c3} == {c1, c3}

    def test_is_csv_reflects_sheet_none(self) -> None:
        assert Citation(filename="a.csv", sheet=None).is_csv() is True
        assert Citation(filename="a.xlsx", sheet="Q1").is_csv() is False


class TestMergeCitations:
    def test_groups_by_filename_and_sheet(self) -> None:
        citations = [
            Citation(filename="a.csv", sheet=None, rows=(1, 2)),
            Citation(filename="a.csv", sheet=None, rows=(3,)),
            Citation(filename="b.xlsx", sheet="Q1", rows=(0,)),
        ]
        merged = merge_citations(citations)
        by_key = {(c.filename, c.sheet): c.rows for c in merged}
        assert by_key[("a.csv", None)] == (1, 2, 3)
        assert by_key[("b.xlsx", "Q1")] == (0,)

    def test_dedupes_and_sorts_rows(self) -> None:
        citations = [
            Citation(filename="a.csv", sheet=None, rows=(5, 1, 3)),
            Citation(filename="a.csv", sheet=None, rows=(3, 1, 2)),
        ]
        merged = merge_citations(citations)
        assert merged[0].rows == (1, 2, 3, 5)

    def test_does_not_truncate_large_row_sets(self) -> None:
        # merge_citations must keep the *true* row set -- truncating here
        # would silently understate how many rows actually contributed, and
        # format_citations (not this function) owns making a large set
        # readable, via range-collapsing or an honest count.
        citations = [Citation(filename="a.csv", sheet=None, rows=tuple(range(200)))]
        merged = merge_citations(citations)
        assert len(merged[0].rows) == 200
        assert merged[0].rows == tuple(range(200))

    def test_empty_input_yields_empty_output(self) -> None:
        assert merge_citations([]) == []


class TestFormatCitations:
    def test_empty_list_is_empty_string(self) -> None:
        assert format_citations([]) == ""

    def test_csv_omits_sheet_clause(self) -> None:
        # (0, 1) is a two-element consecutive run -- it collapses to "0-1".
        out = format_citations([Citation(filename="data.csv", sheet=None, rows=(0, 1))])
        assert out == "Sources: data.csv [Rows: 0-1]"

    def test_excel_includes_sheet_clause(self) -> None:
        out = format_citations([Citation(filename="sales.xlsx", sheet="Q1", rows=(3, 7, 9))])
        assert out == "Sources: sales.xlsx [Sheet: Q1, Rows: 3, 7, 9]"

    def test_multiple_citations_are_pipe_joined(self) -> None:
        out = format_citations(
            [
                Citation(filename="sales.xlsx", sheet="Q1", rows=(3, 7, 9)),
                Citation(filename="data.csv", sheet=None, rows=(0, 1)),
            ]
        )
        assert out == "Sources: sales.xlsx [Sheet: Q1, Rows: 3, 7, 9] | data.csv [Rows: 0-1]"

    def test_empty_rows_renders_a_none_marker(self) -> None:
        out = format_citations([Citation(filename="data.csv", sheet=None, rows=())])
        assert out == "Sources: data.csv [Rows: (none)]"

    def test_single_row(self) -> None:
        out = format_citations([Citation(filename="data.csv", sheet=None, rows=(5,))])
        assert out == "Sources: data.csv [Rows: 5]"

    def test_contiguous_run_collapses_to_a_range(self) -> None:
        out = format_citations(
            [Citation(filename="data.csv", sheet=None, rows=tuple(range(3, 10)))]
        )
        assert out == "Sources: data.csv [Rows: 3-9]"

    def test_mixed_runs_and_singletons(self) -> None:
        rows = (3, 4, 5, 6, 7, 8, 9, 14, 20, 21, 22)
        out = format_citations([Citation(filename="data.csv", sheet=None, rows=rows)])
        assert out == "Sources: data.csv [Rows: 3-9, 14, 20-22]"

    def test_over_row_threshold_renders_as_an_honest_count(self) -> None:
        # The motivating real-world case: a total-revenue aggregate over all
        # 240 rows of sales_2024.csv. Even though it collapses to a single
        # clean range (0-239), 240 exceeds _MAX_ROW_INDICES, so it renders as
        # a count rather than a list -- "Rows: 0-239" is accurate but a count
        # is judged more honest/readable at this size (see ARCHITECTURE.md).
        out = format_citations(
            [Citation(filename="sales_2024.csv", sheet=None, rows=tuple(range(240)))]
        )
        assert out == "Sources: sales_2024.csv [240 rows]"

    def test_over_range_token_threshold_renders_as_an_honest_count(self) -> None:
        # 30 widely-scattered singletons collapse to 30 tokens (> _MAX_RANGE_TOKENS)
        # even though the total row count (30) is small enough on its own.
        rows = tuple(i * 100 for i in range(30))
        out = format_citations([Citation(filename="data.csv", sheet=None, rows=rows)])
        assert out == "Sources: data.csv [30 rows]"

    def test_over_threshold_count_form_preserves_the_sheet_clause(self) -> None:
        out = format_citations(
            [Citation(filename="sales.xlsx", sheet="Q1", rows=tuple(range(100)))]
        )
        assert out == "Sources: sales.xlsx [Sheet: Q1, 100 rows]"


class TestToolResultToText:
    def test_no_truncation_note_when_rows_fit(self) -> None:
        tr = ToolResult(data=pd.DataFrame({"a": range(5)}), summary="s", citations=[])
        text = tr.to_text(max_rows=20)
        assert "[showing" not in text
        assert text.startswith("s\n\n")

    def test_truncation_note_when_rows_exceed_max(self) -> None:
        tr = ToolResult(data=pd.DataFrame({"a": range(50)}), summary="s", citations=[])
        text = tr.to_text(max_rows=10)
        assert "[showing 10 of 50 rows]" in text

    def test_default_max_rows_is_20(self) -> None:
        tr = ToolResult(data=pd.DataFrame({"a": range(25)}), summary="s", citations=[])
        text = tr.to_text()
        assert "[showing 20 of 25 rows]" in text


class TestToolResultPreview:
    def test_defaults_to_five_rows(self) -> None:
        tr = ToolResult(data=pd.DataFrame({"a": range(10)}), summary="s", citations=[])
        assert len(tr.preview()) == 5

    def test_json_safety_of_numpy_and_pandas_scalars(self) -> None:
        df = pd.DataFrame(
            {
                "int64": pd.array([1, 2], dtype="int64"),
                "float_nan": [1.5, float("nan")],
                "bool_col": [True, False],
                "ts": pd.to_datetime(["2024-01-01", None]),
            }
        )
        tr = ToolResult(data=df, summary="s", citations=[])
        preview = tr.preview()
        # Must be directly json.dumps-able with no custom encoder.
        dumped = json.dumps(preview)
        assert dumped

        assert isinstance(preview[0]["int64"], int)
        assert preview[0]["ts"] == "2024-01-01T00:00:00"
        assert preview[1]["float_nan"] is None  # NaN -> None
        assert preview[1]["ts"] is None  # NaT -> None
        assert isinstance(preview[0]["bool_col"], bool)

    def test_numpy_float_nan_becomes_none_not_nan_literal(self) -> None:
        df = pd.DataFrame({"f": np.array([np.nan], dtype="float64")})
        tr = ToolResult(data=df, summary="s", citations=[])
        value = tr.preview()[0]["f"]
        assert value is None
        assert not (isinstance(value, float) and math.isnan(value))


class TestJsonSafeDirectly:
    """White-box coverage of _json_safe's individual branches for raw numpy
    scalar types that pandas' own to_dict() usually already converts to
    native Python types before preview() ever sees them (so these branches
    are otherwise dead code from ToolResult.preview()'s normal call path).
    """

    def test_numpy_datetime64_becomes_isoformat_string(self) -> None:
        assert _json_safe(np.datetime64("2024-01-01")) == "2024-01-01T00:00:00"

    def test_python_date_becomes_isoformat_string(self) -> None:
        assert _json_safe(date(2024, 1, 1)) == "2024-01-01"

    def test_python_datetime_becomes_isoformat_string(self) -> None:
        assert _json_safe(datetime(2024, 1, 1, 5, 0)) == "2024-01-01T05:00:00"

    def test_numpy_bool_becomes_python_bool(self) -> None:
        result = _json_safe(np.bool_(True))
        assert result is True
        assert isinstance(result, bool)

    def test_numpy_integer_becomes_python_int(self) -> None:
        result = _json_safe(np.int64(5))
        assert result == 5
        assert isinstance(result, int)

    def test_numpy_float_becomes_python_float(self) -> None:
        result = _json_safe(np.float64(1.5))
        assert result == 1.5
        assert isinstance(result, float)

    def test_numpy_nan_float_becomes_none(self) -> None:
        assert _json_safe(np.float64("nan")) is None

    def test_other_numpy_generic_falls_back_to_item(self) -> None:
        assert _json_safe(np.complex128(1 + 2j)) == 1 + 2j

    def test_plain_value_passes_through_unchanged(self) -> None:
        assert _json_safe("plain string") == "plain string"
        assert _json_safe(5) == 5

    def test_list_value_is_not_mistaken_for_na_and_passes_through(self) -> None:
        # pd.isna() on a list returns an elementwise array, not a scalar bool
        # -- `is_na is True` correctly comes out False rather than raising.
        assert _json_safe([1, 2, 3]) == [1, 2, 3]
