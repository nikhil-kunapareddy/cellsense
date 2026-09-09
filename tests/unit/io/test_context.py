"""Pins cellsense.io.context.build_context: schema digest rendering and the
"stay under max_chars, degrade gracefully" contract.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cellsense.io.context import _render, build_context
from cellsense.io.schema import FileData, Workspace


def _single_file_workspace(df: pd.DataFrame, filename: str = "data.csv") -> Workspace:
    return Workspace(
        files={filename: FileData(path=Path(filename), sheets={"default": df}, file_type="csv")}
    )


def _many_column_workspace(n_cols: int = 80, n_rows: int = 5) -> Workspace:
    cols = {f"col_{i}": [f"value_{i}_{j}" for j in range(n_rows)] for i in range(n_cols)}
    return _single_file_workspace(pd.DataFrame(cols), filename="big.csv")


def test_empty_workspace_renders_placeholder_text() -> None:
    assert build_context(Workspace()) == "No files are currently loaded."


def test_digest_includes_file_header_and_category() -> None:
    df = pd.DataFrame({"revenue": [1, 2, 3], "region": ["a", "b", "c"]})
    ws = _single_file_workspace(df, "sales.csv")
    out = build_context(ws)
    assert "=== File: sales.csv (type: csv, total rows: 3) ===" in out
    assert "category: Financial" in out


def test_digest_includes_shape_and_column_lines() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    out = build_context(_single_file_workspace(df))
    assert "Shape: 2 rows x 2 columns" in out
    assert "'a': integer" in out
    assert "'b':" in out


def test_numeric_columns_show_min_max_mean() -> None:
    df = pd.DataFrame({"n": [1, 2, 3, 4]})
    out = build_context(_single_file_workspace(df))
    assert "min=1" in out and "max=4" in out and "mean=2.5" in out


def test_null_counts_are_reported() -> None:
    df = pd.DataFrame({"n": [1, None, 3]})
    out = build_context(_single_file_workspace(df))
    assert "(1 nulls)" in out


class TestRenderDegradeSteps:
    """White-box: exercise the private _render() dimensions directly rather
    than reverse-engineering a max_chars value that happens to trigger them.
    """

    def test_max_examples_limits_example_value_count(self) -> None:
        df = pd.DataFrame({"cat": ["v1", "v2", "v3", "v4", "v5"]})
        ws = _single_file_workspace(df)

        rendered_3 = _render(ws, max_examples=3, max_cols=None)
        rendered_1 = _render(ws, max_examples=1, max_cols=None)

        line_3 = next(line for line in rendered_3.splitlines() if "'cat'" in line)
        line_1 = next(line for line in rendered_1.splitlines() if "'cat'" in line)
        assert "e.g. v1, v2, v3" in line_3
        assert "e.g. v1" in line_1 and "v2" not in line_1 and "v3" not in line_1

    def test_max_cols_limits_columns_and_reports_omitted_count(self) -> None:
        ws = _many_column_workspace(n_cols=10, n_rows=2)
        rendered = _render(ws, max_examples=1, max_cols=3)
        assert "'col_0'" in rendered
        assert "'col_1'" in rendered
        assert "'col_2'" in rendered
        assert "'col_3'" not in rendered
        assert "… 7 more column(s)" in rendered

    def test_max_cols_none_shows_every_column(self) -> None:
        ws = _many_column_workspace(n_cols=10, n_rows=2)
        rendered = _render(ws, max_examples=1, max_cols=None)
        assert "more column(s)" not in rendered
        assert "'col_9'" in rendered


class TestBuildContextDegradation:
    def test_generous_budget_uses_the_first_degrade_step(self) -> None:
        df = pd.DataFrame({"cat": ["v1", "v2", "v3"]})
        ws = _single_file_workspace(df)
        assert build_context(ws, max_chars=6000) == _render(ws, 3, None)

    def test_tight_budget_picks_a_later_degrade_step_that_fits(self) -> None:
        ws = _many_column_workspace(n_cols=80, n_rows=5)
        out = build_context(ws, max_chars=500)
        assert len(out) <= 500
        assert "more column(s)" in out  # proves a max_cols degrade step was needed

    def test_impossible_budget_hard_truncates_with_a_marker(self) -> None:
        ws = _many_column_workspace(n_cols=80, n_rows=5)
        out = build_context(ws, max_chars=100)
        assert out.endswith("[context truncated]")


def test_build_context_result_never_exceeds_max_chars_even_in_hard_truncation() -> None:
    ws = _many_column_workspace(n_cols=80, n_rows=5)
    out = build_context(ws, max_chars=100)
    assert len(out) <= 100
