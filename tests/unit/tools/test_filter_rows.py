"""Pins cellsense.tools.filter_rows: happy paths cross-checked against raw
pandas boolean indexing, citation correctness, empty results, and failures.
"""

from __future__ import annotations

import pytest

from cellsense.errors import ToolError
from cellsense.tools.filter_rows import SPEC


def _run(args, workspace):
    return SPEC.handler(args, workspace)


def test_single_condition_matches_raw_pandas(make_workspace, sales_df) -> None:
    ws = make_workspace(sales=sales_df)
    result = _run({"conditions": [{"column": "region", "operator": "eq", "value": "North"}]}, ws)
    expected = sales_df[sales_df["region"].str.lower() == "north"]
    assert result.data.index.tolist() == expected.index.tolist()


def test_and_combine_matches_raw_pandas(make_workspace, sales_df) -> None:
    ws = make_workspace(sales=sales_df)
    result = _run(
        {
            "conditions": [
                {"column": "region", "operator": "eq", "value": "North"},
                {"column": "units", "operator": "gt", "value": 25},
            ],
            "combine": "and",
        },
        ws,
    )
    expected = sales_df[(sales_df["region"] == "North") & (sales_df["units"] > 25)]
    assert result.data.index.tolist() == expected.index.tolist()


def test_or_combine_matches_raw_pandas(make_workspace, sales_df) -> None:
    ws = make_workspace(sales=sales_df)
    result = _run(
        {
            "conditions": [
                {"column": "region", "operator": "eq", "value": "North"},
                {"column": "units", "operator": "gt", "value": 45},
            ],
            "combine": "or",
        },
        ws,
    )
    expected = sales_df[(sales_df["region"] == "North") | (sales_df["units"] > 45)]
    assert result.data.index.tolist() == expected.index.tolist()


def test_citation_rows_match_result_indices(make_workspace, sales_df) -> None:
    ws = make_workspace(sales=sales_df)
    result = _run({"conditions": [{"column": "region", "operator": "eq", "value": "North"}]}, ws)
    assert set(result.citations[0].rows) == set(result.data.index)
    assert result.citations[0].filename == "sales.csv"
    assert result.citations[0].sheet is None


def test_no_matches_yields_empty_result_and_empty_citation(make_workspace, sales_df) -> None:
    ws = make_workspace(sales=sales_df)
    result = _run({"conditions": [{"column": "region", "operator": "eq", "value": "Atlantis"}]}, ws)
    assert len(result.data) == 0
    assert result.citations[0].rows == ()


def test_does_not_mutate_source_dataframe(make_workspace, sales_df, assert_no_mutation) -> None:
    ws = make_workspace(sales=sales_df)
    verify = assert_no_mutation(ws)
    _run({"conditions": [{"column": "region", "operator": "eq", "value": "North"}]}, ws)
    verify()


class TestFailures:
    def test_no_conditions_raises(self, make_workspace, sales_df) -> None:
        ws = make_workspace(sales=sales_df)
        with pytest.raises(ToolError, match="at least one condition"):
            _run({"conditions": []}, ws)

    def test_unknown_column_raises_with_suggestion(self, make_workspace, sales_df) -> None:
        ws = make_workspace(sales=sales_df)
        with pytest.raises(ToolError) as exc_info:
            _run({"conditions": [{"column": "regoin", "operator": "eq", "value": "x"}]}, ws)
        assert "region" in (exc_info.value.hint or "")

    def test_unknown_operator_raises(self, make_workspace, sales_df) -> None:
        ws = make_workspace(sales=sales_df)
        with pytest.raises(ToolError, match="Unknown operator"):
            _run({"conditions": [{"column": "region", "operator": "matches", "value": "x"}]}, ws)


def test_spec_metadata() -> None:
    assert SPEC.name == "filter_rows"
    assert SPEC.parameters["required"] == ["conditions"]
    assert SPEC.side_effect is False
