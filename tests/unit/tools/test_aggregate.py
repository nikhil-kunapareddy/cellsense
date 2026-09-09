"""Pins cellsense.tools.aggregate: happy paths cross-checked against raw
pandas, citation correctness, and failure paths.
"""

from __future__ import annotations

import pytest

from cellsense.errors import ToolError
from cellsense.tools.aggregate import SPEC


def _run(args, workspace):
    return SPEC.handler(args, workspace)


class TestGroupedAggregate:
    def test_sum_by_group_matches_raw_pandas(self, make_workspace, sales_df) -> None:
        ws = make_workspace(sales=sales_df)
        result = _run({"group_by": ["region"], "aggregations": {"revenue": "sum"}}, ws)

        expected = sales_df.groupby("region", dropna=False)["revenue"].sum().reset_index()
        got = result.data.sort_values("region").reset_index(drop=True)
        exp = expected.sort_values("region").reset_index(drop=True)
        assert got["region"].tolist() == exp["region"].tolist()
        assert got["sum_revenue"].tolist() == pytest.approx(exp["revenue"].tolist(), nan_ok=True)

    def test_citation_rows_match_the_grouped_source_indices(self, make_workspace, sales_df) -> None:
        ws = make_workspace(sales=sales_df)
        result = _run({"group_by": ["region"], "aggregations": {"revenue": "sum"}}, ws)
        assert len(result.citations) == 1
        citation = result.citations[0]
        assert citation.filename == "sales.csv"
        assert citation.sheet is None
        assert set(citation.rows) == set(sales_df.index)

    def test_multiple_aggregations_produce_expected_columns(self, make_workspace, sales_df) -> None:
        ws = make_workspace(sales=sales_df)
        result = _run(
            {"group_by": ["region"], "aggregations": {"revenue": "sum", "units": "mean"}}, ws
        )
        assert set(result.data.columns) == {"region", "sum_revenue", "mean_units"}

    def test_does_not_mutate_source_dataframe(
        self, make_workspace, sales_df, assert_no_mutation
    ) -> None:
        ws = make_workspace(sales=sales_df)
        verify = assert_no_mutation(ws)
        _run({"group_by": ["region"], "aggregations": {"revenue": "sum"}}, ws)
        verify()


class TestWholeTableAggregate:
    def test_no_group_by_reduces_to_a_single_row(self, make_workspace, sales_df) -> None:
        ws = make_workspace(sales=sales_df)
        result = _run({"aggregations": {"revenue": "sum"}}, ws)
        assert len(result.data) == 1
        assert result.data["sum_revenue"].iloc[0] == pytest.approx(
            sales_df["revenue"].sum(), nan_ok=True
        )

    def test_citation_rows_cover_every_source_row(self, make_workspace, sales_df) -> None:
        ws = make_workspace(sales=sales_df)
        result = _run({"aggregations": {"revenue": "sum"}}, ws)
        assert set(result.citations[0].rows) == set(sales_df.index)


class TestAggregateFailures:
    def test_empty_aggregations_raises(self, make_workspace, sales_df) -> None:
        ws = make_workspace(sales=sales_df)
        with pytest.raises(ToolError, match="at least one entry"):
            _run({"aggregations": {}}, ws)

    def test_unknown_column_raises_with_suggestion(self, make_workspace, sales_df) -> None:
        ws = make_workspace(sales=sales_df)
        with pytest.raises(ToolError) as exc_info:
            _run({"group_by": ["regoin"], "aggregations": {"revenue": "sum"}}, ws)
        assert "region" in (exc_info.value.hint or "")

    def test_unsupported_aggregation_function_raises(self, make_workspace, sales_df) -> None:
        ws = make_workspace(sales=sales_df)
        with pytest.raises(ToolError, match="Unknown aggregation function"):
            _run({"aggregations": {"revenue": "mode"}}, ws)

    def test_unknown_aggregation_column_raises(self, make_workspace, sales_df) -> None:
        ws = make_workspace(sales=sales_df)
        with pytest.raises(ToolError):
            _run({"aggregations": {"nope": "sum"}}, ws)


def test_spec_metadata() -> None:
    assert SPEC.name == "aggregate"
    assert SPEC.side_effect is False
    assert SPEC.reads_filesystem is False
    assert "aggregations" in SPEC.parameters["required"]
