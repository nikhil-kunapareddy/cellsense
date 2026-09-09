"""Pins cellsense.tools.plot: it writes a real PNG, is permission-gated
(side_effect=True), supports every chart type, and validates its arguments.
"""

from __future__ import annotations

import pytest

from cellsense.errors import ToolError
from cellsense.tools import plot as plot_module
from cellsense.tools.plot import SPEC

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(autouse=True)
def _redirect_output_dir(tmp_path, monkeypatch):
    """Never write into the real project ./output/ directory from tests."""
    monkeypatch.setattr(plot_module, "OUTPUT_DIR", tmp_path / "output")
    return tmp_path / "output"


def _run(args, workspace):
    return SPEC.handler(args, workspace)


def test_side_effect_flag_is_set() -> None:
    assert SPEC.side_effect is True


def test_bar_chart_writes_a_real_png(make_workspace, sales_df, _redirect_output_dir) -> None:
    ws = make_workspace(sales=sales_df)
    result = _run({"chart_type": "bar", "x_column": "region", "y_column": "revenue"}, ws)
    out_path_str = result.summary.split("Chart saved to ", 1)[1].split(" (", 1)[0]
    from pathlib import Path

    out_path = Path(out_path_str)
    assert out_path.exists()
    assert out_path.read_bytes()[:8] == _PNG_MAGIC
    assert out_path.parent == _redirect_output_dir


@pytest.mark.parametrize(
    "chart_type,extra",
    [
        # "units" (not "revenue") deliberately -- it has no NaNs, unlike
        # sales_df's revenue column. A pie chart requires every value finite;
        # see test_pie_chart_with_nan_values_raises_tool_error_not_a_raw_valueerror.
        ("bar", {"y_column": "units"}),
        ("line", {"y_column": "units"}),
        ("scatter", {"y_column": "units"}),
        ("pie", {"y_column": "units"}),
        ("hist", {}),
    ],
)
def test_every_chart_type_produces_a_png(make_workspace, sales_df, chart_type, extra) -> None:
    ws = make_workspace(sales=sales_df)
    args = {"chart_type": chart_type, "x_column": "region", **extra}
    result = _run(args, ws)
    assert f"({chart_type}," in result.summary


def test_pie_chart_with_nan_values_raises_tool_error_not_a_raw_valueerror(
    make_workspace, sales_df
) -> None:
    ws = make_workspace(sales=sales_df)
    assert sales_df["revenue"].isna().any()  # precondition: this column really has NaNs
    with pytest.raises(ToolError):
        _run({"chart_type": "pie", "x_column": "region", "y_column": "revenue"}, ws)


def test_group_by_and_agg_func_matches_raw_pandas(make_workspace, sales_df) -> None:
    ws = make_workspace(sales=sales_df)
    result = _run(
        {
            "chart_type": "bar",
            "x_column": "region",
            "y_column": "revenue",
            "group_by": "region",
            "agg_func": "sum",
        },
        ws,
    )
    expected = sales_df.groupby("region", dropna=False)["revenue"].sum().reset_index()
    got = result.data.sort_values("region").reset_index(drop=True)
    exp = expected.sort_values("region").reset_index(drop=True)
    assert got["region"].tolist() == exp["region"].tolist()
    assert got["sum_revenue"].tolist() == pytest.approx(exp["revenue"].tolist())


def test_conditions_filter_before_plotting(make_workspace, sales_df) -> None:
    ws = make_workspace(sales=sales_df)
    result = _run(
        {
            "chart_type": "bar",
            "x_column": "region",
            "y_column": "revenue",
            "conditions": [{"column": "region", "operator": "eq", "value": "North"}],
        },
        ws,
    )
    assert set(result.data["region"].unique()) == {"North"}


def test_citation_rows_match_plotted_data(make_workspace, sales_df) -> None:
    ws = make_workspace(sales=sales_df)
    result = _run(
        {
            "chart_type": "bar",
            "x_column": "region",
            "y_column": "revenue",
            "conditions": [{"column": "region", "operator": "eq", "value": "North"}],
        },
        ws,
    )
    expected_rows = set(sales_df[sales_df["region"] == "North"].index)
    assert set(result.citations[0].rows) == expected_rows


def test_does_not_mutate_source_dataframe(make_workspace, sales_df, assert_no_mutation) -> None:
    ws = make_workspace(sales=sales_df)
    verify = assert_no_mutation(ws)
    _run({"chart_type": "bar", "x_column": "region", "y_column": "revenue"}, ws)
    verify()


class TestPlotFailures:
    def test_unknown_chart_type_raises(self, make_workspace, sales_df) -> None:
        ws = make_workspace(sales=sales_df)
        with pytest.raises(ToolError, match="Unknown chart_type"):
            _run({"chart_type": "radar", "x_column": "region"}, ws)

    def test_non_hist_without_y_column_raises(self, make_workspace, sales_df) -> None:
        ws = make_workspace(sales=sales_df)
        with pytest.raises(ToolError, match="requires 'y_column'"):
            _run({"chart_type": "bar", "x_column": "region"}, ws)

    def test_unknown_column_raises(self, make_workspace, sales_df) -> None:
        ws = make_workspace(sales=sales_df)
        with pytest.raises(ToolError):
            _run({"chart_type": "hist", "x_column": "nope"}, ws)


def test_spec_metadata() -> None:
    assert SPEC.name == "plot"
    assert SPEC.parameters["required"] == ["chart_type", "x_column"]
