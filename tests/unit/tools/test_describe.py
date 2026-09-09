"""Pins cellsense.tools.describe: schema/summary stats cross-checked against
raw pandas, and failure paths.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cellsense.errors import ToolError
from cellsense.tools.describe import SPEC


def _run(args, workspace):
    return SPEC.handler(args, workspace)


def test_numeric_column_stats_match_raw_pandas(make_workspace, sales_df) -> None:
    ws = make_workspace(sales=sales_df)
    result = _run({"columns": ["revenue"]}, ws)
    row = result.data.iloc[0]
    non_null = sales_df["revenue"].dropna()
    assert row["min"] == pytest.approx(non_null.min())
    assert row["max"] == pytest.approx(non_null.max())
    assert row["mean"] == pytest.approx(non_null.mean())
    assert row["std"] == pytest.approx(non_null.std())
    assert row["nulls"] == int(sales_df["revenue"].isna().sum())
    assert row["non_null"] == int(sales_df["revenue"].notna().sum())


def test_categorical_column_reports_top_value(make_workspace, sales_df) -> None:
    ws = make_workspace(sales=sales_df)
    result = _run({"columns": ["region"]}, ws)
    row = result.data.iloc[0]
    assert row["top"] == sales_df["region"].mode().iloc[0]
    assert row["unique"] == sales_df["region"].nunique()


def test_omitting_columns_describes_every_column(make_workspace, sales_df) -> None:
    ws = make_workspace(sales=sales_df)
    result = _run({}, ws)
    assert set(result.data["column"]) == set(sales_df.columns)


def test_citation_covers_every_row(make_workspace, sales_df) -> None:
    ws = make_workspace(sales=sales_df)
    result = _run({"columns": ["revenue"]}, ws)
    assert set(result.citations[0].rows) == set(sales_df.index)


def test_unknown_column_raises_with_suggestion(make_workspace, sales_df) -> None:
    ws = make_workspace(sales=sales_df)
    with pytest.raises(ToolError) as exc_info:
        _run({"columns": ["revenu"]}, ws)
    assert "revenue" in (exc_info.value.hint or "")


def test_single_value_std_is_none(make_workspace) -> None:
    ws = make_workspace(one_row=pd.DataFrame({"n": [42]}))
    result = _run({"columns": ["n"]}, ws)
    assert result.data.iloc[0]["std"] is None


def test_does_not_mutate_source_dataframe(make_workspace, sales_df, assert_no_mutation) -> None:
    ws = make_workspace(sales=sales_df)
    verify = assert_no_mutation(ws)
    _run({}, ws)
    verify()


def test_spec_metadata() -> None:
    assert SPEC.name == "describe"
    assert SPEC.parameters["required"] == []
    assert SPEC.side_effect is False
