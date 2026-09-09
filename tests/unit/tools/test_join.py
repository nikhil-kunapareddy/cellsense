"""Pins cellsense.tools.join: merge correctness cross-checked against raw
pandas, exact per-join-type citation provenance, suffixing, and failures.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cellsense.errors import ToolError
from cellsense.tools.join import SPEC


def _run(args, workspace):
    return SPEC.handler(args, workspace)


@pytest.fixture
def left_right_workspace(make_workspace):
    left = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    right = pd.DataFrame({"id": [2, 3, 4], "val": [20, 30, 40]})
    return make_workspace(left=left, right=right)


class TestJoinCorrectness:
    def test_inner_join_matches_raw_pandas(self, left_right_workspace) -> None:
        ws = left_right_workspace
        result = _run(
            {
                "left_filename": "left.csv",
                "right_filename": "right.csv",
                "left_on": ["id"],
                "right_on": ["id"],
                "how": "inner",
            },
            ws,
        )
        expected = pd.merge(
            ws.table("left.csv"), ws.table("right.csv"), left_on="id", right_on="id", how="inner"
        )
        assert result.data["id"].tolist() == expected["id"].tolist()
        assert result.data["val"].tolist() == expected["val"].tolist()

    @pytest.mark.parametrize(
        "how,expected_left_rows,expected_right_rows",
        [
            ("inner", {1, 2}, {0, 1}),
            ("left", {0, 1, 2}, {0, 1}),
            ("right", {1, 2}, {0, 1, 2}),
            ("outer", {0, 1, 2}, {0, 1, 2}),
        ],
    )
    def test_citation_provenance_per_join_type(
        self, left_right_workspace, how, expected_left_rows, expected_right_rows
    ) -> None:
        ws = left_right_workspace
        result = _run(
            {
                "left_filename": "left.csv",
                "right_filename": "right.csv",
                "left_on": ["id"],
                "right_on": ["id"],
                "how": how,
            },
            ws,
        )
        left_citation, right_citation = result.citations
        assert left_citation.filename == "left.csv"
        assert right_citation.filename == "right.csv"
        assert set(left_citation.rows) == expected_left_rows
        assert set(right_citation.rows) == expected_right_rows

    def test_scratch_index_columns_never_appear_in_result(self, left_right_workspace) -> None:
        result = _run(
            {
                "left_filename": "left.csv",
                "right_filename": "right.csv",
                "left_on": ["id"],
                "right_on": ["id"],
            },
            left_right_workspace,
        )
        assert not any(str(c).startswith("__cellsense_") for c in result.data.columns)

    def test_does_not_mutate_source_dataframes(
        self, left_right_workspace, assert_no_mutation
    ) -> None:
        verify = assert_no_mutation(left_right_workspace)
        _run(
            {
                "left_filename": "left.csv",
                "right_filename": "right.csv",
                "left_on": ["id"],
                "right_on": ["id"],
            },
            left_right_workspace,
        )
        verify()


class TestJoinSuffixing:
    def test_overlapping_non_key_columns_get_filename_stem_suffixes(self, make_workspace) -> None:
        df_a = pd.DataFrame({"key": [1, 2], "note": ["x", "y"]})
        df_b = pd.DataFrame({"key": [1, 2], "note": ["p", "q"]})
        ws = make_workspace(alpha=df_a, beta=df_b)
        result = _run(
            {
                "left_filename": "alpha.csv",
                "right_filename": "beta.csv",
                "left_on": ["key"],
                "right_on": ["key"],
            },
            ws,
        )
        assert "note_alpha" in result.data.columns
        assert "note_beta" in result.data.columns


class TestJoinFailures:
    def test_missing_left_key_column_raises_with_suggestion(self, left_right_workspace) -> None:
        with pytest.raises(ToolError) as exc_info:
            _run(
                {
                    "left_filename": "left.csv",
                    "right_filename": "right.csv",
                    "left_on": ["nam"],
                    "right_on": ["id"],
                },
                left_right_workspace,
            )
        assert "name" in (exc_info.value.hint or "")

    def test_missing_right_key_column_raises(self, left_right_workspace) -> None:
        with pytest.raises(ToolError):
            _run(
                {
                    "left_filename": "left.csv",
                    "right_filename": "right.csv",
                    "left_on": ["id"],
                    "right_on": ["nope"],
                },
                left_right_workspace,
            )

    def test_unknown_join_type_raises(self, left_right_workspace) -> None:
        with pytest.raises(ToolError, match="Unknown join type"):
            _run(
                {
                    "left_filename": "left.csv",
                    "right_filename": "right.csv",
                    "left_on": ["id"],
                    "right_on": ["id"],
                    "how": "cross",
                },
                left_right_workspace,
            )

    def test_mismatched_key_lengths_raises(self, left_right_workspace) -> None:
        with pytest.raises(ToolError, match="same number of columns"):
            _run(
                {
                    "left_filename": "left.csv",
                    "right_filename": "right.csv",
                    "left_on": ["id", "name"],
                    "right_on": ["id"],
                },
                left_right_workspace,
            )


def test_spec_metadata() -> None:
    assert SPEC.name == "join"
    assert set(SPEC.parameters["required"]) == {
        "left_filename",
        "right_filename",
        "left_on",
        "right_on",
    }
    assert SPEC.side_effect is False
