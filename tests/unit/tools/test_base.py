"""Pins cellsense.tools.base: ToolSpec.validate_args coercion, close-match
column error messages, condition_mask/apply_conditions, and the
resolve_within_cwd filesystem sandbox.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cellsense.errors import ToolError
from cellsense.tools.base import (
    apply_conditions,
    check_columns,
    condition_mask,
    resolve_within_cwd,
    tool_spec,
    unknown_column_error,
)


def _dummy_spec(**overrides):
    params = {
        "type": "object",
        "properties": {
            "flag": {"type": "boolean"},
            "count": {"type": "integer"},
            "ratio": {"type": "number"},
            "items": {"type": "array", "items": {"type": "string"}},
            "name": {"type": "string"},
        },
        "required": ["name"],
    }
    params.update(overrides.pop("parameters", {}))
    return tool_spec(name="dummy", description="d", parameters=params, **overrides)(
        lambda args, ws: None
    )


class TestValidateArgsRequired:
    def test_missing_required_argument_raises_tool_error(self) -> None:
        spec = _dummy_spec()
        with pytest.raises(ToolError, match="missing required argument"):
            spec.validate_args({})

    def test_none_value_for_required_argument_counts_as_missing(self) -> None:
        spec = _dummy_spec()
        with pytest.raises(ToolError):
            spec.validate_args({"name": None})

    def test_present_required_argument_passes(self) -> None:
        spec = _dummy_spec()
        out = spec.validate_args({"name": "x"})
        assert out["name"] == "x"


class TestValidateArgsCoercion:
    @pytest.mark.parametrize(
        "raw,expected",
        [("true", True), ("1", True), ("yes", True), ("false", False), ("0", False), ("no", False)],
    )
    def test_boolean_coercion(self, raw, expected) -> None:
        spec = _dummy_spec()
        out = spec.validate_args({"name": "x", "flag": raw})
        assert out["flag"] is expected

    def test_boolean_coercion_leaves_unrecognized_strings_untouched(self) -> None:
        spec = _dummy_spec()
        out = spec.validate_args({"name": "x", "flag": "maybe"})
        assert out["flag"] == "maybe"

    def test_integer_coercion(self) -> None:
        spec = _dummy_spec()
        out = spec.validate_args({"name": "x", "count": "42"})
        assert out["count"] == 42

    def test_integer_coercion_leaves_unparseable_strings_untouched(self) -> None:
        spec = _dummy_spec()
        out = spec.validate_args({"name": "x", "count": "abc"})
        assert out["count"] == "abc"

    def test_number_coercion(self) -> None:
        spec = _dummy_spec()
        out = spec.validate_args({"name": "x", "ratio": "3.14"})
        assert out["ratio"] == pytest.approx(3.14)

    def test_array_coercion_from_json_string(self) -> None:
        spec = _dummy_spec()
        out = spec.validate_args({"name": "x", "items": '["a", "b"]'})
        assert out["items"] == ["a", "b"]

    def test_array_coercion_from_bare_string_wraps_in_a_list(self) -> None:
        spec = _dummy_spec()
        out = spec.validate_args({"name": "x", "items": "a"})
        assert out["items"] == ["a"]

    def test_array_coercion_from_empty_string_is_empty_list(self) -> None:
        spec = _dummy_spec()
        out = spec.validate_args({"name": "x", "items": ""})
        assert out["items"] == []

    def test_already_correct_types_pass_through_untouched(self) -> None:
        spec = _dummy_spec()
        out = spec.validate_args({"name": "x", "flag": True, "count": 3, "items": ["a"]})
        assert out["flag"] is True
        assert out["count"] == 3
        assert out["items"] == ["a"]

    def test_unknown_keys_pass_through_uncoerced(self) -> None:
        spec = _dummy_spec()
        out = spec.validate_args({"name": "x", "extra": "true"})
        assert out["extra"] == "true"


class TestUnknownColumnError:
    def test_message_names_the_column(self) -> None:
        df = pd.DataFrame({"region": [1], "revenue": [1]})
        err = unknown_column_error(df, "regoin")
        assert "'regoin'" in err.message

    def test_hint_suggests_close_match(self) -> None:
        df = pd.DataFrame({"region": [1], "revenue": [1]})
        err = unknown_column_error(df, "regoin")
        assert "region" in (err.hint or "")

    def test_hint_lists_all_columns_when_no_close_match(self) -> None:
        df = pd.DataFrame({"region": [1], "revenue": [1]})
        err = unknown_column_error(df, "zzz_completely_unrelated")
        assert "region" in (err.hint or "") and "revenue" in (err.hint or "")

    def test_check_columns_raises_on_first_bad_name(self) -> None:
        df = pd.DataFrame({"region": [1]})
        with pytest.raises(ToolError):
            check_columns(df, ["region", "nope"])

    def test_check_columns_passes_when_all_present(self) -> None:
        df = pd.DataFrame({"region": [1], "revenue": [1]})
        check_columns(df, ["region", "revenue"])  # must not raise


class TestConditionMask:
    def setup_method(self) -> None:
        self.df = pd.DataFrame(
            {
                "region": ["North", "south", "East", None],
                "revenue": [100, 200, None, 400],
            }
        )

    def test_eq_is_case_insensitive_for_strings(self) -> None:
        mask = condition_mask(self.df, "region", "eq", "NORTH")
        assert mask.tolist() == [True, False, False, False]

    def test_ne_is_case_insensitive_for_strings(self) -> None:
        mask = condition_mask(self.df, "region", "ne", "north")
        assert mask.tolist() == [False, True, True, True]

    def test_gt_on_numeric_column(self) -> None:
        mask = condition_mask(self.df, "revenue", "gt", 150)
        assert mask.fillna(False).tolist() == [False, True, False, True]

    def test_contains_is_case_insensitive(self) -> None:
        mask = condition_mask(self.df, "region", "contains", "orth")
        assert mask.tolist() == [True, False, False, False]

    def test_startswith(self) -> None:
        mask = condition_mask(self.df, "region", "startswith", "sou")
        assert mask.tolist() == [False, True, False, False]

    def test_endswith(self) -> None:
        mask = condition_mask(self.df, "region", "endswith", "TH")
        assert mask.tolist() == [True, True, False, False]

    def test_isnull(self) -> None:
        mask = condition_mask(self.df, "region", "isnull", None)
        assert mask.tolist() == [False, False, False, True]

    def test_notnull(self) -> None:
        mask = condition_mask(self.df, "revenue", "notnull", None)
        assert mask.tolist() == [True, True, False, True]

    def test_in_case_insensitive(self) -> None:
        mask = condition_mask(self.df, "region", "in", ["NORTH", "east"])
        assert mask.tolist() == [True, False, True, False]

    def test_notin_case_insensitive(self) -> None:
        mask = condition_mask(self.df, "region", "notin", ["north"])
        assert mask.tolist() == [False, True, True, True]

    def test_in_requires_a_list_value(self) -> None:
        with pytest.raises(ToolError, match="requires a list"):
            condition_mask(self.df, "region", "in", "north")

    def test_unknown_column_raises(self) -> None:
        with pytest.raises(ToolError):
            condition_mask(self.df, "nope", "eq", 1)

    def test_unknown_operator_raises_with_supported_list(self) -> None:
        with pytest.raises(ToolError) as exc_info:
            condition_mask(self.df, "region", "startswiht", "a")
        assert "startswith" in (exc_info.value.hint or "")

    def test_comparison_type_mismatch_raises_tool_error(self) -> None:
        df = pd.DataFrame({"region": ["a", "b"]})
        with pytest.raises(ToolError):
            condition_mask(df, "region", "gt", 5)


class TestApplyConditions:
    def test_and_combine(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
        out = apply_conditions(
            df,
            [
                {"column": "a", "operator": "gt", "value": 1},
                {"column": "b", "operator": "eq", "value": "x"},
            ],
            combine="and",
        )
        assert out.index.tolist() == [2]

    def test_or_combine(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
        out = apply_conditions(
            df,
            [
                {"column": "a", "operator": "eq", "value": 1},
                {"column": "b", "operator": "eq", "value": "x"},
            ],
            combine="or",
        )
        assert out.index.tolist() == [0, 2]

    def test_empty_conditions_returns_df_unchanged(self) -> None:
        df = pd.DataFrame({"a": [1, 2]})
        out = apply_conditions(df, [], combine="and")
        assert out is df

    def test_unknown_combine_mode_raises(self) -> None:
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ToolError, match="combine mode"):
            apply_conditions(df, [{"column": "a", "operator": "eq", "value": 1}], combine="xor")


class TestResolveWithinCwd:
    def test_default_path_is_cwd(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        assert resolve_within_cwd(None) == tmp_path.resolve()

    def test_relative_path_inside_root_is_allowed(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "sub").mkdir()
        assert resolve_within_cwd("sub") == (tmp_path / "sub").resolve()

    def test_path_escaping_root_raises(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ToolError, match="outside the working directory"):
            resolve_within_cwd("../")

    def test_absolute_path_outside_root_raises(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ToolError):
            resolve_within_cwd(str(Path(tmp_path).parent))

    def test_symlink_escaping_root_is_rejected(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        outside = tmp_path.parent / "outside_target"
        outside.mkdir(exist_ok=True)
        link = tmp_path / "escape_link"
        link.symlink_to(outside, target_is_directory=True)
        with pytest.raises(ToolError, match="outside the working directory"):
            resolve_within_cwd("escape_link")

    def test_symlink_staying_inside_root_is_allowed(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real_dir, target_is_directory=True)
        assert resolve_within_cwd("link") == real_dir.resolve()


class TestMalformedConditionsAreActionable:
    """A live run watched a model burn six tool calls and 22k tokens looping on
    a malformed condition, because the handler let a bare ``KeyError('column')``
    escape. These pin the contract that every malformed condition produces a
    ``ToolError`` whose message tells the model what to change.
    """

    @staticmethod
    def _df() -> pd.DataFrame:
        return pd.DataFrame({"name": ["Ana", "Ben"], "manager": ["Ana", None]})

    def test_missing_column_key_raises_tool_error_not_keyerror(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            apply_conditions(self._df(), [{"operator": "eq", "value": "Ana"}])
        assert "column" in str(excinfo.value)

    def test_missing_operator_key_lists_the_supported_operators(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            apply_conditions(self._df(), [{"column": "name", "value": "Ana"}])
        assert "operator" in str(excinfo.value)
        assert "startswith" in (excinfo.value.hint or "")

    def test_a_non_dict_condition_is_rejected_with_the_expected_shape(self) -> None:
        with pytest.raises(ToolError, match="must be an object"):
            apply_conditions(self._df(), ["name = Ana"])

    def test_missing_value_errors_instead_of_silently_matching_nothing(self) -> None:
        """The nastiest of the three: defaulting a missing value to None used to
        match zero rows, which the model reads as a truthful "no results".
        """
        with pytest.raises(ToolError, match="no value"):
            apply_conditions(self._df(), [{"column": "name", "operator": "eq"}])

    def test_isnull_and_notnull_may_omit_a_value(self) -> None:
        assert len(apply_conditions(self._df(), [{"column": "manager", "operator": "isnull"}])) == 1
        assert (
            len(apply_conditions(self._df(), [{"column": "manager", "operator": "notnull"}])) == 1
        )

    @pytest.mark.parametrize(
        "condition",
        [
            {"col": "name", "op": "eq", "value": "Ana"},
            {"field": "name", "operator": "eq", "val": "Ana"},
            {"column": "name", "op": "eq", "value": "Ana"},
        ],
        ids=["col+op", "field+val", "op-only"],
    )
    def test_common_key_synonyms_are_accepted(self, condition: dict) -> None:
        """Models reach for these; accepting them costs nothing and is the
        difference between recovering in one round and not at all.
        """
        assert len(apply_conditions(self._df(), [condition])) == 1

    def test_the_error_names_the_offending_condition_index(self) -> None:
        with pytest.raises(ToolError, match=r"conditions\[1\]"):
            apply_conditions(
                self._df(),
                [{"column": "name", "operator": "eq", "value": "Ana"}, {"column": "name"}],
            )
