"""The tool contract every module under ``cellsense/tools/`` implements.

A tool is a JSON Schema plus a pure function over a ``Workspace``. The
``@tool_spec`` decorator turns a handler function directly into the
``ToolSpec`` a module exposes as its module-level ``SPEC`` -- there is no
separate registration step, which is what lets ``registry.py`` discover
tools by just importing the listed modules and reading ``module.SPEC``.

This module also carries the handful of behaviours every tool needs and none
should reimplement: coercing an LLM's stringly-typed arguments back to the
JSON Schema type it declared, turning "unknown column" into a message with
close-match suggestions, and evaluating the filter conditions shared by
``filter_rows`` and ``plot`` through one hardened code path instead of a
raw ``DataFrame.query()`` string (which the prototype used, and which is an
arbitrary-expression evaluation risk once query strings can come from a
model).
"""

from __future__ import annotations

import json
import operator as op
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import pandas as pd

from cellsense.errors import ToolError
from cellsense.io.schema import ToolResult, Workspace

__all__ = [
    "OPERATORS",
    "ToolHandler",
    "ToolSpec",
    "apply_conditions",
    "check_columns",
    "condition_mask",
    "resolve_within_cwd",
    "tool_spec",
    "unknown_column_error",
]

ToolHandler = Callable[[dict[str, Any], Workspace], ToolResult]


@dataclass(frozen=True)
class ToolSpec:
    """Everything the registry, the LangChain export, and the permission
    layer need to know about one tool.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    side_effect: bool = False
    reads_filesystem: bool = False

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Check required keys are present and coerce common LLM type slips.

        Meta's Llama models (and, less consistently, other OpenAI-compatible
        backends) routinely emit JSON where booleans, numbers, and arrays are
        serialized as strings -- ``"true"`` instead of ``true``, ``"3"``
        instead of ``3``. The prototype patched this once, for one provider,
        as ``normalize_bool_args``. Doing it here instead, driven by each
        argument's declared JSON Schema type, fixes it for every provider and
        every argument type at once.
        """
        props: dict[str, Any] = self.parameters.get("properties", {})
        required: list[str] = self.parameters.get("required", [])
        missing = [key for key in required if args.get(key) is None]
        if missing:
            raise ToolError(
                f"{self.name}: missing required argument(s): {', '.join(missing)}.",
                hint=f"Expected arguments: {', '.join(props) or '(none)'}",
            )
        return {
            key: (_coerce(value, props[key]) if key in props else value)
            for key, value in args.items()
        }


def tool_spec(
    *,
    name: str,
    description: str,
    parameters: dict[str, Any],
    side_effect: bool = False,
    reads_filesystem: bool = False,
) -> Callable[[ToolHandler], ToolSpec]:
    """Decorator factory: wraps a handler function into a ``ToolSpec``.

    Usage, in a tool module::

        @tool_spec(name="describe", description="...", parameters={...})
        def SPEC(args: dict[str, Any], workspace: Workspace) -> ToolResult:
            ...

    After decoration, the module-level name ``SPEC`` *is* the ``ToolSpec``
    instance (its ``.handler`` is the function body above) -- exactly the
    ``SPEC: ToolSpec`` contract ``registry.py`` looks for.
    """

    def decorator(handler: ToolHandler) -> ToolSpec:
        return ToolSpec(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            side_effect=side_effect,
            reads_filesystem=reads_filesystem,
        )

    return decorator


def _coerce(value: Any, schema: dict[str, Any]) -> Any:
    """Coerce one argument value toward its declared JSON Schema type. Only
    acts on strings that look like a type slip; anything already the right
    shape (or not a recognizable slip) passes through untouched.
    """
    json_type = schema.get("type")
    if not isinstance(value, str):
        return value

    if json_type == "boolean":
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
        return value

    if json_type == "integer":
        try:
            return int(value.strip())
        except ValueError:
            return value

    if json_type == "number":
        try:
            return float(value.strip())
        except ValueError:
            return value

    if json_type == "array":
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                return json.loads(stripped)
            except (json.JSONDecodeError, TypeError):
                return value
        return [stripped] if stripped else []

    return value


# ── column lookup errors ─────────────────────────────────────────────────────


def unknown_column_error(df: pd.DataFrame, column: str) -> ToolError:
    """Build a ``ToolError`` for a missing column, listing close matches
    first (so the model can self-correct a typo in one retry) and the full
    column list after.
    """
    available = [str(c) for c in df.columns]
    matches = get_close_matches(column, available, n=3)
    hint = f"Available columns: {', '.join(available)}"
    if matches:
        hint = f"Did you mean: {', '.join(matches)}? {hint}"
    return ToolError(f"Column {column!r} not found.", hint=hint)


def check_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise ``unknown_column_error`` for the first name in ``columns`` that
    isn't in ``df``. Called before touching the DataFrame so a bad column
    name never surfaces as a confusing pandas ``KeyError``.
    """
    for column in columns:
        if column not in df.columns:
            raise unknown_column_error(df, column)


# ── shared condition evaluation (filter_rows, plot) ──────────────────────────

OPERATORS = (
    "eq",
    "ne",
    "gt",
    "gte",
    "lt",
    "lte",
    "contains",
    "startswith",
    "endswith",
    "in",
    "notin",
    "isnull",
    "notnull",
)

_CMP_OPS: dict[str, Callable[[pd.Series, Any], pd.Series]] = {
    "gt": op.gt,
    "gte": op.ge,
    "lt": op.lt,
    "lte": op.le,
}


def condition_mask(df: pd.DataFrame, column: str, operator_name: str, value: Any) -> pd.Series:
    """Evaluate one ``{column, operator, value}`` condition into a boolean
    mask. String comparisons (``eq``/``ne``/``contains``/``startswith``/
    ``endswith``/``in``/``notin``) are case-insensitive by default, matching
    how a human would expect "Region is West" to behave against "west".
    """
    if column not in df.columns:
        raise unknown_column_error(df, column)
    series = df[column]

    if operator_name == "isnull":
        return series.isna()
    if operator_name == "notnull":
        return series.notna()

    if operator_name in ("in", "notin"):
        if not isinstance(value, (list, tuple, set)):
            raise ToolError(f"Operator {operator_name!r} requires a list value, got {value!r}.")
        mask = _isin_case_insensitive(series, value)
        return mask if operator_name == "in" else ~mask

    if operator_name in ("contains", "startswith", "endswith"):
        str_series = series.astype(str).str.lower()
        needle = str(value).lower()
        if operator_name == "contains":
            return str_series.str.contains(needle, regex=False, na=False)
        if operator_name == "startswith":
            return str_series.str.startswith(needle, na=False)
        return str_series.str.endswith(needle, na=False)

    if operator_name in ("eq", "ne"):
        mask = _eq_case_insensitive(series, value)
        return mask if operator_name == "eq" else ~mask

    if operator_name in _CMP_OPS:
        try:
            return _CMP_OPS[operator_name](series, value)
        except TypeError as exc:
            raise ToolError(f"Cannot compare column {column!r} to {value!r}: {exc}") from exc

    raise ToolError(
        f"Unknown operator {operator_name!r}.",
        hint=f"Supported operators: {', '.join(OPERATORS)}",
    )


def apply_conditions(
    df: pd.DataFrame, conditions: list[dict[str, Any]], combine: str = "and"
) -> pd.DataFrame:
    """Filter ``df`` by a list of conditions combined with AND or OR."""
    if not conditions:
        return df
    if combine not in ("and", "or"):
        raise ToolError(f"Unknown combine mode {combine!r}.", hint="Use 'and' or 'or'.")

    masks = [condition_mask(df, *_normalize_condition(c, i)) for i, c in enumerate(conditions)]
    combined = masks[0]
    for mask in masks[1:]:
        combined = (combined & mask) if combine == "and" else (combined | mask)
    return df[combined]


# Keys a model plausibly reaches for instead of the documented ones. Accepting
# them is deliberate: a live run watched a model burn six tool calls and 22k
# tokens looping on ``{"col": ..., "op": ...}`` because the resulting bare
# ``KeyError('column')`` told it nothing about what was wrong. Being liberal
# about these three synonyms costs nothing and is the difference between the
# agent recovering in one round and not recovering at all.
_COLUMN_KEYS = ("column", "col", "field")
_OPERATOR_KEYS = ("operator", "op")
_VALUE_KEYS = ("value", "val")

_CONDITION_EXAMPLE = '{"column": "region", "operator": "eq", "value": "EMEA"}'


def _pick(cond: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in cond:
            return cond[key]
    return None


def _normalize_condition(cond: Any, index: int) -> tuple[str, str, Any]:
    """Validate one raw condition dict into ``(column, operator, value)``.

    Every failure here raises :class:`ToolError` rather than letting a
    ``KeyError`` escape, because the message is fed straight back to the model
    as a ``ToolMessage`` -- ``KeyError: 'column'`` is not something it can act
    on, whereas naming the missing key and showing the expected shape is.
    """
    where = f"conditions[{index}]"
    if not isinstance(cond, dict):
        raise ToolError(
            f"{where} must be an object, got {type(cond).__name__}.",
            hint=f"Expected shape: {_CONDITION_EXAMPLE}",
        )

    column = _pick(cond, _COLUMN_KEYS)
    if not isinstance(column, str) or not column:
        raise ToolError(
            f"{where} is missing a column name (got keys: {sorted(cond)}).",
            hint=f'Use the "column" key, e.g. {_CONDITION_EXAMPLE}',
        )

    operator_name = _pick(cond, _OPERATOR_KEYS)
    if not isinstance(operator_name, str) or not operator_name:
        raise ToolError(
            f"{where} is missing an operator (got keys: {sorted(cond)}).",
            hint=(
                f'Use the "operator" key, e.g. {_CONDITION_EXAMPLE}. '
                f"Supported operators: {', '.join(OPERATORS)}"
            ),
        )

    # isnull/notnull are the only operators that are complete without a value.
    # Defaulting the rest to None used to make a malformed condition match zero
    # rows *silently*, which reads to the model as a real "no results" answer.
    if operator_name in ("isnull", "notnull"):
        return column, operator_name, None
    if not any(key in cond for key in _VALUE_KEYS):
        raise ToolError(
            f"{where} uses operator {operator_name!r} but has no value.",
            hint=(
                f'Add a "value" key, e.g. {_CONDITION_EXAMPLE}. '
                "Only 'isnull' and 'notnull' may omit it."
            ),
        )
    return column, operator_name, _pick(cond, _VALUE_KEYS)


def _eq_case_insensitive(series: pd.Series, value: Any) -> pd.Series:
    if isinstance(value, str) and not pd.api.types.is_numeric_dtype(series):
        return series.astype(str).str.lower() == value.lower()
    return series == value


def _isin_case_insensitive(series: pd.Series, values: Iterable[Any]) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.isin(list(values))
    lowered = {str(v).lower() for v in values}
    return series.astype(str).str.lower().isin(lowered)


# ── filesystem sandboxing (list_directory, find_files) ───────────────────────


def resolve_within_cwd(raw: str | None) -> Path:
    """Resolve a user/model-supplied path, refusing to leave the invocation
    root (the process' current working directory).

    ``reads_filesystem=True`` tools are not permission-gated, so this is the
    only thing standing between "list the current directory" and an LLM
    being coaxed into reading ``~/.ssh``. Resolving through ``Path.resolve()``
    dereferences symlinks *before* the containment check, so a symlink that
    points outside the root is rejected too, not silently followed.
    """
    root = Path.cwd().resolve()
    candidate = Path(raw or ".").expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ToolError(
            f"Path {raw!r} resolves outside the working directory.",
            hint=f"Paths must stay within {root}.",
        ) from None
    return resolved
