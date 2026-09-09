"""Compresses a ``Workspace`` into a schema digest for the system prompt.

The model never sees raw DataFrames -- only this string, generated once per
turn (or once per session, depending on the caller) and handed to whichever
provider backend is active. It needs to be dense enough that the model can
guess plausible column names and dtypes without a `describe` call, and small
enough that it doesn't dominate the context window on a workspace with a
dozen sheets. ``max_chars`` is enforced by degrading gracefully rather than
by a hard truncation mid-column: fewer example values first, then fewer
columns per table (with an explicit "... N more columns" marker), and only
if that still doesn't fit, a last-resort character truncation.
"""

from __future__ import annotations

import pandas as pd

from cellsense.io.classify import classify_table
from cellsense.io.schema import Workspace

__all__ = ["build_context"]

# (max_example_values, max_columns_per_table), tried in order until the
# rendered digest fits under max_chars.
_DEGRADE_STEPS: list[tuple[int, int | None]] = [
    (3, None),
    (2, None),
    (1, None),
    (1, 20),
    (1, 10),
    (1, 5),
]


def build_context(workspace: Workspace, max_chars: int = 6000) -> str:
    """Return a digest of every loaded table, capped at ``max_chars``."""
    if workspace.is_empty():
        return "No files are currently loaded."

    rendered = ""
    for max_examples, max_cols in _DEGRADE_STEPS:
        rendered = _render(workspace, max_examples, max_cols)
        if len(rendered) <= max_chars:
            return rendered

    # Nothing fit -- hard truncate as a last resort. The slice budget is
    # computed from the marker's actual length (not a hardcoded guess), and
    # even a max_chars smaller than the marker itself must not overshoot --
    # callers size prompt budgets off this cap, so "almost" respecting it is
    # as good as not respecting it at all.
    marker = "\n… [context truncated]"
    if max_chars <= len(marker):
        return marker[:max_chars]
    return rendered[: max_chars - len(marker)].rstrip() + marker


# ── private ──────────────────────────────────────────────────────────────────


def _render(workspace: Workspace, max_examples: int, max_cols: int | None) -> str:
    sections: list[str] = []
    for fd in workspace.files.values():
        header = (
            f"=== File: {fd.filename} (type: {fd.file_type}, total rows: {fd.total_rows:,}) ==="
        )
        sheet_sections = [
            _summarise_table(df, name, fd.file_type, max_examples, max_cols)
            for name, df in fd.sheets.items()
        ]
        sections.append(header + "\n" + "\n\n".join(sheet_sections))
    return "\n\n" + "\n\n".join(sections) + "\n"


def _summarise_table(
    df: pd.DataFrame,
    sheet_name: str,
    file_type: str,
    max_examples: int,
    max_cols: int | None,
) -> str:
    category = classify_table(df, sheet_name)
    lines: list[str] = []
    if file_type == "excel":
        lines.append(f"--- Sheet: {sheet_name} (category: {category}) ---")
    else:
        lines.append(f"(category: {category})")
    lines.append(f"Shape: {len(df):,} rows x {len(df.columns)} columns")

    columns = list(df.columns)
    shown_cols = columns if max_cols is None else columns[:max_cols]
    omitted = len(columns) - len(shown_cols)

    col_lines = ["Columns:"]
    for col in shown_cols:
        col_lines.append(_describe_column(df[col], max_examples))
    if omitted > 0:
        col_lines.append(f"  … {omitted} more column(s)")
    lines.append("\n".join(col_lines))

    return "\n".join(lines)


def _describe_column(series: pd.Series, max_examples: int) -> str:
    name = series.name
    dtype_label = _human_dtype(series)
    null_count = int(series.isna().sum())
    entry = f"  - {name!r}: {dtype_label}"
    if null_count:
        entry += f" ({null_count} nulls)"

    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        non_null = series.dropna()
        if not non_null.empty:
            entry += f" [min={non_null.min():g}, max={non_null.max():g}, mean={non_null.mean():g}]"
        return entry

    if max_examples > 0:
        examples = series.dropna().astype(str).unique()[:max_examples]
        if len(examples):
            entry += f" e.g. {', '.join(examples)}"
    return entry


def _human_dtype(series: pd.Series) -> str:
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_integer_dtype(dtype):
        return "integer"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_string_dtype(dtype) or dtype == object:  # noqa: E721 -- pandas idiom
        n_unique = series.nunique()
        return f"categorical ({n_unique} unique)" if n_unique <= 20 else "text"
    return str(dtype)
