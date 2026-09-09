"""describe: cheap, no-LLM schema and summary statistics for a table.

This is the tool the model should reach for before guessing a column name
in ``aggregate``/``filter_rows``/``join`` -- it costs nothing but a pandas
pass over the table (no row-level reshaping, no plotting), so there is no
reason for the model to ever hallucinate a column name when this is one
call away.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from cellsense.io.schema import Citation, ToolResult, Workspace
from cellsense.tools.base import check_columns, tool_spec

_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "filename": {
            "type": "string",
            "description": "File to read. Omit if only one file is loaded.",
        },
        "sheet": {
            "type": "string",
            "description": "Sheet name for Excel files. Omit for CSV or single-sheet Excel.",
        },
        "columns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Subset of columns to describe. Omit for every column.",
        },
    },
    "required": [],
}


def _handle(args: dict[str, Any], workspace: Workspace) -> ToolResult:
    ref, df = workspace.resolve(args.get("filename"), args.get("sheet"))
    columns: list[str] = args.get("columns") or list(df.columns)
    check_columns(df, columns)

    rows: list[dict[str, Any]] = []
    for col in columns:
        series = df[col]
        row: dict[str, Any] = {
            "column": col,
            "dtype": str(series.dtype),
            "non_null": int(series.notna().sum()),
            "nulls": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            non_null = series.dropna()
            row["min"] = non_null.min() if not non_null.empty else None
            row["max"] = non_null.max() if not non_null.empty else None
            row["mean"] = non_null.mean() if not non_null.empty else None
            row["std"] = non_null.std() if len(non_null) > 1 else None
        else:
            mode = series.dropna().mode()
            row["top"] = mode.iloc[0] if not mode.empty else None
        rows.append(row)

    result = pd.DataFrame(rows)
    all_rows = tuple(int(i) for i in df.index)
    citations = [Citation(filename=ref.filename, sheet=ref.sheet, rows=all_rows)]
    summary = f"describe({ref.label}): {len(df):,} row(s) x {len(columns)} column(s)"
    return ToolResult(data=result, citations=citations, summary=summary)


SPEC = tool_spec(
    name="describe",
    description=(
        "Get the schema and summary statistics (nulls, unique count, "
        "min/max/mean for numeric columns, most common value for others) "
        "for a table, or a subset of its columns. Cheap -- call this before "
        "guessing a column name."
    ),
    parameters=_PARAMETERS,
)(_handle)
