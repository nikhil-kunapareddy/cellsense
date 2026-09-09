"""aggregate: group rows and reduce one or more columns to a single value.

The one subtlety worth calling out is citations. ``groupby(...).groups``
hands back, for each group key, the *original* index labels of the rows that
fell into that group -- exactly the source-row provenance the tool contract
requires, and it falls out of pandas' own bookkeeping rather than needing a
manual index-tracking merge the way ``join`` does.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from cellsense.errors import ToolError
from cellsense.io.schema import Citation, ToolResult, Workspace
from cellsense.tools.base import check_columns, tool_spec

_FUNCS = ("count", "sum", "mean", "median", "min", "max", "nunique", "std")

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
        "group_by": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Column(s) to group by. Omit to aggregate the whole table into one row.",
        },
        "aggregations": {
            "type": "object",
            "description": (
                f"Mapping of column name to aggregation function ({', '.join(_FUNCS)})."
            ),
        },
    },
    "required": ["aggregations"],
}


def _handle(args: dict[str, Any], workspace: Workspace) -> ToolResult:
    ref, df = workspace.resolve(args.get("filename"), args.get("sheet"))
    group_by: list[str] = args.get("group_by") or []
    aggregations: dict[str, str] = args["aggregations"]

    if not aggregations:
        raise ToolError("aggregate requires at least one entry in 'aggregations'.")
    check_columns(df, [*group_by, *aggregations])
    _check_funcs(aggregations)

    if group_by:
        grouped = df.groupby(group_by, dropna=False)
        result = grouped.agg(aggregations).reset_index()
        result.columns = [*group_by, *(f"{func}_{col}" for col, func in aggregations.items())]
        citation_rows = sorted({int(i) for idx in grouped.groups.values() for i in idx})
    else:
        values = {f"{func}_{col}": [df[col].agg(func)] for col, func in aggregations.items()}
        result = pd.DataFrame(values)
        citation_rows = [int(i) for i in df.index]

    citations = [Citation(filename=ref.filename, sheet=ref.sheet, rows=tuple(citation_rows))]
    summary = (
        f"aggregate({ref.label}): group_by={group_by or '(none)'}, "
        f"{aggregations} -> {len(result)} row(s)"
    )
    return ToolResult(data=result, citations=citations, summary=summary)


def _check_funcs(aggregations: dict[str, str]) -> None:
    bad = {func for func in aggregations.values() if func not in _FUNCS}
    if bad:
        raise ToolError(
            f"Unknown aggregation function(s): {', '.join(sorted(bad))}.",
            hint=f"Supported functions: {', '.join(_FUNCS)}",
        )


SPEC = tool_spec(
    name="aggregate",
    description=(
        "Group rows by one or more columns and reduce other columns with "
        f"an aggregation function ({', '.join(_FUNCS)}). Omit group_by to "
        "aggregate the entire table into a single summary row."
    ),
    parameters=_PARAMETERS,
)(_handle)
