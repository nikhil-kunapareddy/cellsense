"""join: merge two loaded tables on one or more key columns.

Citations here need a temporary index column on each side because a merge's
output index is fresh (0..n-1 of the *result*), not the source rows that
contributed to each output row. Reset each side's real index into a scratch
column, merge, then read the scratch columns back off the merged frame to
recover which left/right source rows are actually cited -- and drop the
scratch columns before returning, so the model never sees them.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from cellsense.errors import ToolError
from cellsense.io.schema import Citation, ToolResult, Workspace
from cellsense.tools.base import check_columns, tool_spec

_JOIN_TYPES = ("inner", "left", "right", "outer")
_LEFT_IDX = "__cellsense_left_idx__"
_RIGHT_IDX = "__cellsense_right_idx__"

_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "left_filename": {"type": "string"},
        "left_sheet": {"type": "string", "description": "Sheet name for Excel files."},
        "right_filename": {"type": "string"},
        "right_sheet": {"type": "string", "description": "Sheet name for Excel files."},
        "left_on": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Join key column(s) in the left table.",
        },
        "right_on": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Join key column(s) in the right table.",
        },
        "how": {
            "type": "string",
            "enum": list(_JOIN_TYPES),
            "description": "Join type. Defaults to 'inner'.",
        },
    },
    "required": ["left_filename", "right_filename", "left_on", "right_on"],
}


def _handle(args: dict[str, Any], workspace: Workspace) -> ToolResult:
    left_ref, left_df = workspace.resolve(args["left_filename"], args.get("left_sheet"))
    right_ref, right_df = workspace.resolve(args["right_filename"], args.get("right_sheet"))

    left_on = _as_list(args["left_on"])
    right_on = _as_list(args["right_on"])
    how = args.get("how", "inner")

    if how not in _JOIN_TYPES:
        raise ToolError(f"Unknown join type {how!r}.", hint=f"Supported: {', '.join(_JOIN_TYPES)}")
    if len(left_on) != len(right_on):
        raise ToolError("left_on and right_on must have the same number of columns.")
    check_columns(left_df, left_on)
    check_columns(right_df, right_on)

    left_tmp = left_df.reset_index().rename(columns={"index": _LEFT_IDX})
    right_tmp = right_df.reset_index().rename(columns={"index": _RIGHT_IDX})
    suffixes = (f"_{_stem(left_ref.filename)}", f"_{_stem(right_ref.filename)}")

    merged = pd.merge(
        left_tmp,
        right_tmp,
        left_on=left_on,
        right_on=right_on,
        how=how,
        suffixes=suffixes,
    )

    left_rows = tuple(sorted(int(i) for i in merged[_LEFT_IDX].dropna().unique()))
    right_rows = tuple(sorted(int(i) for i in merged[_RIGHT_IDX].dropna().unique()))
    merged = merged.drop(columns=[_LEFT_IDX, _RIGHT_IDX])

    citations = [
        Citation(filename=left_ref.filename, sheet=left_ref.sheet, rows=left_rows),
        Citation(filename=right_ref.filename, sheet=right_ref.sheet, rows=right_rows),
    ]
    summary = (
        f"join({how}): {left_ref.label}.{left_on} <-> {right_ref.label}.{right_on} "
        f"-> {len(merged)} row(s)"
    )
    return ToolResult(data=merged, citations=citations, summary=summary)


def _as_list(value: Any) -> list[str]:
    return [value] if isinstance(value, str) else list(value)


def _stem(filename: str) -> str:
    return filename.rsplit(".", 1)[0]


SPEC = tool_spec(
    name="join",
    description=(
        "Join two loaded tables on matching key column(s) "
        f"({', '.join(_JOIN_TYPES)}), preserving row provenance from both sides."
    ),
    parameters=_PARAMETERS,
)(_handle)
