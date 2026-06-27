"""join_files tool: schema + handler.

Merges two loaded files on matching columns, preserving row provenance.
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from src.data import FileData
from src.types import ToolResult, _get_sheet, _sheet_label
from src.utils.citations import Citation

SCHEMA: Dict[str, Any] = {
    "name": "join_files",
    "description": (
        "Merge two loaded files on matching columns. "
        "Uses an inner join by default and returns the combined result with provenance."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "left_filename": {"type": "string"},
            "left_sheet": {"type": "string"},
            "right_filename": {"type": "string"},
            "right_sheet": {"type": "string"},
            "left_on": {"type": "string", "description": "Join column in the left file."},
            "right_on": {"type": "string", "description": "Join column in the right file."},
            "how": {
                "type": "string",
                "enum": ["inner", "left", "right", "outer"],
                "description": "Join type (default: inner).",
            },
        },
        "required": [
            "left_filename", "left_sheet",
            "right_filename", "right_sheet",
            "left_on", "right_on",
        ],
    },
}


def handle(inputs: Dict[str, Any], file_data: Dict[str, FileData]) -> ToolResult:
    lf = inputs["left_filename"]
    ls = inputs["left_sheet"]
    rf = inputs["right_filename"]
    rs = inputs["right_sheet"]
    left_on = inputs["left_on"]
    right_on = inputs["right_on"]
    how = inputs.get("how", "inner")

    left_df = _get_sheet(file_data, lf, ls)
    right_df = _get_sheet(file_data, rf, rs)

    merged = pd.merge(
        left_df.reset_index().rename(columns={"index": "_left_idx"}),
        right_df.reset_index().rename(columns={"index": "_right_idx"}),
        left_on=left_on,
        right_on=right_on,
        how=how,
        suffixes=(f"_{lf.split('.')[0]}", f"_{rf.split('.')[0]}"),
    )

    left_indices = merged["_left_idx"].dropna().astype(int).tolist()
    right_indices = merged["_right_idx"].dropna().astype(int).tolist()
    merged = merged.drop(columns=["_left_idx", "_right_idx"])

    citations = [
        Citation(filename=lf, sheet_name=_sheet_label(ls), row_indices=left_indices),
        Citation(filename=rf, sheet_name=_sheet_label(rs), row_indices=right_indices),
    ]
    summary = (
        f"join_files: {lf}[{ls}].{left_on} ↔ {rf}[{rs}].{right_on}"
        f" ({how}) → {len(merged)} rows"
    )
    return ToolResult(data=merged, citations=citations, summary=summary)
