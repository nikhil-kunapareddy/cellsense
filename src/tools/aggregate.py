"""aggregate tool: schema + handler.

Groups rows by one or more columns and computes an aggregation on a target column.
"""
from __future__ import annotations

from typing import Any, Dict

from src.data import FileData
from src.types import ToolResult, _get_sheet, _sheet_label
from src.utils.citations import Citation

SCHEMA: Dict[str, Any] = {
    "name": "aggregate",
    "description": (
        "Group rows by one or more columns and compute an aggregation "
        "(sum | mean | count | max | min) on a target column."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "filename": {"type": "string"},
            "sheet_name": {"type": "string"},
            "group_by": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Column name(s) to group by.",
            },
            "agg_column": {"type": "string", "description": "Column to aggregate."},
            "agg_func": {
                "type": "string",
                "enum": ["sum", "mean", "count", "max", "min"],
            },
        },
        "required": ["filename", "sheet_name", "group_by", "agg_column", "agg_func"],
    },
}


def handle(inputs: Dict[str, Any], file_data: Dict[str, FileData]) -> ToolResult:
    filename = inputs["filename"]
    sheet_name = inputs["sheet_name"]
    group_by: list[str] = inputs["group_by"]
    agg_col: str = inputs["agg_column"]
    agg_func: str = inputs["agg_func"]

    df = _get_sheet(file_data, filename, sheet_name)
    grouped = df.groupby(group_by)[agg_col].agg(agg_func).reset_index()
    grouped.columns = [*group_by, f"{agg_func}_{agg_col}"]

    citations = [Citation(
        filename=filename,
        sheet_name=_sheet_label(sheet_name),
        row_indices=list(df.index),
    )]
    summary = (
        f"aggregate on {filename}[{sheet_name}]: "
        f"groupby={group_by}, {agg_func}({agg_col}) → {len(grouped)} groups"
    )
    return ToolResult(data=grouped, citations=citations, summary=summary)
