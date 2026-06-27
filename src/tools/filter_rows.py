"""filter_rows tool: schema + handler.

Filters rows in a file/sheet using a pandas-compatible query string.
"""
from __future__ import annotations

from typing import Any, Dict

from src.data import FileData
from src.types import ToolResult, _get_sheet, _sheet_label
from src.utils.citations import Citation

SCHEMA: Dict[str, Any] = {
    "name": "filter_rows",
    "description": (
        "Filter rows in a file/sheet using a pandas-compatible query string. "
        "Returns matching rows with their original indices for citation."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Exact filename as provided at startup.",
            },
            "sheet_name": {
                "type": "string",
                "description": "Sheet name for Excel files; use 'default' for CSV files.",
            },
            "query": {
                "type": "string",
                "description": (
                    "A pandas DataFrame.query() expression, "
                    "e.g. 'Revenue > 10000 and Region == \"West\"'."
                ),
            },
        },
        "required": ["filename", "sheet_name", "query"],
    },
}


def handle(inputs: Dict[str, Any], file_data: Dict[str, FileData]) -> ToolResult:
    filename = inputs["filename"]
    sheet_name = inputs["sheet_name"]
    query = inputs["query"]

    df = _get_sheet(file_data, filename, sheet_name)
    try:
        result = df.query(query)
    except Exception as exc:
        raise ValueError(f"Invalid query {query!r}: {exc}") from exc

    citations = [Citation(
        filename=filename,
        sheet_name=_sheet_label(sheet_name),
        row_indices=list(result.index),
    )]
    summary = (
        f"filter_rows on {filename}[{sheet_name}] with query={query!r}"
        f" → {len(result)} rows matched"
    )
    return ToolResult(data=result, citations=citations, summary=summary)
