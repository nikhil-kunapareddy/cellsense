"""Tool definitions (schemas + handlers) for the CellSense agent.

Each tool exposes:
  SCHEMA  – dict passed to the LLM's tools= parameter
  handler – function called when the model requests the tool
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from src.data import FileData
from src.utils.citations import Citation

# ── Anthropic tool schemas ─────────────────────────────────────────────────────

FILTER_ROWS_SCHEMA: Dict[str, Any] = {
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

AGGREGATE_SCHEMA: Dict[str, Any] = {
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

JOIN_FILES_SCHEMA: Dict[str, Any] = {
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

ALL_SCHEMAS = [FILTER_ROWS_SCHEMA, AGGREGATE_SCHEMA, JOIN_FILES_SCHEMA]


# ── OpenAI-compatible schemas (for agents/llama.py) ────────────────────────────

def _to_oai(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an Anthropic-style tool schema to OpenAI function-calling format."""
    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema["description"],
            "parameters": schema["input_schema"],
        },
    }


ALL_OAI_SCHEMAS = [_to_oai(s) for s in ALL_SCHEMAS]


# ── Tool handlers ──────────────────────────────────────────────────────────────

class ToolResult:
    def __init__(self, data: pd.DataFrame, citations: list[Citation], summary: str):
        self.data = data
        self.citations = citations
        self.summary = summary

    def to_text(self, max_rows: int = 20) -> str:
        """Serialise result for insertion into the next LLM message."""
        rows_shown = min(max_rows, len(self.data))
        table = self.data.head(rows_shown).to_string(index=True, max_colwidth=50)
        note = f"\n[showing {rows_shown} of {len(self.data)} rows]" if len(
            self.data) > rows_shown else ""
        return f"{self.summary}\n\n{table}{note}"


def handle_filter_rows(inputs: Dict[str, Any], file_data: Dict[str, FileData]) -> ToolResult:
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


def handle_aggregate(inputs: Dict[str, Any], file_data: Dict[str, FileData]) -> ToolResult:
    filename = inputs["filename"]
    sheet_name = inputs["sheet_name"]
    group_by: list[str] = inputs["group_by"]
    agg_col: str = inputs["agg_column"]
    agg_func: str = inputs["agg_func"]

    df = _get_sheet(file_data, filename, sheet_name)
    grouped = df.groupby(group_by)[agg_col].agg(agg_func).reset_index()
    grouped.columns = [*group_by, f"{agg_func}_{agg_col}"]

    citations = [Citation(filename=filename, sheet_name=_sheet_label(
        sheet_name), row_indices=list(df.index))]
    summary = (
        f"aggregate on {filename}[{sheet_name}]: "
        f"groupby={group_by}, {agg_func}({agg_col}) → {len(grouped)} groups"
    )
    return ToolResult(data=grouped, citations=citations, summary=summary)


def handle_join_files(inputs: Dict[str, Any], file_data: Dict[str, FileData]) -> ToolResult:
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


TOOL_HANDLERS = {
    "filter_rows": handle_filter_rows,
    "aggregate": handle_aggregate,
    "join_files": handle_join_files,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_sheet(file_data: Dict[str, FileData], filename: str, sheet_name: str) -> pd.DataFrame:
    if filename not in file_data:
        available = ", ".join(file_data.keys())
        raise ValueError(f"File {filename!r} not loaded. Available: {available}")
    fd = file_data[filename]
    if sheet_name not in fd.sheets:
        available = ", ".join(fd.sheets.keys())
        raise ValueError(f"Sheet {sheet_name!r} not found in {filename}. Available: {available}")
    return fd.sheets[sheet_name]


def _sheet_label(sheet_name: str) -> str | None:
    return None if sheet_name == "default" else sheet_name


# ── Late imports (avoid circular dependencies) ─────────────────────────────────

from src.visualizations import PLOT_SCHEMA, handle_plot  # noqa: E402

ALL_SCHEMAS.append(PLOT_SCHEMA)
ALL_OAI_SCHEMAS.append(_to_oai(PLOT_SCHEMA))
TOOL_HANDLERS["plot"] = handle_plot

from src.mcp import MCP_SCHEMAS, handle_list_directory, handle_find_files  # noqa: E402

for _schema in MCP_SCHEMAS:
    ALL_SCHEMAS.append(_schema)
    ALL_OAI_SCHEMAS.append(_to_oai(_schema))
TOOL_HANDLERS["list_directory"] = handle_list_directory
TOOL_HANDLERS["find_files"] = handle_find_files
