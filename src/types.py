"""Shared types used across tools, visualizations, and mcp.

Lives outside src/tools/ so that visualizations and mcp can import
ToolResult without triggering a circular dependency through tools/__init__.py.
"""
from __future__ import annotations

import pandas as pd

from src.data import FileData


class ToolResult:
    def __init__(self, data: pd.DataFrame, citations: list, summary: str):
        self.data = data
        self.citations = citations
        self.summary = summary

    def to_text(self, max_rows: int = 20) -> str:
        rows_shown = min(max_rows, len(self.data))
        table = self.data.head(rows_shown).to_string(index=True, max_colwidth=50)
        note = (
            f"\n[showing {rows_shown} of {len(self.data)} rows]"
            if len(self.data) > rows_shown
            else ""
        )
        return f"{self.summary}\n\n{table}{note}"


def _get_sheet(file_data: dict[str, FileData], filename: str, sheet_name: str) -> pd.DataFrame:
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
