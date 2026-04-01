"""MCP filesystem tools for CellSense.

Exposes two tools the agent can call during a conversation:

  list_directory  — list files and subdirectories at a given path
  find_files      — find spreadsheet files matching a pattern, then load
                    them into file_data so all other tools can use them

These are registered in TOOL_HANDLERS like any other tool. The agent uses
them when the user hasn't provided explicit file paths at startup, or when
they ask to "look in" a different directory.
"""
from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.data import FileData, load_files
from src.tools import ToolResult
from src.utils.citations import Citation

SUPPORTED_EXTENSIONS = {".xlsx", ".xls", ".csv"}

# ── Schemas ────────────────────────────────────────────────────────────────────

LIST_DIRECTORY_SCHEMA: Dict[str, Any] = {
    "name": "list_directory",
    "description": (
        "List files and subdirectories at a given path. "
        "Use this to explore the filesystem before deciding which files to load."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list. Defaults to current working directory.",
            },
        },
        "required": [],
    },
}

FIND_FILES_SCHEMA: Dict[str, Any] = {
    "name": "find_files",
    "description": (
        "Search for spreadsheet files (.xlsx, .xls, .csv) under a directory, "
        "load the matching files, and make them available for analysis. "
        "Call this when the user hasn't specified a file path or asks to "
        "find data files in a folder."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory to search in. Defaults to current working directory.",
            },
            "pattern": {
                "type": "string",
                "description": (
                    "Optional filename pattern to filter results (e.g. 'sales*', '*2024*'). "
                    "Matches all spreadsheet files if omitted."
                ),
            },
            "recursive": {
                "type": "boolean",
                "description": "Search subdirectories recursively. Defaults to false.",
            },
        },
        "required": [],
    },
}

MCP_SCHEMAS = [LIST_DIRECTORY_SCHEMA, FIND_FILES_SCHEMA]


# ── Handlers ───────────────────────────────────────────────────────────────────

def handle_list_directory(inputs: Dict[str, Any], file_data: Dict[str, FileData]) -> ToolResult:
    root = Path(inputs.get("path") or ".").expanduser().resolve()

    if not root.exists():
        raise ValueError(f"Path does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    rows = []
    for entry in entries:
        kind = "file" if entry.is_file() else "dir"
        spreadsheet = "yes" if entry.suffix.lower() in SUPPORTED_EXTENSIONS else ""
        rows.append({"name": entry.name, "type": kind, "spreadsheet": spreadsheet})

    df = pd.DataFrame(rows)
    summary = f"list_directory: {root}  ({len(rows)} entries)"
    return ToolResult(data=df, citations=[], summary=summary)


def handle_find_files(inputs: Dict[str, Any], file_data: Dict[str, FileData]) -> ToolResult:
    root = Path(inputs.get("path") or ".").expanduser().resolve()
    pattern = inputs.get("pattern") or "*"
    recursive = bool(inputs.get("recursive", False))

    if not root.exists():
        raise ValueError(f"Path does not exist: {root}")

    glob_fn = root.rglob if recursive else root.glob
    candidates = [
        p for p in glob_fn("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and fnmatch.fnmatch(p.name.lower(), pattern.lower())
    ]

    if not candidates:
        df = pd.DataFrame(columns=["file", "status"])
        return ToolResult(
            data=df, citations=[], summary=f"No spreadsheet files found under {root}"
        )

    # Load discovered files into the shared file_data dict
    newly_loaded: list[str] = []
    errors: list[str] = []
    for path in candidates:
        if path.name in file_data:
            continue  # already loaded
        try:
            loaded = load_files([path])
            file_data.update(loaded)
            newly_loaded.append(path.name)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    rows = [{"file": p.name, "status": "loaded" if p.name in newly_loaded else "already loaded"}
            for p in candidates]
    if errors:
        for e in errors:
            rows.append({"file": e, "status": "error"})

    df = pd.DataFrame(rows)
    citations = [
        Citation(filename=name, sheet_name=None, row_indices=[])
        for name in newly_loaded
    ]

    # Append schema info for newly loaded files so the agent knows what columns
    # are available and can call filter_rows / aggregate within the same turn.
    schema_lines: list[str] = []
    for name in newly_loaded:
        fd = file_data[name]
        for sheet_name, sheet_df in fd.sheets.items():
            cols = ", ".join(str(c) for c in sheet_df.columns.tolist())
            label = f"{name}[{sheet_name}]" if fd.file_type == "excel" else name
            sheet_ref = 'default' if fd.file_type == 'csv' else repr(sheet_name)
            schema_lines.append(
                f"  {label}: {len(sheet_df):,} rows, "
                f"sheet_name={sheet_ref}, columns: {cols}"
            )

    summary = (
        f"find_files under {root}: found {len(candidates)} file(s), "
        f"loaded {len(newly_loaded)} new  →  "
        + ", ".join(newly_loaded or ["none new"])
    )
    if schema_lines:
        summary += "\n\nLoaded file schemas:\n" + "\n".join(schema_lines)

    return ToolResult(data=df, citations=citations, summary=summary)
