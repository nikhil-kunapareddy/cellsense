"""list_directory tool: schema + handler.

Lists files and subdirectories at a given path so the agent can explore the
filesystem before deciding which spreadsheet files to load.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.data import FileData, SUPPORTED_EXTENSIONS
from src.types import ToolResult

SCHEMA: Dict[str, Any] = {
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


def handle(inputs: Dict[str, Any], file_data: Dict[str, FileData]) -> ToolResult:
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
