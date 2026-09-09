"""find_files: discover spreadsheet files under a directory and load them.

``reads_filesystem=True``, sandboxed the same way as ``list_directory``.
Unlike a pure read-only browse, this tool's whole point is to make newly
discovered files queryable in the same turn -- "find sales files in ./data"
should not require a second round trip just to load what it found -- so it
calls ``Workspace.add()`` for each new match. That is the one tool in this
package that mutates ``workspace.files``, and it only ever *adds* entries;
it never touches a DataFrame that was already loaded.
"""

from __future__ import annotations

import fnmatch
import zipfile
from typing import Any

import pandas as pd

from cellsense.errors import DataError
from cellsense.io.loaders import SUPPORTED_EXTENSIONS
from cellsense.io.schema import Citation, FileData, ToolResult, Workspace
from cellsense.tools.base import resolve_within_cwd, tool_spec

# Real-world "the file on disk is garbage" failures a corrupt/truncated/
# permission-denied spreadsheet can raise underneath Workspace.add() ->
# _load_excel()/_load_csv() -- openpyxl unzips .xlsx (BadZipFile), pandas'
# C parser can choke on malformed CSVs (ParserError), encoding can be wrong
# (UnicodeDecodeError), and the filesystem itself can misbehave (OSError).
# Deliberately a tuple of known failure modes, not a bare Exception catch:
# a programming error in our own code (a real KeyError/TypeError bug) should
# still crash loudly instead of being reported as a demure "error: ..." row.
_LOAD_FAILURE_TYPES = (
    DataError,
    zipfile.BadZipFile,
    UnicodeDecodeError,
    pd.errors.ParserError,
    OSError,
)

_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Directory to search under. Defaults to the current working directory.",
        },
        "pattern": {
            "type": "string",
            "description": (
                "Optional filename glob, e.g. 'sales*' or '*2024*'. "
                "Matches all spreadsheet files if omitted."
            ),
        },
        "recursive": {
            "type": "boolean",
            "description": "Search subdirectories recursively. Defaults to false.",
        },
    },
    "required": [],
}


def _handle(args: dict[str, Any], workspace: Workspace) -> ToolResult:
    root = resolve_within_cwd(args.get("path"))
    if not root.is_dir():
        data = pd.DataFrame(columns=["file", "status"])
        return ToolResult(data=data, citations=[], summary=f"Not a directory: {root}")

    pattern = args.get("pattern") or "*"
    recursive = bool(args.get("recursive", False))
    glob_fn = root.rglob if recursive else root.glob

    candidates = []
    for p in glob_fn("*"):
        if not p.is_file() or p.is_symlink():
            continue
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if not fnmatch.fnmatch(p.name.lower(), pattern.lower()):
            continue
        try:
            p.resolve().relative_to(root)
        except ValueError:
            continue  # symlinked into a directory outside root; refuse to follow it
        candidates.append(p)
    candidates.sort()

    if not candidates:
        data = pd.DataFrame(columns=["file", "status"])
        summary = f"No spreadsheet files found under {root}"
        return ToolResult(data=data, citations=[], summary=summary)

    already_loaded = {fd.filename for fd in workspace.files.values()}
    newly_loaded: list[FileData] = []
    rows: list[dict[str, str]] = []
    for path in candidates:
        if path.name in already_loaded:
            rows.append({"file": path.name, "status": "already loaded"})
            continue
        try:
            fd = workspace.add(path)
        except _LOAD_FAILURE_TYPES as exc:
            rows.append({"file": path.name, "status": f"error: {exc}"})
            continue
        already_loaded.add(fd.filename)
        newly_loaded.append(fd)
        rows.append({"file": fd.filename, "status": "loaded"})

    citations = [Citation(filename=fd.filename, sheet=None, rows=()) for fd in newly_loaded]
    schema_lines = [_schema_line(fd) for fd in newly_loaded]

    summary = (
        f"find_files under {root}: found {len(candidates)} file(s), loaded {len(newly_loaded)} new"
    )
    if schema_lines:
        summary += "\n\nLoaded file schemas:\n" + "\n".join(schema_lines)

    return ToolResult(data=pd.DataFrame(rows), citations=citations, summary=summary)


def _schema_line(fd: FileData) -> str:
    parts = []
    for sheet_name, df in fd.sheets.items():
        label = f"{fd.filename}[{sheet_name}]" if fd.file_type == "excel" else fd.filename
        cols = ", ".join(str(c) for c in df.columns)
        parts.append(f"  {label}: {len(df):,} rows, columns: {cols}")
    return "\n".join(parts)


SPEC = tool_spec(
    name="find_files",
    description=(
        "Search for spreadsheet files (.xlsx, .xls, .csv) under a directory, "
        "load the matches, and report their schemas so they can be queried "
        "immediately. Call this when the user hasn't given an exact file path."
    ),
    parameters=_PARAMETERS,
    reads_filesystem=True,
)(_handle)
