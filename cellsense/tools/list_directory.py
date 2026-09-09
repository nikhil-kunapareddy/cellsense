"""list_directory: browse the filesystem before deciding what to load.

``reads_filesystem=True`` -- read-only, so it is not permission-gated, but
still sandboxed to the invocation root by ``resolve_within_cwd`` (see
``tools/base.py``). Symlinked entries are omitted entirely rather than
partially followed: simpler to reason about than "list it but don't
traverse it", and directory listings rarely need to show symlinks anyway.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from cellsense.errors import ToolError
from cellsense.io.loaders import SUPPORTED_EXTENSIONS
from cellsense.io.schema import ToolResult, Workspace
from cellsense.tools.base import resolve_within_cwd, tool_spec

_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Directory to list. Defaults to the current working directory.",
        },
    },
    "required": [],
}


def _handle(args: dict[str, Any], workspace: Workspace) -> ToolResult:
    root = resolve_within_cwd(args.get("path"))
    if not root.exists():
        raise ToolError(f"Path does not exist: {root}")
    if not root.is_dir():
        raise ToolError(f"Not a directory: {root}")

    entries = sorted(
        (p for p in root.iterdir() if not p.is_symlink()),
        key=lambda p: (p.is_file(), p.name.lower()),
    )
    rows = [
        {
            "name": p.name,
            "type": "file" if p.is_file() else "dir",
            "spreadsheet": p.suffix.lower() in SUPPORTED_EXTENSIONS,
        }
        for p in entries
    ]
    data = pd.DataFrame(rows, columns=["name", "type", "spreadsheet"])
    summary = f"list_directory({root}): {len(rows)} entrie(s)"
    return ToolResult(data=data, citations=[], summary=summary)


SPEC = tool_spec(
    name="list_directory",
    description=(
        "List files and subdirectories at a given path, to explore the "
        "filesystem before deciding which spreadsheet files to load."
    ),
    parameters=_PARAMETERS,
    reads_filesystem=True,
)(_handle)
