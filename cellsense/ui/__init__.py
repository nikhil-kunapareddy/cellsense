"""CellSense's terminal front-end (prompt_toolkit + rich).

Public surface: :func:`run_repl` (interactive mode), :func:`run_print`
(``--print`` mode), :func:`render_event` (the pure event -> renderable
mapping), and the theming helpers. See ``docs/ARCHITECTURE.md`` §1/§6/§7 for
how this package fits into the rest of CellSense -- in short, it depends only
on ``cellsense.events`` at runtime and treats ``cellsense.graph.Engine`` as an
opaque :class:`~cellsense.ui.repl.EngineLike` for typing purposes only.
"""

from __future__ import annotations

from cellsense.ui.printer import run_print
from cellsense.ui.render import render_event
from cellsense.ui.repl import (
    EngineLike,
    FileSummary,
    ModelChoice,
    OnApproval,
    OnAttach,
    PermissionsSummary,
    SessionInfo,
    SessionState,
    SheetSummary,
    TranscriptEntry,
    run_repl,
)
from cellsense.ui.theme import Theme, ThemeMode, make_console, resolve_theme

__all__ = [
    "EngineLike",
    "FileSummary",
    "ModelChoice",
    "OnApproval",
    "OnAttach",
    "PermissionsSummary",
    "SessionInfo",
    "SessionState",
    "SheetSummary",
    "Theme",
    "ThemeMode",
    "TranscriptEntry",
    "make_console",
    "render_event",
    "resolve_theme",
    "run_print",
    "run_repl",
]
