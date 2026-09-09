"""The non-interactive ``--print`` renderer: one question in, one answer out.

This is what makes CellSense pipeable: stdout carries only the final answer
(plus citations) as clean text with no ANSI when piped; the tool trace goes
to stderr and only when ``--verbose`` is set. No ``Live``, no spinner, no
prompt_toolkit -- a single pass over the event stream.
"""

from __future__ import annotations

import sys
import threading
from typing import TYPE_CHECKING, Literal

from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from cellsense.events import Event, TurnFailed, TurnFinished
from cellsense.ui.render import render_event
from cellsense.ui.theme import ThemeMode, make_console, resolve_theme

if TYPE_CHECKING:  # pragma: no cover - typing only, matches repl.py's rule.
    from cellsense.ui.repl import EngineLike, OnApproval


def _deny_all(_request: object) -> Literal["allow", "allow_always", "deny"]:
    """Default ``on_approval`` for print mode: there is no human to ask.

    Denying is the safe default for a non-interactive pipeline; a caller that
    wants side-effect tools (e.g. ``plot``) to run unattended in ``--print``
    mode should pass its own ``on_approval`` sourced from
    ``cellsense.permissions`` config (mode ``"allow"``).
    """
    return "deny"


def run_print(
    engine: EngineLike,
    question: str,
    *,
    thread_id: str,
    verbose: bool = False,
    theme_mode: ThemeMode = "auto",
    on_approval: OnApproval | None = None,
    console: Console | None = None,
    error_console: Console | None = None,
) -> int:
    """Answer one question and return a process exit code.

    0 on success, 1 on a non-cancelled failure, 130 on cancellation (matching
    the conventional 128+SIGINT code and ``errors.TurnCancelledError.exit_code``).
    Renders through the same ``render_event`` used by the interactive UI, so
    ``--verbose`` tool trace matches what the REPL shows, just on stderr.
    """
    # A piped stdout should never carry ANSI escapes; force no-color in that case.
    effective_mode: ThemeMode = theme_mode if sys.stdout.isatty() else "no-color"
    theme = resolve_theme(effective_mode)
    out = console or make_console(theme, file=sys.stdout)
    err = error_console or make_console(theme, file=sys.stderr)

    approval = on_approval or _deny_all
    cancel = threading.Event()
    terminal_event: Event = TurnFailed(message="the engine produced no events")

    events = engine.stream_turn(question, thread_id=thread_id, on_approval=approval, cancel=cancel)
    try:
        for item in events:
            if isinstance(item, (TurnFinished, TurnFailed)):
                terminal_event = item
                continue
            if verbose:
                renderable = render_event(item, theme, show_tool_args=True)
                if renderable is not None:
                    err.print(renderable)
    except KeyboardInterrupt:
        cancel.set()
        err.print(Text("Interrupted.", style=theme.style("error")))
        return 130

    if isinstance(terminal_event, TurnFinished):
        out.print(Markdown(terminal_event.answer))
        if terminal_event.citations:
            citation_style = theme.style("answer.citation")
            out.print(Text(" · ".join(terminal_event.citations), style=citation_style))
        return 0

    assert isinstance(terminal_event, TurnFailed)
    err.print(Text(f"Error: {terminal_event.message}", style=theme.style("error")))
    if terminal_event.hint:
        err.print(Text(terminal_event.hint, style=theme.style("muted")))
    return 130 if terminal_event.cancelled else 1
