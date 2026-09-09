"""The interactive REPL: banner, prompt, slash commands, @-mentions, live turns.

Codes against :class:`EngineLike`, a structural ``typing.Protocol`` capturing
exactly the surface documented in ``docs/ARCHITECTURE.md`` §7
(``graph/engine.py``'s ``Engine`` class) -- this module never imports
``cellsense.graph`` at runtime, only under ``TYPE_CHECKING``. That is what
lets this whole package be developed and tested against a fake engine before
the real one exists, and keeps the enforced import direction (``ui -> events
<- graph``) intact.

For the same reason this module doesn't import ``cellsense.io`` (the real
``Workspace``/``FileData`` types): the banner and ``/files`` render from a
small UI-owned :class:`FileSummary`/:class:`SheetSummary` pair instead. The
CLI wiring layer, which *does* own both sides, is expected to map
``Workspace`` -> ``list[FileSummary]`` when calling :func:`run_repl`.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    runtime_checkable,
)

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer
from prompt_toolkit.document import Document
from rich.console import Console
from rich.table import Table
from rich.text import Text

from cellsense.events import ApprovalRequested, Event, TurnFinished
from cellsense.ui.filepicker import AtMentionCompleter, FilePicker, extract_attachments
from cellsense.ui.slash import SlashCompleter, SlashContext, SlashOutcome
from cellsense.ui.slash import execute as execute_slash
from cellsense.ui.stream import run_turn
from cellsense.ui.theme import Theme, ThemeMode, make_console, resolve_theme

if TYPE_CHECKING:  # pragma: no cover - typing only; see the module docstring.
    from prompt_toolkit.input import Input
    from prompt_toolkit.output import Output

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
    "TranscriptEntry",
    "run_repl",
]

VERSION = "0.1.0"

OnApproval = Callable[[ApprovalRequested], Literal["allow", "allow_always", "deny"]]
OnAttach = Callable[[str], None]


# ── The seam: structural types the UI codes against ─────────────────────────


@runtime_checkable
class SessionInfo(Protocol):
    """Shape of one entry from ``EngineLike.sessions()``.

    ``docs/ARCHITECTURE.md`` doesn't pin exact field names for this yet, so
    ``/sessions`` and ``/resume`` read these defensively via ``getattr`` with
    fallbacks (see ``slash.py``) rather than assuming this Protocol matches
    the real ``graph.checkpoint`` type byte for byte.
    """

    thread_id: str
    updated_at: str
    preview: str
    turns: int


class EngineLike(Protocol):
    """Structural type for ``graph.engine.Engine`` -- the documented seam.

    Captures only what this package calls. Anything satisfying this shape
    (the real Engine, or a scripted fake) works with :func:`run_repl`.
    """

    @property
    def model_label(self) -> str: ...

    @property
    def provider_label(self) -> str: ...

    def stream_turn(
        self,
        question: str,
        *,
        thread_id: str,
        on_approval: OnApproval,
        cancel: threading.Event | None = None,
    ) -> Iterator[Event]: ...

    def sessions(self) -> list[SessionInfo]: ...

    def history(self, thread_id: str) -> list[tuple[str, str]]: ...


# ── UI-owned data shapes (decoupled from cellsense.io) ───────────────────────


@dataclass(frozen=True)
class SheetSummary:
    """One sheet's shape, for the banner/``/files`` -- never a live DataFrame."""

    name: str | None  # None for CSV
    rows: int
    cols: int
    columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class FileSummary:
    filename: str
    file_type: Literal["csv", "excel"]
    sheets: tuple[SheetSummary, ...]


@dataclass(frozen=True)
class ModelChoice:
    """One row of ``/model``'s catalog listing. Supplied by the CLI, which owns
    ``cellsense.providers`` -- the UI never imports that package."""

    selector: str
    provider: str
    has_key: bool


@dataclass(frozen=True)
class PermissionsSummary:
    """What ``/permissions`` shows. Supplied by the CLI, which owns ``cellsense.permissions``."""

    mode: str
    allow: tuple[str, ...]


@dataclass(frozen=True)
class TranscriptEntry:
    role: Literal["user", "assistant"]
    text: str


@dataclass
class SessionState:
    """Mutable, UI-local session bookkeeping: usage totals and the transcript
    used by ``/cost`` and ``/export``. Conversation *history* for the model
    itself lives in the engine's checkpointer, per the architecture doc --
    this is display-only."""

    thread_id: str
    show_tool_args: bool = True
    turns: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    transcript: list[TranscriptEntry] = field(default_factory=list)

    def record_turn(self, event: Event) -> None:
        if isinstance(event, TurnFinished):
            self.total_input_tokens += event.input_tokens
            self.total_output_tokens += event.output_tokens
            if event.cost_usd is not None:
                self.total_cost_usd += event.cost_usd


# ── Completion ────────────────────────────────────────────────────────────────


class _CombinedCompleter(Completer):
    """Routes to the slash-command menu or the @-mention picker, never both."""

    def __init__(self, picker: FilePicker) -> None:
        self._slash = SlashCompleter()
        self._mention = AtMentionCompleter(picker)

    def get_completions(self, document: Document, complete_event: object) -> Any:
        text = document.text_before_cursor
        if text.startswith("/") and " " not in text:
            yield from self._slash.get_completions(document, complete_event)
            return
        yield from self._mention.get_completions(document, complete_event)


# ── Banner ────────────────────────────────────────────────────────────────────


def _render_banner(
    console: Console,
    theme: Theme,
    files: Sequence[FileSummary],
    model_label: str,
    provider_label: str,
) -> None:
    """4 lines max: title + model, loaded files, a hint. Compact like Claude Code's."""
    console.print()
    title = Text("✻ CellSense", style=theme.style("banner.title"))
    title += Text(f"  v{VERSION}", style=theme.style("muted"))
    meta = Text(f"{provider_label} · {model_label}", style=theme.style("banner.meta"))
    header = Table.grid(expand=True)
    header.add_column(justify="left")
    header.add_column(justify="right")
    header.add_row(title, meta)
    console.print(header)

    if files:
        parts = []
        for f in files:
            rows = sum(sheet.rows for sheet in f.sheets)
            if len(f.sheets) > 1:
                parts.append(f"{f.filename} ({len(f.sheets)} sheets, {rows:,} rows)")
            else:
                parts.append(f"{f.filename} ({rows:,} rows)")
        console.print(Text("  " + " · ".join(parts), style=theme.style("banner.meta")))
    else:
        hint = "  No files loaded -- type @ to attach one"
        console.print(Text(hint, style=theme.style("banner.meta")))

    tip = "  Ask a question · /help for commands · @ to attach a file"
    console.print(Text(tip, style=theme.style("banner.hint")))
    console.print()


# ── Public entry point ───────────────────────────────────────────────────────


def run_repl(
    engine: EngineLike,
    *,
    files: Sequence[FileSummary] = (),
    thread_id: str | None = None,
    theme_mode: ThemeMode = "auto",
    show_tool_args: bool = True,
    on_attach: OnAttach,
    on_approval: OnApproval | None = None,
    on_model_change: Callable[[str], None] | None = None,
    list_models: Callable[[], list[ModelChoice]] | None = None,
    permissions_summary: Callable[[], PermissionsSummary] | None = None,
    console: Console | None = None,
    input: Input | None = None,
    output: Output | None = None,
) -> None:
    """Run the REPL until the user exits. Blocks the calling thread.

    Parameters mirror what the CLI must supply beyond the engine itself:

    * ``on_attach(path)`` -- called once per ``@path`` token found in a
      submitted question (after stripping them from the question text), so
      the caller can add the file to the live workspace mid-conversation.
    * ``on_approval(request) -> "allow" | "allow_always" | "deny"`` -- forwarded
      verbatim to ``Engine.stream_turn``. If omitted, a console-based prompt
      built in ``stream.py`` is used.
    * ``on_model_change(selector)`` -- called when ``/model <selector>`` is
      used to switch models; the Engine API has no setter for this, so
      reacting to it (e.g. rebuilding the Engine) is the CLI's job.
    * ``list_models`` / ``permissions_summary`` -- optional catalog/policy
      callbacks for ``/model`` and ``/permissions``; without them those
      commands degrade to an informational message rather than erroring,
      since ``cellsense.providers``/``cellsense.permissions`` are outside
      this package's ownership.
    * ``input`` / ``output`` -- forwarded to ``PromptSession`` so tests can
      drive the loop non-interactively via ``prompt_toolkit``'s
      ``create_pipe_input()`` and a captured ``Output``.
    """
    theme = resolve_theme(theme_mode)
    console = console or make_console(theme)
    session = SessionState(thread_id=thread_id or uuid.uuid4().hex, show_tool_args=show_tool_args)

    picker = FilePicker(Path.cwd())
    prompt_session: PromptSession[str] = PromptSession(
        completer=_CombinedCompleter(picker),
        complete_while_typing=True,
        input=input,
        output=output,
    )

    _render_banner(console, theme, files, engine.model_label, engine.provider_label)

    consecutive_interrupts = 0
    while True:
        try:
            raw = prompt_session.prompt("cellsense > ")
        except KeyboardInterrupt:
            consecutive_interrupts += 1
            if consecutive_interrupts >= 2:
                break
            console.print(Text("(Ctrl-C again to exit)", style=theme.style("muted")))
            continue
        except EOFError:
            break
        consecutive_interrupts = 0

        line = raw.strip()
        if not line:
            continue

        if line.startswith("/"):
            ctx = SlashContext(
                args=[],
                console=console,
                theme=theme,
                engine=engine,
                session=session,
                files=files,
                list_models=list_models,
                permissions_summary=permissions_summary,
            )
            outcome = execute_slash(line, ctx)
            if _apply_slash_outcome(
                outcome,
                console=console,
                theme=theme,
                session=session,
                files=files,
                engine=engine,
                on_model_change=on_model_change,
            ):
                break
            continue

        question, attachments = extract_attachments(line)
        for path in attachments:
            on_attach(path)
        if not question:
            continue

        session.turns += 1
        session.transcript.append(TranscriptEntry("user", question))
        terminal_event = run_turn(
            engine,
            question,
            thread_id=session.thread_id,
            console=console,
            theme=theme,
            show_tool_args=session.show_tool_args,
            on_approval=on_approval,
        )
        session.record_turn(terminal_event)
        if isinstance(terminal_event, TurnFinished):
            session.transcript.append(TranscriptEntry("assistant", terminal_event.answer))

    console.print()
    console.print(
        Text(
            f"Session ended · {session.turns} question(s) · {len(files)} file(s)",
            style=theme.style("muted"),
        )
    )


def _apply_slash_outcome(
    outcome: SlashOutcome,
    *,
    console: Console,
    theme: Theme,
    session: SessionState,
    files: Sequence[FileSummary],
    engine: EngineLike,
    on_model_change: Callable[[str], None] | None,
) -> bool:
    """Apply a handler's outcome to REPL-level state. Returns True if the REPL should exit."""
    if outcome.clear_screen:
        console.clear()
    if outcome.redraw_banner:
        _render_banner(console, theme, files, engine.model_label, engine.provider_label)
    if outcome.set_model is not None and on_model_change is not None:
        on_model_change(outcome.set_model)
    if outcome.resume_thread_id is not None:
        session.thread_id = outcome.resume_thread_id
    return outcome.exit_repl
