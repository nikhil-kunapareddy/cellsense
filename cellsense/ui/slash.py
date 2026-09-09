"""Slash commands: table, dispatch, and the prompt_toolkit completion menu.

Commands are resolved and executed entirely here, before a question ever
reaches the engine (``docs/ARCHITECTURE.md`` explicitly keeps this a UI-only
concern). Handlers receive a :class:`SlashContext` assembled fresh by
``repl.py`` for each invocation and return a :class:`SlashOutcome` describing
what the REPL loop should do next (redraw, clear, exit, switch model/session)
-- handlers never touch the terminal loop or session state directly, which
keeps them trivially unit-testable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from rich.console import Console
from rich.table import Table
from rich.text import Text

from cellsense.ui.theme import Theme

if TYPE_CHECKING:  # typing only -- see the no-runtime-import rule in repl.py
    from cellsense.ui.repl import (
        EngineLike,
        ModelChoice,
        PermissionsSummary,
        SessionState,
    )


@dataclass
class SlashContext:
    """Everything a handler needs, assembled per invocation by the REPL loop.

    ``args`` are the whitespace-split tokens after the command name (e.g. for
    ``/model groq:llama-3.3-70b-versatile`` -> ``["groq:llama-3.3-70b-versatile"]``).
    """

    args: list[str]
    console: Console
    theme: Theme
    engine: EngineLike
    session: SessionState
    files: Sequence[object]  # repl.FileSummary; kept as object to avoid a runtime import
    list_models: Callable[[], list[ModelChoice]] | None = None
    permissions_summary: Callable[[], PermissionsSummary] | None = None


@dataclass
class SlashOutcome:
    """What the REPL loop should do after a command ran. All fields optional/falsy by default."""

    exit_repl: bool = False
    clear_screen: bool = False
    redraw_banner: bool = False
    set_model: str | None = None
    resume_thread_id: str | None = None


@dataclass(frozen=True)
class SlashCommand:
    name: str
    aliases: tuple[str, ...]
    summary: str
    arg_hint: str | None
    handler: Callable[[SlashContext], SlashOutcome]
    group: str = "general"


# ── Handlers ─────────────────────────────────────────────────────────────────


def _cmd_help(ctx: SlashContext) -> SlashOutcome:
    table = Table(
        title="Commands",
        header_style=ctx.theme.style("table.header"),
        box=None,
        pad_edge=False,
    )
    table.add_column("Command", style=ctx.theme.style("tool.name"))
    table.add_column("Args")
    table.add_column("Description")
    groups: dict[str, list[SlashCommand]] = {}
    for cmd in SLASH_TABLE:
        groups.setdefault(cmd.group, []).append(cmd)
    for group_name in ("general", "session", "history"):
        for cmd in groups.get(group_name, []):
            table.add_row(cmd.name, cmd.arg_hint or "", cmd.summary)
    ctx.console.print(table)
    hint = "Type @ mid-question to attach a CSV/XLSX file."
    ctx.console.print(Text(hint, style=ctx.theme.style("muted")))
    return SlashOutcome()


def _cmd_files(ctx: SlashContext) -> SlashOutcome:
    if not ctx.files:
        empty = "No files loaded. Type @ to attach one."
        ctx.console.print(Text(empty, style=ctx.theme.style("muted")))
        return SlashOutcome()
    for file_summary in ctx.files:
        filename = getattr(file_summary, "filename", "?")
        for sheet in getattr(file_summary, "sheets", ()):
            name = getattr(sheet, "name", None)
            label = f"{filename}[{name}]" if name else filename
            rows = getattr(sheet, "rows", 0)
            cols = getattr(sheet, "cols", 0)
            ctx.console.print(f"  {label}  [dim]{rows:,} rows x {cols} cols[/dim]")
            columns = getattr(sheet, "columns", ())
            if columns:
                preview = ", ".join(columns[:10])
                if len(columns) > 10:
                    preview += f" … +{len(columns) - 10} more"
                ctx.console.print(Text(f"    columns: {preview}", style=ctx.theme.style("muted")))
    return SlashOutcome()


def _cmd_model(ctx: SlashContext) -> SlashOutcome:
    if ctx.args:
        return SlashOutcome(set_model=ctx.args[0])
    if ctx.list_models is None:
        unavailable = "Model catalog unavailable in this session."
        ctx.console.print(Text(unavailable, style=ctx.theme.style("muted")))
        return SlashOutcome()
    choices = ctx.list_models()
    if not choices:
        ctx.console.print(Text("No models registered.", style=ctx.theme.style("muted")))
        return SlashOutcome()
    table = Table(header_style=ctx.theme.style("table.header"), box=None, pad_edge=False)
    table.add_column("Selector")
    table.add_column("Provider")
    table.add_column("Key")
    for choice in choices:
        table.add_row(choice.selector, choice.provider, "✓" if choice.has_key else "—")
    ctx.console.print(table)
    switch_hint = "Use /model <selector> to switch."
    ctx.console.print(Text(switch_hint, style=ctx.theme.style("muted")))
    return SlashOutcome()


def _cmd_clear(ctx: SlashContext) -> SlashOutcome:
    return SlashOutcome(clear_screen=True, redraw_banner=True)


def _cmd_cost(ctx: SlashContext) -> SlashOutcome:
    session = ctx.session
    total_tokens = session.total_input_tokens + session.total_output_tokens
    ctx.console.print(
        Text(
            f"{session.turns} turn(s) · {total_tokens:,} tokens · ${session.total_cost_usd:.4f}",
            style=ctx.theme.style("banner.meta"),
        )
    )
    return SlashOutcome()


def _render_sessions_table(ctx: SlashContext, sessions: Sequence[object]) -> None:
    table = Table(header_style=ctx.theme.style("table.header"), box=None, pad_edge=False)
    table.add_column("Thread")
    table.add_column("Updated")
    table.add_column("Turns")
    table.add_column("Preview")
    for info in sessions:
        table.add_row(
            str(getattr(info, "thread_id", "?")),
            str(getattr(info, "updated_at", "")),
            str(getattr(info, "turns", "")),
            str(getattr(info, "preview", ""))[:60],
        )
    ctx.console.print(table)


def _cmd_sessions(ctx: SlashContext) -> SlashOutcome:
    sessions = ctx.engine.sessions()
    if not sessions:
        ctx.console.print(Text("No saved sessions.", style=ctx.theme.style("muted")))
        return SlashOutcome()
    _render_sessions_table(ctx, list(sessions))
    return SlashOutcome()


def _cmd_resume(ctx: SlashContext) -> SlashOutcome:
    sessions = list(ctx.engine.sessions())
    if not sessions:
        ctx.console.print(Text("No saved sessions.", style=ctx.theme.style("muted")))
        return SlashOutcome()
    if not ctx.args:
        _render_sessions_table(ctx, sessions)
        resume_hint = "Use /resume <thread-id> to switch."
        ctx.console.print(Text(resume_hint, style=ctx.theme.style("muted")))
        return SlashOutcome()
    target = ctx.args[0]
    match = next((s for s in sessions if str(getattr(s, "thread_id", None)) == target), None)
    if match is None:
        ctx.console.print(Text(f"No session {target!r}.", style=ctx.theme.style("warning")))
        return SlashOutcome()
    ctx.console.print(Text(f"Resumed session {target}.", style=ctx.theme.style("success")))
    return SlashOutcome(resume_thread_id=target)


def _cmd_export(ctx: SlashContext) -> SlashOutcome:
    default_path = Path(f"cellsense-session-{ctx.session.thread_id}.md")
    path = Path(ctx.args[0]) if ctx.args else default_path
    lines = [f"# CellSense session `{ctx.session.thread_id}`", ""]
    for entry in ctx.session.transcript:
        lines.append(f"**{entry.role.capitalize()}:**")
        lines.append("")
        lines.append(entry.text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    ctx.console.print(Text(f"Exported transcript to {path}", style=ctx.theme.style("success")))
    return SlashOutcome()


def _cmd_permissions(ctx: SlashContext) -> SlashOutcome:
    if ctx.permissions_summary is None:
        unavailable = "Permission policy unavailable in this session."
        ctx.console.print(Text(unavailable, style=ctx.theme.style("muted")))
        return SlashOutcome()
    summary = ctx.permissions_summary()
    ctx.console.print(Text(f"Mode: {summary.mode}", style=ctx.theme.style("banner.meta")))
    if summary.allow:
        allowed = "Always allowed: " + ", ".join(summary.allow)
        ctx.console.print(Text(allowed, style=ctx.theme.style("muted")))
    return SlashOutcome()


def _cmd_exit(ctx: SlashContext) -> SlashOutcome:
    return SlashOutcome(exit_repl=True)


SLASH_TABLE: tuple[SlashCommand, ...] = (
    SlashCommand("/help", (), "Show this message", None, _cmd_help, group="general"),
    SlashCommand(
        "/files", (), "List loaded files, sheets, and columns", None, _cmd_files, group="session"
    ),
    SlashCommand(
        "/model",
        (),
        "List or switch the active model",
        "[provider:model]",
        _cmd_model,
        group="session",
    ),
    SlashCommand(
        "/cost", (), "Show token usage and cost for this session", None, _cmd_cost, group="session"
    ),
    SlashCommand(
        "/permissions",
        (),
        "Show the current permission policy",
        None,
        _cmd_permissions,
        group="session",
    ),
    SlashCommand(
        "/clear", (), "Clear the screen and redraw the banner", None, _cmd_clear, group="general"
    ),
    SlashCommand("/sessions", (), "List saved sessions", None, _cmd_sessions, group="history"),
    SlashCommand(
        "/resume", (), "Resume a saved session", "[thread-id]", _cmd_resume, group="history"
    ),
    SlashCommand(
        "/export",
        (),
        "Write the transcript to a markdown file",
        "[path]",
        _cmd_export,
        group="history",
    ),
    SlashCommand("/exit", ("/quit",), "End the session", None, _cmd_exit, group="general"),
)

_LOOKUP: dict[str, SlashCommand] = {}
for _cmd in SLASH_TABLE:
    _LOOKUP[_cmd.name] = _cmd
    for _alias in _cmd.aliases:
        _LOOKUP[_alias] = _cmd


def find_command(token: str) -> SlashCommand | None:
    """Look up a command by its name or an alias, case-insensitively."""
    normalized = token.lower()
    if not normalized.startswith("/"):
        normalized = "/" + normalized
    return _LOOKUP.get(normalized)


def execute(line: str, ctx: SlashContext) -> SlashOutcome:
    """Parse and run a slash command line, mutating nothing but ``ctx.console``.

    Unknown commands print a hint and are otherwise a no-op -- never raise,
    since a typo shouldn't be able to take down the REPL.
    """
    tokens = line.strip().split()
    if not tokens:
        return SlashOutcome()
    name, args = tokens[0], tokens[1:]
    command = find_command(name)
    if command is None:
        unknown = f"Unknown command: {name}  (type /help for options)"
        ctx.console.print(Text(unknown, style=ctx.theme.style("warning")))
        return SlashOutcome()
    ctx.args = args
    return command.handler(ctx)


class SlashCompleter(Completer):
    """Dropdown completion menu for slash commands.

    Fires only while the user is still typing the command token itself (no
    space yet) so free-text arguments aren't fought over; each candidate's
    one-line help renders as the completion's ``display_meta``, matching the
    look of Claude Code's ``/`` menu.
    """

    def __init__(self, commands: Sequence[SlashCommand] = SLASH_TABLE) -> None:
        self._commands = commands

    def get_completions(self, document: Document, complete_event: object) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return
        typed = text[1:].lower()
        for command in self._commands:
            candidates = [command.name[1:], *(alias.lstrip("/") for alias in command.aliases)]
            if any(candidate.startswith(typed) for candidate in candidates):
                display = command.name + (f" {command.arg_hint}" if command.arg_hint else "")
                yield Completion(
                    command.name,
                    start_position=-len(text),
                    display=display,
                    display_meta=command.summary,
                )
