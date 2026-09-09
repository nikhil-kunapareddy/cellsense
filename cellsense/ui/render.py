"""Pure mapping from :mod:`cellsense.events` to rich renderables.

``render_event`` is deliberately a pure function: given an event (plus a theme
and a display flag) it returns a renderable or ``None``, with no console I/O
and no state. That makes it usable from three very different call sites --
the interactive live status line (``stream.py``), the ``--print`` renderer
(``printer.py``), and tests -- without any of them needing to agree on when or
how often it is called.

Per the contract in ``events.py``: unknown event kinds must be ignored, never
raise. New event types added there require a new branch here; forgetting one
degrades to "renders nothing" rather than crashing the UI.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.text import Text

from cellsense import events as ev
from cellsense.ui.theme import Theme, resolve_theme

#: Longest single-line rendering of a tool's argument dict before truncation.
_MAX_ARGS_LEN = 60

_DEFAULT_THEME = resolve_theme("auto")


def render_event(
    event: ev.Event,
    theme: Theme | None = None,
    *,
    show_tool_args: bool = True,
) -> RenderableType | None:
    """Map one event to a renderable, or ``None`` if it has no visual.

    ``theme`` defaults to an auto-resolved theme so ``render_event(event)`` --
    the signature shown in the spec -- works standalone; callers that already
    hold a :class:`Theme` (the REPL, the printer) should pass it explicitly so
    styling stays consistent across a session.
    """
    handler = _HANDLERS.get(event.kind)
    if handler is None:
        return None
    return handler(event, theme or _DEFAULT_THEME, show_tool_args)


# ── Formatting helpers ───────────────────────────────────────────────────────


def _fmt_value(value: Any) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    return repr(value)


def _format_args(args: dict[str, Any]) -> str:
    """One-line ``key=val, key2=val2`` rendering, ellipsised past ``_MAX_ARGS_LEN``."""
    if not args:
        return ""
    rendered = ", ".join(f"{key}={_fmt_value(value)}" for key, value in args.items())
    if len(rendered) > _MAX_ARGS_LEN:
        rendered = rendered[: _MAX_ARGS_LEN - 1].rstrip() + "…"
    return rendered


def _tool_header(name: str, args: dict[str, Any], theme: Theme, show_tool_args: bool) -> Text:
    if show_tool_args and args:
        return Text(f"⏺ {name}({_format_args(args)})", style=theme.style("tool.name"))
    return Text(f"⏺ {name}", style=theme.style("tool.name"))


# ── Per-event renderers ──────────────────────────────────────────────────────


def _no_visual(event: ev.Event, theme: Theme, show_tool_args: bool) -> None:
    return None


def _render_plan_created(
    event: ev.PlanCreated, theme: Theme, show_tool_args: bool
) -> RenderableType | None:
    # A single subtask means the planner chose not to decompose -- render
    # nothing, per the docstring in events.py.
    if len(event.subtasks) <= 1:
        return None
    lines = [Text(f"Plan · {len(event.subtasks)} subtasks", style=theme.style("plan.title"))]
    for index, (_subtask_id, question) in enumerate(event.subtasks, start=1):
        lines.append(Text(f"  {index}. {question}", style=theme.style("plan.item")))
    return Group(*lines)


def _render_text_delta(event: ev.TextDelta, theme: Theme, show_tool_args: bool) -> RenderableType:
    if event.channel == "worker":
        return Text(f"  {event.text}", style=theme.style("worker"))
    if event.channel == "reasoning":
        return Text(event.text, style=theme.style("reasoning"))
    # "answer": a bare text fragment. Partial markdown can't be parsed safely
    # mid-stream, so callers that accumulate the full answer (stream.py) render
    # it as Markdown once, at TurnFinished; this branch exists so render_event
    # still returns something sensible if a caller feeds it a raw delta.
    return Text(event.text)


def _render_tool_started(
    event: ev.ToolStarted, theme: Theme, show_tool_args: bool
) -> RenderableType:
    return _tool_header(event.name, event.args, theme, show_tool_args)


def _render_tool_finished(
    event: ev.ToolFinished, theme: Theme, show_tool_args: bool
) -> RenderableType:
    return Text(
        f"  ⎿ {event.summary} · {event.duration_s:.1f}s",
        style=theme.style("tool.result"),
    )


def _render_tool_failed(event: ev.ToolFailed, theme: Theme, show_tool_args: bool) -> RenderableType:
    return Text(f"  ⎿ Error: {event.error}", style=theme.style("tool.error"))


def _render_approval_requested(
    event: ev.ApprovalRequested, theme: Theme, show_tool_args: bool
) -> RenderableType:
    args_str = f"({_format_args(event.args)})" if show_tool_args and event.args else ""
    header = Text(f"⏺ Approval needed: {event.tool}{args_str}", style=theme.style("approval.title"))
    reason = Text(f"  ⎿ {event.reason}", style=theme.style("approval.arg"))
    return Group(header, reason)


def _render_approval_decision(
    event: ev.ApprovalDecision, theme: Theme, show_tool_args: bool
) -> RenderableType:
    verbs = {"allow": "Approved", "allow_always": "Approved (always)", "deny": "Denied"}
    verb = verbs[event.outcome]
    style = theme.style("error") if event.outcome == "deny" else theme.style("success")
    return Text(f"  ⎿ {verb}: {event.tool}", style=style)


def _render_notice(event: ev.Notice, theme: Theme, show_tool_args: bool) -> RenderableType:
    prefix = {"info": "·", "warn": "⚠", "error": "✗"}[event.level]
    token = {"info": "notice.info", "warn": "notice.warn", "error": "notice.error"}[event.level]
    return Text(f"{prefix} {event.message}", style=theme.style(token))


def _render_turn_finished(
    event: ev.TurnFinished, theme: Theme, show_tool_args: bool
) -> RenderableType:
    parts: list[RenderableType] = [Markdown(event.answer)]
    if event.citations:
        parts.append(Text(""))
        parts.append(Text(" · ".join(event.citations), style=theme.style("answer.citation")))
    return Group(*parts)


def _render_turn_failed(event: ev.TurnFailed, theme: Theme, show_tool_args: bool) -> RenderableType:
    if event.cancelled:
        lines: list[RenderableType] = [Text("⎿ Interrupted by user", style=theme.style("muted"))]
    else:
        lines = [Text(f"Error: {event.message}", style=theme.style("error"))]
    if event.hint:
        lines.append(Text(f"  {event.hint}", style=theme.style("muted")))
    return Group(*lines)


_HANDLERS: dict[str, Callable[[Any, Theme, bool], RenderableType | None]] = {
    "run_started": _no_visual,
    "plan_created": _render_plan_created,
    "node_started": _no_visual,
    "text_delta": _render_text_delta,
    "tool_started": _render_tool_started,
    "tool_finished": _render_tool_finished,
    "tool_failed": _render_tool_failed,
    "approval_requested": _render_approval_requested,
    "approval_decision": _render_approval_decision,
    "usage_updated": _no_visual,
    "notice": _render_notice,
    "turn_finished": _render_turn_finished,
    "turn_failed": _render_turn_failed,
}
