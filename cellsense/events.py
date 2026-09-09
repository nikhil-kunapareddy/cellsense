"""The event protocol between the execution engine and any front-end.

``Engine.stream_turn()`` yields a stream of these dataclasses. The terminal UI in
:mod:`cellsense.ui` renders them; ``--print`` mode collapses them to plain text;
tests assert on them directly. Nothing in :mod:`cellsense.graph` may import from
:mod:`cellsense.ui`, and nothing in :mod:`cellsense.ui` may import from
:mod:`cellsense.graph` -- this module is the only shared vocabulary.

Adding a field to an existing event is backwards compatible. Adding a new event
type requires a matching branch in ``cellsense.ui.render.render_event``; the
renderer must ignore unknown events rather than raise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "ApprovalDecision",
    "ApprovalRequested",
    "Event",
    "NodeStarted",
    "Notice",
    "PlanCreated",
    "RunStarted",
    "TextDelta",
    "ToolFailed",
    "ToolFinished",
    "ToolStarted",
    "TurnFailed",
    "TurnFinished",
    "UsageUpdated",
]


@dataclass(slots=True)
class Event:
    """Base class. ``kind`` is a stable string usable for switch-style dispatch."""

    kind: str = field(init=False, default="event")


# ── Lifecycle ──────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class RunStarted(Event):
    """Emitted once at the top of every turn."""

    question: str
    thread_id: str
    model: str
    provider: str
    kind: str = field(init=False, default="run_started")


@dataclass(slots=True)
class PlanCreated(Event):
    """The planner decomposed the question. ``subtasks`` is ``[(id, question)]``.

    A single-element list means the planner chose not to decompose; the UI should
    stay quiet in that case rather than render a one-item plan.
    """

    subtasks: list[tuple[str, str]]
    kind: str = field(init=False, default="plan_created")


@dataclass(slots=True)
class NodeStarted(Event):
    """A graph node began. ``subtask_id`` is set for fan-out workers."""

    node: str
    subtask_id: str | None = None
    kind: str = field(init=False, default="node_started")


# ── Model output ───────────────────────────────────────────────────────────────


@dataclass(slots=True)
class TextDelta(Event):
    """An incremental chunk of assistant text.

    ``channel`` distinguishes the final answer from intermediate worker chatter so
    the UI can dim or hide the latter.
    """

    text: str
    channel: Literal["answer", "worker", "reasoning"] = "answer"
    subtask_id: str | None = None
    kind: str = field(init=False, default="text_delta")


# ── Tools ──────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ToolStarted(Event):
    call_id: str
    name: str
    args: dict[str, Any]
    subtask_id: str | None = None
    kind: str = field(init=False, default="tool_started")


@dataclass(slots=True)
class ToolFinished(Event):
    """A tool succeeded.

    ``summary`` is the one-line human description from ``ToolResult.summary``.
    ``preview`` is a small list-of-row-dicts (at most 5 rows) for table rendering;
    it is never the full result set.
    """

    call_id: str
    name: str
    summary: str
    row_count: int
    duration_s: float
    preview: list[dict[str, Any]] = field(default_factory=list)
    subtask_id: str | None = None
    kind: str = field(init=False, default="tool_finished")


@dataclass(slots=True)
class ToolFailed(Event):
    call_id: str
    name: str
    error: str
    duration_s: float
    subtask_id: str | None = None
    kind: str = field(init=False, default="tool_failed")


# ── Human in the loop ──────────────────────────────────────────────────────────


@dataclass(slots=True)
class ApprovalRequested(Event):
    """A permission-gated tool is waiting on the user.

    The engine yields this event *and then* calls the ``on_approval`` callback
    passed to ``stream_turn``. The event is informational (so ``--print`` mode and
    tests can log it); the callback's return value is what actually decides.
    """

    call_id: str
    tool: str
    args: dict[str, Any]
    reason: str
    kind: str = field(init=False, default="approval_requested")


@dataclass(slots=True)
class ApprovalDecision(Event):
    """The resolution of an :class:`ApprovalRequested`."""

    call_id: str
    tool: str
    outcome: Literal["allow", "allow_always", "deny"]
    kind: str = field(init=False, default="approval_decision")


# ── Accounting ─────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class UsageUpdated(Event):
    """Cumulative token/cost totals for the current turn.

    ``cost_usd`` is ``None`` when the model has no known price entry.
    """

    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    kind: str = field(init=False, default="usage_updated")


@dataclass(slots=True)
class Notice(Event):
    """Out-of-band message: a retry, a guardrail rejection, a degraded fallback."""

    message: str
    level: Literal["info", "warn", "error"] = "info"
    kind: str = field(init=False, default="notice")


# ── Terminal states ────────────────────────────────────────────────────────────


@dataclass(slots=True)
class TurnFinished(Event):
    """The turn completed successfully. Always the last event of a good turn."""

    answer: str
    citations: list[str]
    duration_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    kind: str = field(init=False, default="turn_finished")


@dataclass(slots=True)
class TurnFailed(Event):
    """The turn aborted. Always the last event of a failed turn."""

    message: str
    hint: str | None = None
    cancelled: bool = False
    kind: str = field(init=False, default="turn_failed")
