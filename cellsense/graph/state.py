"""``TurnState``: the mutable state threaded through every node of one turn.

Every node reads a subset of these fields and returns a partial dict of the
ones it updates; LangGraph merges partial returns into this shape using the
reducer attached to each ``Annotated`` field (plain fields are last-write-wins,
which is safe here because only one branch at a time ever writes to them --
see the fan-out note on ``WorkerState`` below).

``TurnState`` is deliberately one flat TypedDict for the whole graph (guardrail,
planner, the top-level ``agent``/``tools`` cycle, ``worker``, ``synthesize``) --
see LANGGRAPH_NOTES.md mechanic #2. A ``Send("worker", payload)`` call does not
have to supply every key; ``total=False`` makes every field optional so a
fan-out payload can be a narrow slice (documented precisely by
:class:`WorkerState`) without upsetting the type checker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, TypedDict, TypeVar

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from cellsense.io.schema import Citation

__all__ = ["SubtaskResult", "TurnState", "WorkerState", "reset_per_turn"]

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class SubtaskResult:
    """One fan-out worker's contribution, ready for :func:`nodes.synthesize`.

    Plain dataclass rather than another TypedDict because ``results`` entries
    are never partially updated -- a worker produces exactly one of these, in
    one shot, or not at all (if it errors, see ``nodes.worker``'s fail-soft
    handling) -- so there is no reducer-merge behavior to model.
    """

    subtask_id: str
    question: str
    answer: str


def reset_per_turn(existing: list[_T] | None, new: list[_T] | None) -> list[_T]:
    """Reducer for ``results``/``citations``: accumulate like ``operator.add``
    within one turn, but let an explicit ``None`` reset the list to empty.

    Both fields persist in the checkpointer across a whole multi-turn thread
    (that is the point of the checkpointer), but they must each represent
    *this turn's* fan-out, not a running total since the thread began.
    ``Engine.stream_turn`` sends ``None`` for both exactly once, as part of
    the very first input of every new turn -- never mid-turn, and never from
    a node's own return value -- so this is the one place old turns' entries
    get dropped instead of silently accumulating forever (which would corrupt
    ``nodes.synthesize``'s merge on turn 2 onward, and inflate
    ``TurnFinished.citations``).
    """
    if new is None:
        return []
    return [*(existing or []), *new]


class TurnState(TypedDict, total=False):
    """The single state dict threaded through every node in the turn graph.

    Fields:
        messages: The **top-level** chat transcript -- system/human/AI/tool
            messages for the direct (single-subtask) ``agent``/``tools``
            cycle only. Fan-out workers keep their own message list as a
            local Python variable inside ``nodes.worker`` and never write it
            back here (see :class:`WorkerState`'s docstring for why): mixing
            every worker's internal tool-calling chatter into one shared,
            checkpointed list would make ``Engine.history()`` unreadable and,
            since ``add_messages`` has no way to un-merge branches, would
            leak worker prompts into what looks like the user-facing
            conversation. Threaded through ``add_messages`` so replays and
            resumes merge/dedupe by message ``id`` instead of colliding
            (LANGGRAPH_NOTES.md's ``MessagesState`` gotcha).
        question: The original user question for this turn, verbatim. Never
            overwritten after ``engine.py`` sets it at the top of the turn.
        context: The compressed workspace schema digest
            (``cellsense.io.context.build_context()``) injected into every
            system/planner/guardrail prompt.
        thread_id: The checkpointer session key. Carried in state (not just
            graph ``config``) so any node can attach it to a log line or
            event without reaching into ``get_config()``.
        relevant: Guardrail verdict. Absent or ``True`` means "proceed";
            guardrail failures must also set this to ``True`` (fail open --
            see ``nodes.guardrail``) so a broken classifier never blocks real
            work.
        rejection: Set only when ``relevant`` is ``False``: the message shown
            to the user in lieu of running any tools. ``build.py`` routes
            straight to ``END`` in that case and ``engine.py`` reports it as
            a (successful, zero-cost) ``TurnFinished``.
        subtasks: Planner output, ``[(subtask_id, sub_question), ...]``. A
            single-element list means "do not decompose"; the graph then
            takes the direct ``agent`` path and bypasses ``synthesize``
            entirely for latency (ARCHITECTURE.md §6).
        subtask_id: ``None``/absent on the top-level branch; set to the
            owning subtask's id inside each fan-out worker invocation so
            events and citations can be attributed to it.
        tool_rounds: Count of agent<->tools round trips on the top-level
            branch. Compared against ``config.agent.max_tool_rounds`` inside
            ``nodes.agent``; fan-out workers keep their own local counter
            instead (again, never written back here).
        results: Fan-out worker outputs, ``operator.add``-reduced so however
            many parallel ``Send`` branches ran, their single-element lists
            concatenate into one list on the parent without a lock or a race.
        citations: Citations gathered on the top-level branch, or merged from
            every fan-out branch. ``operator.add``-reduced for the same
            reason as ``results``; the real de-duplication (grouping by
            file/sheet, unioning row indices) happens once, in
            ``nodes.synthesize`` or ``engine.py``, via
            ``cellsense.io.schema.merge_citations`` -- reducers stay dumb
            concatenation so they compose safely with checkpoint replay.
        answer: The final answer text for the turn. Set directly by the
            top-level ``agent`` node (single-subtask path) or by
            ``nodes.synthesize`` (fan-out path).
    """

    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    context: str
    thread_id: str
    relevant: bool
    rejection: str | None
    subtasks: list[tuple[str, str]]
    subtask_id: str | None
    tool_rounds: int
    results: Annotated[list[SubtaskResult] | None, reset_per_turn]
    citations: Annotated[list[Citation] | None, reset_per_turn]
    answer: str


class WorkerState(TypedDict, total=False):
    """Documents the narrow slice of :class:`TurnState` a ``Send("worker", ...)``
    payload actually needs to supply.

    ``nodes.worker`` runs an entire independent agent loop (model calls +
    tool execution) using **local** Python variables for its own messages and
    tool-round counter -- it is not a separate compiled subgraph with its own
    schema, just a plain node function, because LangGraph only checks
    concurrent-write conflicts against keys a node's *return value* touches
    (LANGGRAPH_NOTES.md mechanic #2). As long as every worker's return dict
    only ever updates the reducer-guarded ``results``/``citations`` keys, any
    number of them can run in the same superstep with no risk of the
    "can receive only one value per step" error a shared, unreduced field
    (like a naively shared ``messages`` or ``tool_rounds``) would trigger.

    This type exists purely so the fan-out router in ``nodes.py`` has a
    documented, narrower shape to construct instead of a full (mostly
    irrelevant) ``TurnState`` dict -- LangGraph itself only ever sees
    ``TurnState``, since ``worker`` is a node in that same graph.
    """

    question: str
    context: str
    thread_id: str
    subtask_id: str
    results: Annotated[list[SubtaskResult] | None, reset_per_turn]
    citations: Annotated[list[Citation] | None, reset_per_turn]
