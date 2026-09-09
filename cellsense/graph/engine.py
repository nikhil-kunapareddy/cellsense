"""``Engine`` -- the public seam between the graph and any front-end.

This is the one class ``cellsense.ui`` (and ``--print`` mode, and every test
in the verification suite) codes against. Nothing downstream of
``stream_turn()`` needs to know LangGraph exists: it consumes
``app.stream(..., stream_mode=["custom", "updates", "messages"])`` internally
and yields nothing but :mod:`cellsense.events` dataclasses.

Two invariants callers may rely on absolutely:

1. The iterator returned by :meth:`Engine.stream_turn` always terminates with
   exactly one :class:`~cellsense.events.TurnFinished` **or** exactly one
   :class:`~cellsense.events.TurnFailed` -- never neither, never both.
2. Setting ``cancel`` mid-stream stops at the next node boundary, yields
   ``TurnFailed(cancelled=True)``, and never leaves a partially-written
   checkpoint (LangGraph only commits a checkpoint once a superstep finishes,
   so simply abandoning the stream generator early is enough).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from langgraph.types import StreamMode as _StreamMode

from cellsense.config import Config
from cellsense.errors import CellSenseError
from cellsense.events import (
    ApprovalDecision,
    ApprovalRequested,
    Event,
    NodeStarted,
    Notice,
    PlanCreated,
    RunStarted,
    TextDelta,
    ToolFailed,
    ToolFinished,
    ToolStarted,
    TurnFailed,
    TurnFinished,
    UsageUpdated,
)
from cellsense.graph.build import build_graph
from cellsense.graph.checkpoint import (
    SessionInfo,
    delete_session,
    list_sessions,
    load_history,
    open_checkpointer,
)
from cellsense.io.context import build_context
from cellsense.io.schema import Citation, Workspace, format_citations, merge_citations
from cellsense.observability.logging import LOGGER_NAME, bind
from cellsense.observability.usage import UsageTracker
from cellsense.permissions import PermissionPolicy
from cellsense.providers.base import translate_provider_error
from cellsense.providers.registry import resolve_model
from cellsense.tools.registry import ToolRegistry

__all__ = ["Engine"]

OnApproval = Callable[[ApprovalRequested], Literal["allow", "allow_always", "deny"]]

_DEFAULT_DB_PATH = Path.home() / ".cellsense" / "sessions.db"

# MUST be a `list`, never a tuple (or any other `Sequence`): LangGraph's
# `Pregel.stream()` only tags each yielded chunk as a `(mode, payload)` pair
# when `isinstance(stream_mode, list)` is true -- a tuple fails that check
# silently, so it yields bare, untagged payloads instead. `Engine.stream_turn`'s
# `for mode, chunk in self._app.stream(...)` then fails to unpack the first
# multi-key custom-event dict that comes through (e.g. guardrail's
# `node_started`) with `ValueError: too many values to unpack (expected 2)`
# -- confirmed live: every real turn failed before this was a `list`. Typed
# as `list[_StreamMode]` (not `list[str]`) so mypy still proves each string is
# a valid `StreamMode`.
STREAM_MODES: list[_StreamMode] = ["custom", "updates", "messages"]


class Engine:
    """Owns one compiled graph, one chat model, and one checkpointer for the
    lifetime of a CellSense process.

    Constructing an ``Engine`` resolves ``config.model.default`` to a concrete
    chat model (raising a :class:`~cellsense.errors.CellSenseError` subclass
    immediately if the provider has no API key or its SDK isn't installed --
    see ``providers.base.build_chat_model``) and opens the SQLite
    checkpointer, so a failed construction never leaves a half-open turn.
    """

    def __init__(
        self,
        workspace: Workspace,
        config: Config,
        *,
        registry: ToolRegistry,
        db_path: Path | None = None,
    ) -> None:
        self._workspace = workspace
        self._config = config
        self._registry = registry
        self._logger = logging.getLogger(LOGGER_NAME)

        resolved = resolve_model(config.model.default, config.model.to_settings())
        self._provider_name = resolved.provider_spec.name
        self._provider_label = resolved.provider_spec.label
        self._model_spec = resolved.model_spec
        self._chat_model = resolved.chat_model  # built now: fail fast, not mid-turn

        self._permissions = PermissionPolicy(
            mode=config.permissions.mode,
            allow_list=frozenset(config.permissions.allow),
        )
        self._usage = UsageTracker()

        self._saver = open_checkpointer(db_path if db_path is not None else _DEFAULT_DB_PATH)
        self._app = build_graph(
            chat_model=self._chat_model,
            tool_registry=registry,
            workspace=workspace,
            permissions=self._permissions,
            provider_name=self._provider_name,
            checkpointer=self._saver,
            max_tool_rounds=config.agent.max_tool_rounds,
            max_subtasks=config.agent.max_subtasks,
            guardrail_enabled=config.agent.guardrail,
            planner_enabled=config.agent.planner,
        )

    # ── identity ─────────────────────────────────────────────────────────────

    @property
    def model_label(self) -> str:
        """The provider-native model id actually sent over the wire."""
        return self._model_spec.id

    @property
    def provider_label(self) -> str:
        """The human-readable provider name, e.g. ``"Groq"``."""
        return self._provider_label

    # ── the seam ─────────────────────────────────────────────────────────────

    def stream_turn(
        self,
        question: str,
        *,
        thread_id: str,
        on_approval: OnApproval,
        cancel: threading.Event | None = None,
    ) -> Iterator[Event]:
        """Run one turn end to end, yielding :mod:`cellsense.events` as it goes.

        See the module docstring for the two hard guarantees this makes.
        """
        start = time.monotonic()
        self._usage.reset_turn()

        # Bound here, at the top of the turn, for whatever thread drives this
        # generator (normally the UI's own thread). Graph nodes re-bind the
        # same fields themselves the moment they run (see graph/nodes.py's
        # `_node_span`) because LangGraph dispatches fan-out `Send` branches
        # (the `worker` node) onto their own threads, and a ContextVar set
        # here never propagates into one of those.
        with bind(thread_id=thread_id):
            self._logger.info(
                "turn started",
                extra={
                    "question_len": len(question),
                    "model": self.model_label,
                    "provider": self.provider_label,
                },
            )
            # The full question text can carry arbitrary user/business data, so
            # it deliberately never appears in the INFO line above -- only here,
            # at DEBUG (off unless `--debug`, and only ever written to the local
            # JSONL file, never printed or shipped anywhere). `question_len`
            # alone is enough to notice something odd (e.g. an accidental huge
            # paste) at the default log level; the full text is for a human
            # reproducing one specific turn with `--debug` already on.
            self._logger.debug("turn question", extra={"question": question})

            yield RunStarted(
                question=question,
                thread_id=thread_id,
                model=self.model_label,
                provider=self.provider_label,
            )

            config: RunnableConfig = RunnableConfig(configurable={"thread_id": thread_id})
            context = build_context(self._workspace)
            # `results`/`citations` are explicitly reset to None (see
            # `state.reset_per_turn`) so this turn starts from an empty
            # accumulator instead of appending onto everything ever recorded on
            # this thread. Every other key here is a plain (unreduced) field, so
            # passing it is a straightforward overwrite for the new turn.
            stream_input: dict[str, Any] | Command = {
                "question": question,
                "context": context,
                "thread_id": thread_id,
                "subtask_id": None,
                "subtasks": [],
                "tool_rounds": 0,
                "relevant": True,
                "rejection": None,
                "answer": "",
                "results": None,
                "citations": None,
            }

            seen_events: set[tuple[str, str]] = set()
            cancelled = False

            try:
                while True:
                    interrupt_payload: dict[str, Any] | None = None

                    for mode, chunk in self._app.stream(
                        stream_input, config, stream_mode=STREAM_MODES
                    ):
                        if cancel is not None and cancel.is_set():
                            cancelled = True
                            break

                        if mode == "custom":
                            event = self._translate_custom(chunk, seen_events)
                            if event is not None:
                                yield event
                            continue

                        if mode == "messages":
                            # Token-level text is already surfaced via the
                            # "text_delta" custom events nodes emit themselves;
                            # this mode would just duplicate it.
                            continue

                        if mode == "updates":
                            if not isinstance(chunk, dict):
                                continue
                            metadata = chunk.get("__metadata__")
                            if isinstance(metadata, dict) and metadata.get("cached"):
                                # A completed sibling branch replaying on resume --
                                # LANGGRAPH_NOTES.md gotcha #2. Not a new update.
                                continue
                            if "__interrupt__" in chunk:
                                interrupt_payload = chunk["__interrupt__"][0].value
                                break
                            continue

                    if cancelled:
                        break
                    if interrupt_payload is None:
                        break

                    request = ApprovalRequested(
                        call_id=str(interrupt_payload.get("call_id", "")),
                        tool=str(interrupt_payload.get("tool", "")),
                        args=dict(interrupt_payload.get("args") or {}),
                        reason=str(interrupt_payload.get("reason", "")),
                    )
                    self._logger.info(
                        "approval requested",
                        extra={"tool": request.tool, "call_id": request.call_id},
                    )
                    yield request
                    decision = on_approval(request)
                    self._logger.info(
                        "approval decided",
                        extra={
                            "tool": request.tool,
                            "call_id": request.call_id,
                            "outcome": decision,
                        },
                    )
                    yield ApprovalDecision(
                        call_id=request.call_id, tool=request.tool, outcome=decision
                    )
                    stream_input = Command(resume=decision)
            except CellSenseError as exc:
                self._logger.warning(
                    "turn failed",
                    extra={
                        "error_type": type(exc).__name__,
                        "error": exc.message,
                        "duration_s": round(time.monotonic() - start, 4),
                    },
                )
                yield TurnFailed(message=exc.message, hint=exc.hint)
                return
            except Exception as exc:  # translate + log; never let a turn crash the process
                self._logger.exception("Unhandled error during turn on thread %s", thread_id)
                translated = translate_provider_error(exc, self._provider_name)
                yield TurnFailed(message=translated.message, hint=translated.hint)
                return

            if cancelled:
                self._logger.warning(
                    "turn cancelled",
                    extra={"duration_s": round(time.monotonic() - start, 4)},
                )
                yield TurnFailed(message="Turn cancelled.", cancelled=True)
                return

            state_values = self._app.get_state(config).values
            answer = _strip_model_citations(str(state_values.get("answer") or ""))
            citations = merge_citations(state_values.get("citations") or [])
            totals = self._usage.turn_totals()
            tool_call_count = len(
                {call_id for kind, call_id in seen_events if kind == "tool_started"}
            )

            self._logger.info(
                "turn finished",
                extra={
                    "duration_s": round(time.monotonic() - start, 4),
                    "input_tokens": totals.input_tokens,
                    "output_tokens": totals.output_tokens,
                    "cost_usd": totals.cost_usd,
                    "tool_call_count": tool_call_count,
                    "citation_count": len(citations),
                },
            )
            yield TurnFinished(
                answer=answer,
                citations=_citation_strings(citations),
                duration_s=time.monotonic() - start,
                input_tokens=totals.input_tokens,
                output_tokens=totals.output_tokens,
                cost_usd=totals.cost_usd,
            )

    # ── sessions ─────────────────────────────────────────────────────────────

    def sessions(self) -> list[SessionInfo]:
        """Every thread with at least one checkpoint, most recently used first."""
        return list_sessions(self._saver)

    def history(self, thread_id: str) -> list[tuple[str, str]]:
        """The full ``[(role, text), ...]`` transcript for one thread."""
        return load_history(self._saver, thread_id)

    def delete_session(self, thread_id: str) -> None:
        """Permanently remove one thread's checkpoints."""
        delete_session(self._saver, thread_id)

    # ── private: custom-event translation ───────────────────────────────────

    def _translate_custom(self, chunk: Any, seen_events: set[tuple[str, str]]) -> Event | None:
        """Turn one ``get_stream_writer()`` dict from ``graph.nodes`` into the
        matching :mod:`cellsense.events` dataclass, de-duplicating tool/approval
        events by ``(kind, call_id)`` per LANGGRAPH_NOTES.md gotcha #1.
        """
        if not isinstance(chunk, dict):
            return None
        kind = chunk.get("type")
        subtask_id = chunk.get("subtask_id")

        if kind == "node_started":
            return NodeStarted(node=str(chunk.get("node", "")), subtask_id=subtask_id)

        if kind == "text_delta":
            return TextDelta(
                text=str(chunk.get("text", "")),
                channel=chunk.get("channel", "answer"),
                subtask_id=subtask_id,
            )

        if kind == "plan_created":
            return PlanCreated(subtasks=list(chunk.get("subtasks") or []))

        if kind == "notice":
            return Notice(message=str(chunk.get("message", "")), level=chunk.get("level", "info"))

        if kind == "usage":
            self._usage.record(
                self._model_spec,
                int(chunk.get("input_tokens", 0)),
                int(chunk.get("output_tokens", 0)),
            )
            totals = self._usage.turn_totals()
            return UsageUpdated(
                input_tokens=totals.input_tokens,
                output_tokens=totals.output_tokens,
                cost_usd=totals.cost_usd,
            )

        if kind in ("tool_started", "tool_finished", "tool_failed"):
            call_id = str(chunk.get("call_id", ""))
            key = (kind, call_id)
            if key in seen_events:
                return None
            seen_events.add(key)
            name = str(chunk.get("name", ""))

            if kind == "tool_started":
                return ToolStarted(
                    call_id=call_id,
                    name=name,
                    args=dict(chunk.get("args") or {}),
                    subtask_id=subtask_id,
                )
            if kind == "tool_finished":
                return ToolFinished(
                    call_id=call_id,
                    name=name,
                    summary=str(chunk.get("summary", "")),
                    row_count=int(chunk.get("row_count", 0)),
                    duration_s=float(chunk.get("duration_s", 0.0)),
                    preview=list(chunk.get("preview") or []),
                    subtask_id=subtask_id,
                )
            return ToolFailed(
                call_id=call_id,
                name=name,
                error=str(chunk.get("error", "")),
                duration_s=float(chunk.get("duration_s", 0.0)),
                subtask_id=subtask_id,
            )

        return None


def _citation_strings(citations: list[Citation]) -> list[str]:
    """Render already-merged citations as one string per source, e.g.
    ``"sales.xlsx [Sheet: Q1, Rows: 3, 7-9]"`` or ``"sales.csv [240 rows]"``.

    Delegates to :func:`io.schema.format_citations` rather than formatting
    here. ``format_citations`` is the single renderer by contract
    (ARCHITECTURE.md 2, "Citation rendering rule"): it collapses consecutive
    indices into ranges and, past a threshold, substitutes an honest count.
    This function only reshapes that one string into the per-source list
    ``TurnFinished.citations`` carries, by stripping the constant ``"Sources: "``
    prefix and splitting on the constant ``" | "`` separator.

    An earlier version duplicated the formatting logic here, because
    ``merge_citations`` used to cap each citation at 50 row indices and so
    could not report a true total. That cap has since been removed, so the
    fork is obsolete -- and it had already drifted, rendering a 240-row
    aggregate as ``[Rows: 0-239]`` where ``format_citations`` said
    ``[240 rows]``.
    """
    if not citations:
        return []
    block = format_citations(citations)
    prefix = "Sources: "
    body = block[len(prefix) :] if block.startswith(prefix) else block
    return [part.strip() for part in body.split(" | ") if part.strip()]


# Matches a model-authored citation line such as ``Sources: a.csv [Rows: 1]``,
# ``**Sources:**  ...`` or ``_Sources:_ ...``. Anchored at line start and tolerant
# of the markdown emphasis models like to add, but deliberately NOT matching a
# mid-sentence "sources:" so ordinary prose is never truncated.
_MODEL_SOURCES_LINE_RE = re.compile(r"^\s*[*_]{0,2}\s*Sources\s*[*_]{0,2}\s*:", re.IGNORECASE)


def _strip_model_citations(answer: str) -> str:
    """Remove a trailing model-authored ``Sources: ...`` line from ``answer``.

    ``prompts.SYSTEM_PROMPT`` instructs the model to end every answer with a
    citation block -- useful as a self-check while it is reasoning, but a
    live run showed it can be outright wrong (a model once cited only the
    first row of a 240-row aggregate) and, even when accurate, it duplicates
    the *authoritative* citations this module already computes from real tool
    provenance (see ``_citation_strings``) and reports separately via
    ``TurnFinished.citations``. Rather than show both -- one trustworthy, one
    not, with no way for a reader to tell which is which -- the model's own
    line is dropped here before the answer is ever yielded.
    """
    lines = answer.splitlines()
    cut: int | None = None
    for index, line in enumerate(lines):
        if _MODEL_SOURCES_LINE_RE.match(line):
            cut = index
    if cut is None:
        return answer.strip()
    return "\n".join(lines[:cut]).rstrip()
