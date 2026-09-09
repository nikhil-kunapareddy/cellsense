"""The graph's nodes: guardrail, planner, agent, tools, worker, synthesize.

Every node is a small **factory** -- ``make_x_node(...)`` closes over the
dependencies that are fixed for the lifetime of one compiled graph (the chat
model, the :class:`~cellsense.tools.registry.ToolRegistry`, the
:class:`~cellsense.io.schema.Workspace`, the
:class:`~cellsense.permissions.PermissionPolicy`, the provider's name for
error translation, and the configured limits) and returns the plain
``(TurnState) -> dict`` function LangGraph actually calls. This is what lets
``build.py`` wire a graph against a stub chat model in tests with no
LangGraph-specific test harness required, and what lets ``engine.py`` build
one graph per session without any node reaching into global state.

Every node emits a ``{"type": ..., ...}`` dict through ``get_stream_writer()``
at least once (a ``"node_started"`` marker); ``graph.engine.Engine`` is the
only place that interprets these dicts and turns them into
:mod:`cellsense.events` dataclasses -- nodes never import ``cellsense.events``
themselves, keeping the wire format private to this package.

Two node shapes matter for how fan-out works (see ``state.py``'s docstrings):

* ``agent``/``tools`` cycle back and forth on the **top-level** branch only,
  writing to the shared ``messages``/``tool_rounds``/``answer`` fields of
  :class:`~cellsense.graph.state.TurnState`. There is only ever one such
  branch active at a time, so plain (non-reduced) fields are safe to use.
* ``worker`` runs an entire independent agent loop -- model calls and tool
  execution -- using local Python variables, and returns only the
  reducer-guarded ``results``/``citations`` keys. Any number of ``worker``
  invocations can run concurrently (one per fan-out ``Send``) without ever
  triggering LangGraph's "can receive only one value per step" error, because
  they never touch an unreduced shared key.

Permission gating (ARCHITECTURE.md §6, LANGGRAPH_NOTES.md gotcha #1): for any
tool with ``side_effect=True`` that the active :class:`PermissionPolicy`
doesn't already allow, ``_run_tool_calls`` calls ``interrupt()`` *immediately
before* the one call to ``ToolRegistry.run()`` that would perform the side
effect -- never before -- so that if this node re-executes from the top after
``Command(resume=...)``, only the actual disk-writing call happens exactly
once (the aborted, pre-interrupt pass never reaches a ``return`` statement, so
nothing it computed is ever committed to graph state).

Provider-error handling (added after a live run against Groq surfaced two
real defects -- see ``_stream_and_merge``/``_invoke_with_retry`` and
``_force_final_answer``): every model call in this module goes through
``providers.base.translate_provider_error`` before a caller ever sees an
exception, and rate limits get a bounded, backoff-respecting retry that
honours the provider's own "try again in Ns" hint rather than failing the
turn on the first 429.

Logging (observability): every node re-binds ``thread_id``/``node``/
``subtask_id`` via ``observability.logging.bind`` right at its own entry --
not once at the top of the graph -- because LangGraph's fan-out ``Send``
branches (the ``worker`` node) run on their own thread, and ContextVars do
not propagate into a *new* thread (see ``observability/logging.py``'s module
docstring). Node entry/exit is DEBUG-level tracing; anything that indicates a
turn degraded but kept going (guardrail failing open, the planner falling
back, a rate-limit retry, the tool-round cap, the forced-final-answer
fallback) is logged at WARNING so it is visible without ``--debug``.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, cast

from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable
from langgraph.graph import END
from langgraph.types import Send, interrupt

from cellsense.errors import CellSenseError, RateLimitError
from cellsense.graph.prompts import (
    GUARDRAIL_SYSTEM,
    PLANNER_SYSTEM,
    REJECTION_MESSAGE,
    SYNTHESIZER_USER,
    SYSTEM_PROMPT,
)
from cellsense.graph.state import SubtaskResult, TurnState
from cellsense.io.schema import Citation, Workspace, merge_citations
from cellsense.observability.logging import LOGGER_NAME, bind
from cellsense.observability.usage import extract_usage
from cellsense.permissions import PermissionPolicy
from cellsense.providers.base import translate_provider_error
from cellsense.tools.registry import ToolRegistry

try:  # pragma: no cover - import path is exercised by every real invocation
    from langgraph.config import get_stream_writer
except ImportError:  # pragma: no cover - defensive; see LANGGRAPH_NOTES.md

    def get_stream_writer() -> Callable[[Any], None]:
        return lambda _event: None


__all__ = [
    "make_agent_node",
    "make_guardrail_node",
    "make_planner_node",
    "make_route_after_agent",
    "make_synthesize_node",
    "make_tools_node",
    "make_worker_node",
    "route_after_guardrail",
    "route_after_planner",
]

Node = Callable[[TurnState], dict[str, Any]]

# `chat_model.bind_tools(...)` returns a bound `Runnable`, not a `BaseChatModel`
# -- structurally still "something with .invoke()/.stream() over chat messages",
# which is exactly what `Runnable[LanguageModelInput, BaseMessage]` (the same
# shape `langchain_core` itself uses for `LanguageModelLike`, narrowed from
# `BaseMessage | str` to `BaseMessage` since every call site here only ever
# talks to chat models) describes. Every helper below that takes either a raw
# `BaseChatModel` or a bound bindings-wrapper is typed against this instead.
ChatModelLike = Runnable[LanguageModelInput, BaseMessage]

# Retry policy for rate-limited model calls (BUG 2 in the live-run report: a
# 429 used to surface raw and untranslated). Deliberately small and bounded --
# this is a CLI turn a human is waiting on, not a batch job -- and only ever
# applied to a translated `RateLimitError`; anything else (bad request, auth,
# not-found) is raised immediately since retrying an already-rejected request
# just wastes the same error three more times.
_MAX_RETRIES = 3
_MAX_RETRY_WAIT_S = 60.0
_DEFAULT_RETRY_WAIT_S = 5.0

# Matches the "try again in 24.29s" (Groq) / "retry in 45.3s" (Gemini) /
# "retryDelay": "45s" (Gemini's structured detail) shapes seen live.
_RETRY_AFTER_RE = re.compile(
    r"(?:try again in|retry in|retrydelay['\"]?\s*:\s*['\"]?)\s*([\d.]+)\s*s", re.IGNORECASE
)

logger = logging.getLogger(LOGGER_NAME)


@contextmanager
def _node_span(node: str, *, thread_id: str, subtask_id: str | None = None) -> Iterator[None]:
    """Bind ``thread_id``/``node``/``subtask_id`` for every log line emitted
    inside one node invocation, and log a DEBUG entry/exit pair around it.

    Re-bound at the top of *every* node -- rather than once at the top of the
    turn -- because the ``worker`` node runs on a thread LangGraph spins up
    for each fan-out ``Send``, and ``observability.logging``'s ContextVar
    does not propagate into a new thread; calling ``bind()`` again here, in
    whatever thread this node actually executes on, is what makes the fields
    show up correctly regardless of which node it is.
    """
    with bind(thread_id=thread_id, node=node, subtask_id=subtask_id):
        start = time.monotonic()
        logger.debug("node entry")
        try:
            yield
        finally:
            logger.debug("node exit", extra={"duration_s": round(time.monotonic() - start, 4)})


# ── guardrail ────────────────────────────────────────────────────────────────


def make_guardrail_node(chat_model: BaseChatModel, *, enabled: bool, provider_name: str) -> Node:
    """One cheap, tool-free LLM call rejecting off-topic questions.

    Must fail open: any exception from the LLM call (bad key, timeout,
    malformed response, a rate limit that exhausts its retries) is treated as
    "relevant" so a broken guardrail never blocks real work. Skippable
    entirely via ``config.agent.guardrail = false``.
    """

    def guardrail(state: TurnState) -> dict[str, Any]:
        writer = get_stream_writer()
        writer({"type": "node_started", "node": "guardrail", "subtask_id": None})

        with _node_span("guardrail", thread_id=state.get("thread_id", "")):
            # guardrail runs exactly once at the top of every turn (single-subtask
            # or fan-out alike), so this is the one place that appends this turn's
            # HumanMessage -- and, on the very first turn of a thread, the system
            # prompt -- to the top-level, checkpointed transcript. Neither `agent`
            # nor `worker` re-seed it: `agent` reads `state["messages"]` as-is, and
            # `worker` keeps its own local, per-sub-task message list entirely.
            question = state.get("question", "")
            new_messages: list[BaseMessage] = []
            if not state.get("messages"):
                new_messages.append(
                    SystemMessage(content=SYSTEM_PROMPT.format(context=state.get("context", "")))
                )
            new_messages.append(HumanMessage(content=question))

            if not enabled:
                logger.debug("guardrail disabled")
                return {"relevant": True, "messages": new_messages}

            try:
                prompt = GUARDRAIL_SYSTEM.format(context=state.get("context", ""))
                response = _invoke_with_retry(
                    chat_model,
                    [SystemMessage(content=prompt), HumanMessage(content=question)],
                    writer=writer,
                    provider_name=provider_name,
                )
                _emit_usage(writer, response)
                relevant = _text(response).strip().lower().startswith("yes")
            except Exception as exc:  # fail open -- see docstring above
                logger.warning("guardrail check failed; failing open", extra={"error": str(exc)})
                writer(
                    {
                        "type": "notice",
                        "level": "warn",
                        "message": f"Guardrail check failed ({exc}); proceeding without it.",
                    }
                )
                return {"relevant": True, "messages": new_messages}

            logger.info("guardrail verdict", extra={"relevant": relevant})
            if relevant:
                return {"relevant": True, "messages": new_messages}

            writer(
                {"type": "notice", "level": "info", "message": "Question rejected as off-topic."}
            )
            new_messages.append(AIMessage(content=REJECTION_MESSAGE))
            return {
                "relevant": False,
                "rejection": REJECTION_MESSAGE,
                "answer": REJECTION_MESSAGE,
                "messages": new_messages,
            }

    return guardrail


def route_after_guardrail(state: TurnState) -> str:
    return "planner" if state.get("relevant", True) else END


# ── planner ──────────────────────────────────────────────────────────────────


def make_planner_node(
    chat_model: BaseChatModel, *, enabled: bool, max_subtasks: int, provider_name: str
) -> Node:
    """One LLM call that decomposes the question into independent sub-tasks.

    Malformed JSON, an empty array, or any other planner failure (including a
    rate limit that exhausts its retries) falls back to a single sub-task
    containing the original question -- the planner must never crash the
    turn. Skippable via ``config.agent.planner = false``.
    """

    def planner(state: TurnState) -> dict[str, Any]:
        writer = get_stream_writer()
        writer({"type": "node_started", "node": "planner", "subtask_id": None})
        thread_id = state.get("thread_id", "")
        question = state.get("question", "")
        fallback: list[tuple[str, str]] = [("t1", question)]

        with _node_span("planner", thread_id=thread_id):
            if not enabled:
                writer({"type": "plan_created", "subtasks": fallback})
                logger.info(
                    "planner disabled", extra={"subtask_count": len(fallback), "fell_back": False}
                )
                return {"subtasks": fallback}

            fell_back = False
            try:
                prompt = PLANNER_SYSTEM.format(
                    context=state.get("context", ""), max_subtasks=max_subtasks
                )
                response = _invoke_with_retry(
                    chat_model,
                    [SystemMessage(content=prompt), HumanMessage(content=question)],
                    writer=writer,
                    provider_name=provider_name,
                )
                _emit_usage(writer, response)
                items = _parse_plan(_text(response))
                subtasks = [(str(item["id"]), str(item["question"])) for item in items]
                subtasks = subtasks[:max_subtasks] or fallback
            except Exception as exc:  # never crash -- fall back to a single sub-task
                fell_back = True
                logger.warning(
                    "planner failed; falling back to a single sub-task",
                    extra={"error": str(exc)},
                )
                writer(
                    {
                        "type": "notice",
                        "level": "warn",
                        "message": f"Planner failed ({exc}); proceeding with a single sub-task.",
                    }
                )
                subtasks = fallback

            logger.info(
                "planner produced subtasks",
                extra={"subtask_count": len(subtasks), "fell_back": fell_back},
            )
            writer({"type": "plan_created", "subtasks": subtasks})
            return {"subtasks": subtasks}

    return planner


def _parse_plan(raw: str) -> list[dict[str, Any]]:
    """Parse the planner's JSON array, tolerating a markdown code fence."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned[:4].lower() == "json":
            cleaned = cleaned[4:]
    items = json.loads(cleaned)
    if not isinstance(items, list) or not items:
        raise ValueError("planner did not return a non-empty JSON array")
    for item in items:
        if not isinstance(item, dict) or "id" not in item or "question" not in item:
            raise ValueError("planner item missing 'id' or 'question'")
    return items


def route_after_planner(state: TurnState) -> str | list[Send]:
    """Fan-out router: one sub-task goes straight to ``agent``; several fan
    out to independent ``worker`` invocations via ``Send``.
    """
    subtasks = state.get("subtasks") or [("t1", state.get("question", ""))]
    if len(subtasks) <= 1:
        return "agent"

    context = state.get("context", "")
    thread_id = state.get("thread_id", "")
    return [
        Send(
            "worker",
            {
                "question": sub_question,
                "context": context,
                "thread_id": thread_id,
                "subtask_id": sub_id,
                "results": [],
                "citations": [],
            },
        )
        for sub_id, sub_question in subtasks
    ]


# ── agent (top-level single-subtask path) ───────────────────────────────────


def make_agent_node(
    chat_model: BaseChatModel,
    tool_registry: ToolRegistry,
    *,
    max_tool_rounds: int,
    provider_name: str,
) -> Node:
    """Binds tools to ``chat_model`` and drives the top-level answer loop.

    Streams token deltas as they arrive (``get_stream_writer()``) and caps
    itself at ``max_tool_rounds``: once exceeded, it emits a ``Notice`` and
    forces a final, tool-free answer via ``_force_final_answer`` instead of
    looping forever or erroring.
    """
    tools = tool_registry.langchain_tools()
    bound_model = chat_model.bind_tools(tools) if tools else chat_model

    def agent(state: TurnState) -> dict[str, Any]:
        writer = get_stream_writer()
        subtask_id = state.get("subtask_id")
        writer({"type": "node_started", "node": "agent", "subtask_id": subtask_id})

        with _node_span("agent", thread_id=state.get("thread_id", ""), subtask_id=subtask_id):
            # `guardrail` already seeded the system prompt (once per thread) and
            # this turn's HumanMessage; read the transcript as-is.
            messages = list(state.get("messages") or [])
            rounds = state.get("tool_rounds", 0) + 1
            channel = "worker" if subtask_id else "answer"

            if rounds > max_tool_rounds:
                logger.warning(
                    "tool round cap reached; forcing final answer",
                    extra={"tool_rounds": rounds, "max_tool_rounds": max_tool_rounds},
                )
                writer(
                    {
                        "type": "notice",
                        "level": "warn",
                        "message": (
                            f"Reached the {max_tool_rounds}-round tool limit; "
                            "answering with what is known so far."
                        ),
                    }
                )
                text, extra_messages = _force_final_answer(
                    chat_model,
                    messages,
                    writer=writer,
                    channel=channel,
                    subtask_id=subtask_id,
                    provider_name=provider_name,
                )
                return {"messages": extra_messages, "answer": text, "tool_rounds": rounds}

            response = _stream_and_merge(
                bound_model,
                messages,
                writer=writer,
                channel=channel,
                subtask_id=subtask_id,
                provider_name=provider_name,
            )
            update: dict[str, Any] = {"messages": [response], "tool_rounds": rounds}
            if not getattr(response, "tool_calls", None):
                update["answer"] = _text(response)
            return update

    return agent


def make_route_after_agent(max_tool_rounds: int) -> Callable[[TurnState], str]:
    """Factory (not a bare function) so the router enforces the same cap
    ``agent`` used to decide whether to force a tool-free answer -- belt and
    suspenders against a model that echoes a stale/ignored tool call even
    after being asked, in that same round, to answer without tools.
    """

    def route_after_agent(state: TurnState) -> str:
        if state.get("tool_rounds", 0) > max_tool_rounds:
            return END
        last = _last_message(state)
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    return route_after_agent


# ── tools (top-level single-subtask path) ───────────────────────────────────


def make_tools_node(
    tool_registry: ToolRegistry, workspace: Workspace, permissions: PermissionPolicy
) -> Node:
    """Executes every tool call on the last message through
    ``ToolRegistry.run()``, gated by ``permissions`` for side-effecting tools.
    """

    def tools_node(state: TurnState) -> dict[str, Any]:
        writer = get_stream_writer()
        subtask_id = state.get("subtask_id")
        writer({"type": "node_started", "node": "tools", "subtask_id": subtask_id})
        with _node_span("tools", thread_id=state.get("thread_id", ""), subtask_id=subtask_id):
            last = _last_message(state)
            tool_calls = list(getattr(last, "tool_calls", None) or [])
            messages, citations = _run_tool_calls(
                tool_calls,
                tool_registry=tool_registry,
                workspace=workspace,
                permissions=permissions,
                subtask_id=subtask_id,
                writer=writer,
            )
            return {"messages": messages, "citations": citations}

    return tools_node


# ── worker (fan-out path: one independent agent loop per sub-task) ─────────


def make_worker_node(
    chat_model: BaseChatModel,
    tool_registry: ToolRegistry,
    workspace: Workspace,
    permissions: PermissionPolicy,
    *,
    max_tool_rounds: int,
    provider_name: str,
) -> Node:
    """Runs a full, self-contained agent loop for one fan-out sub-task.

    Deliberately a single node (not a subgraph with its own schema): it keeps
    its conversation and round counter as local Python variables and returns
    only ``results``/``citations`` -- the two reducer-guarded ``TurnState``
    keys -- so any number of these can run concurrently in the same
    superstep with no risk of a conflicting-write error (see the module
    docstring and ``state.WorkerState``).
    """
    tools = tool_registry.langchain_tools()
    bound_model = chat_model.bind_tools(tools) if tools else chat_model

    def worker(state: TurnState) -> dict[str, Any]:
        writer = get_stream_writer()
        subtask_id = state.get("subtask_id") or "t1"
        question = state.get("question", "")
        writer({"type": "node_started", "node": "worker", "subtask_id": subtask_id})

        # This node body runs on whatever thread LangGraph dispatches this
        # fan-out `Send` to -- `_node_span` re-binds thread_id/node/subtask_id
        # here rather than relying on any context bound by the caller, per its
        # own docstring and observability/logging.py's ContextVar caveat.
        with _node_span("worker", thread_id=state.get("thread_id", ""), subtask_id=subtask_id):
            messages: list[BaseMessage] = [
                SystemMessage(content=SYSTEM_PROMPT.format(context=state.get("context", ""))),
                HumanMessage(content=question),
            ]
            citations: list[Citation] = []
            answer = ""

            for round_num in range(1, max_tool_rounds + 2):
                at_cap = round_num > max_tool_rounds
                if at_cap:
                    logger.warning(
                        "tool round cap reached; forcing final answer",
                        extra={"tool_rounds": round_num, "max_tool_rounds": max_tool_rounds},
                    )
                    writer(
                        {
                            "type": "notice",
                            "level": "warn",
                            "message": (
                                f"Sub-task {subtask_id!r} reached the {max_tool_rounds}-round "
                                "tool limit; answering with what is known so far."
                            ),
                        }
                    )
                    text, extra_messages = _force_final_answer(
                        chat_model,
                        messages,
                        writer=writer,
                        channel="worker",
                        subtask_id=subtask_id,
                        provider_name=provider_name,
                    )
                    messages.extend(extra_messages)
                    answer = text
                    break

                response = _stream_and_merge(
                    bound_model,
                    messages,
                    writer=writer,
                    channel="worker",
                    subtask_id=subtask_id,
                    provider_name=provider_name,
                )
                messages.append(response)
                tool_calls = list(getattr(response, "tool_calls", None) or [])

                if not tool_calls:
                    answer = _text(response)
                    break

                tool_messages, round_citations = _run_tool_calls(
                    tool_calls,
                    tool_registry=tool_registry,
                    workspace=workspace,
                    permissions=permissions,
                    subtask_id=subtask_id,
                    writer=writer,
                )
                messages.extend(tool_messages)
                citations.extend(round_citations)
            else:  # pragma: no cover - the at_cap branch above always breaks first
                answer = answer or "I was unable to complete this sub-task."

            result = SubtaskResult(subtask_id=subtask_id, question=question, answer=answer)
            return {"results": [result], "citations": citations}

    return worker


# ── synthesize (fan-out path only) ──────────────────────────────────────────


def make_synthesize_node(chat_model: BaseChatModel, *, provider_name: str) -> Node:
    """One LLM call merging every worker's sub-answer, plus a citation merge."""

    def synthesize(state: TurnState) -> dict[str, Any]:
        writer = get_stream_writer()
        writer({"type": "node_started", "node": "synthesize", "subtask_id": None})
        with _node_span("synthesize", thread_id=state.get("thread_id", "")):
            results = sorted(state.get("results") or [], key=lambda r: r.subtask_id)
            merged_citations = merge_citations(state.get("citations") or [])

            if not results:
                return {"answer": "No sub-task produced a result.", "citations": merged_citations}

            sub_results_text = "\n\n".join(
                f"[{r.subtask_id}] Sub-question: {r.question}\nAnswer: {r.answer}" for r in results
            )
            prompt = SYNTHESIZER_USER.format(
                question=state.get("question", ""), sub_results=sub_results_text
            )
            response = _stream_and_merge(
                chat_model,
                [HumanMessage(content=prompt)],
                writer=writer,
                channel="answer",
                subtask_id=None,
                provider_name=provider_name,
            )
            text = _text(response)
            # Append the final answer to the top-level transcript (guardrail already
            # appended this turn's HumanMessage) so multi-turn history and
            # Engine.history()/list_sessions() see a coherent conversation even
            # when this turn took the fan-out path.
            return {
                "answer": text,
                "citations": merged_citations,
                "messages": [AIMessage(content=text)],
            }

    return synthesize


# ── shared helpers ───────────────────────────────────────────────────────────


def _run_tool_calls(
    tool_calls: list[dict[str, Any]],
    *,
    tool_registry: ToolRegistry,
    workspace: Workspace,
    permissions: PermissionPolicy,
    subtask_id: str | None,
    writer: Callable[[Any], None],
) -> tuple[list[ToolMessage], list[Citation]]:
    """Execute every requested tool call, gating side-effecting ones through
    ``permissions`` and turning any failure into a ``ToolMessage`` instead of
    an exception -- a tool failure is never fatal to the turn.

    Per-call INFO/WARNING logging (name, duration, row count / error) happens
    inside ``ToolRegistry.run()`` itself, not here -- that keeps it in one
    place for both this function's two call sites (``tools_node`` and
    ``worker``) and inherits whichever ``thread_id``/``node``/``subtask_id``
    the caller's ``_node_span`` already bound, since both call sites invoke
    ``run()`` directly on their own thread rather than dispatching it further.
    """
    messages: list[ToolMessage] = []
    citations: list[Citation] = []

    for call in tool_calls:
        call_id = str(call.get("id") or f"{call.get('name', 'tool')}-{len(messages)}")
        name = str(call.get("name", ""))
        args = dict(call.get("args") or {})
        writer(
            {
                "type": "tool_started",
                "call_id": call_id,
                "name": name,
                "args": args,
                "subtask_id": subtask_id,
            }
        )

        try:
            spec = tool_registry.get(name)
        except CellSenseError as exc:
            messages.append(_tool_error_message(call_id, name, exc, 0.0, subtask_id, writer))
            continue

        if spec.side_effect:
            decision = permissions.decide(spec, args)
            if decision == "ask":
                reason = f"{name!r} has side effects and is not on the allow-list."
                # interrupt() sits directly before the one call that performs the
                # side effect -- see the module docstring for why that ordering
                # is the whole safety property this function provides. Engine
                # is the sole source of ApprovalRequested/ApprovalDecision (it
                # reconstructs both straight from this payload and its
                # on_approval callback's outcome, and logs the request/outcome
                # there), so this function does not also emit a custom event or
                # a log record for them.
                outcome = interrupt(
                    {"call_id": call_id, "tool": name, "args": args, "reason": reason}
                )
                if outcome == "allow_always":
                    permissions.remember(name)
                if outcome == "deny":
                    messages.append(
                        ToolMessage(
                            content=f"The user declined to run {name!r}.",
                            tool_call_id=call_id,
                            id=f"{call_id}-deny",
                        )
                    )
                    continue
            elif decision == "deny":
                messages.append(
                    ToolMessage(
                        content=f"{name!r} is not permitted by the current permission policy.",
                        tool_call_id=call_id,
                        id=f"{call_id}-policy-deny",
                    )
                )
                continue

        start = time.monotonic()
        try:
            result = tool_registry.run(name, args, workspace)
        except CellSenseError as exc:
            duration = time.monotonic() - start
            messages.append(_tool_error_message(call_id, name, exc, duration, subtask_id, writer))
            continue
        except Exception as exc:  # a tool failure is never fatal to the turn
            duration = time.monotonic() - start
            messages.append(_tool_error_message(call_id, name, exc, duration, subtask_id, writer))
            continue

        # NOTE: `find_files` can call Workspace.add(), bringing a brand-new file
        # into scope mid-turn. We deliberately do not re-render `context` (the
        # prompt-time schema digest) after that: the model learns the new
        # file's name from this very ToolMessage's text, and ToolSpec.handler
        # calls (describe/aggregate/filter_rows/join/plot) resolve filenames
        # against the live `workspace` object below, not against the frozen
        # `context` string -- so a follow-up call in the same turn against a
        # just-discovered file still works correctly even though `context`
        # itself goes stale until the next turn.
        duration = time.monotonic() - start
        citations.extend(result.citations)
        writer(
            {
                "type": "tool_finished",
                "call_id": call_id,
                "name": name,
                "summary": result.summary,
                "row_count": len(result.data),
                "duration_s": duration,
                "preview": result.preview(),
                "subtask_id": subtask_id,
            }
        )
        messages.append(
            ToolMessage(content=result.to_text(), tool_call_id=call_id, id=f"{call_id}-result")
        )

    return messages, citations


def _tool_error_message(
    call_id: str,
    name: str,
    exc: Exception,
    duration: float,
    subtask_id: str | None,
    writer: Callable[[Any], None],
) -> ToolMessage:
    detail = str(exc)
    hint = getattr(exc, "hint", None)
    if hint:
        detail = f"{detail} Hint: {hint}"
    writer(
        {
            "type": "tool_failed",
            "call_id": call_id,
            "name": name,
            "error": detail,
            "duration_s": duration,
            "subtask_id": subtask_id,
        }
    )
    return ToolMessage(content=f"Tool error: {detail}", tool_call_id=call_id, id=f"{call_id}-error")


# ── final-answer forcing at the round cap (BUG 1) ───────────────────────────

_FORCE_FINAL_INSTRUCTION = (
    "You have reached the maximum number of tool calls allowed for this turn. "
    "Do not call any more tools. Using only the information already gathered "
    "above, answer the user's question now in prose, and end with the "
    "citation block."
)


def _force_final_answer(
    chat_model: BaseChatModel,
    messages: list[BaseMessage],
    *,
    writer: Callable[[Any], None],
    channel: str,
    subtask_id: str | None,
    provider_name: str,
) -> tuple[str, list[BaseMessage]]:
    """Ask the model to answer in prose with no tools bound, once the round
    cap is hit.

    Some providers (observed live: Groq's ``openai/gpt-oss-120b``) reject a
    tools-unbound generation outright -- ``"Tool choice is none, but model
    called a tool"`` -- if the transcript still ends on an unresolved tool
    call/result, rather than simply answering in prose. Appending an explicit
    "stop calling tools, answer now" instruction first resolves that in
    practice. If the provider still rejects the generation, this falls back
    to the raw tool results already gathered instead of failing the turn --
    a round cap must degrade to a worse answer, never a crash. Either fallback
    trigger (the call itself failing, or it succeeding with empty text) is
    logged at WARNING -- this is the turn visibly degrading, not routine
    tracing.

    Returns ``(answer_text, new_messages)`` where ``new_messages`` is what the
    caller should append to its transcript (the instruction plus whatever
    response -- real or synthesized -- produced ``answer_text``).
    """
    instruction = HumanMessage(content=_FORCE_FINAL_INSTRUCTION)
    try:
        response = _stream_and_merge(
            chat_model,
            [*messages, instruction],
            writer=writer,
            channel=channel,
            subtask_id=subtask_id,
            provider_name=provider_name,
        )
        text = _text(response)
        if text.strip():
            return text, [instruction, response]
    except CellSenseError as exc:
        logger.warning(
            "forced final-answer call failed; falling back to raw tool results",
            extra={"error": str(exc)},
        )
        writer(
            {
                "type": "notice",
                "level": "warn",
                "message": (
                    f"Final answer call also failed ({exc}); falling back to raw tool results."
                ),
            }
        )
    else:
        logger.warning(
            "forced final-answer call returned no text; falling back to raw tool results"
        )

    tool_texts = [str(m.content) for m in messages if isinstance(m, ToolMessage) and m.content]
    if tool_texts:
        fallback = (
            "I reached the tool-call limit before finishing this turn. Here is "
            "what was found so far:\n\n" + "\n\n".join(tool_texts)
        )
    else:
        fallback = "I was unable to complete this request within the allowed steps."
    return fallback, [instruction, AIMessage(content=fallback)]


# ── model calls: error translation + rate-limit retry (BUG 2) ──────────────


def _extract_retry_after(message: str) -> float | None:
    match = _RETRY_AFTER_RE.search(message)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _wait_before_retry(
    writer: Callable[[Any], None], exc: Exception, provider_name: str, attempt: int
) -> None:
    wait_s = min(
        _extract_retry_after(str(exc)) or (_DEFAULT_RETRY_WAIT_S * (attempt + 1)),
        _MAX_RETRY_WAIT_S,
    )
    logger.warning(
        "rate-limited; retrying after backoff",
        extra={
            "provider": provider_name,
            "wait_s": round(wait_s, 2),
            "attempt": attempt + 1,
            "max_retries": _MAX_RETRIES,
        },
    )
    writer(
        {
            "type": "notice",
            "level": "warn",
            "message": (
                f"{provider_name} rate-limited this call; waiting {wait_s:.0f}s before "
                f"retrying (attempt {attempt + 1}/{_MAX_RETRIES})."
            ),
        }
    )
    time.sleep(wait_s)


def _invoke_with_retry(
    model: ChatModelLike,
    messages: list[BaseMessage],
    *,
    writer: Callable[[Any], None],
    provider_name: str,
) -> BaseMessage:
    """Plain (non-streaming) model call with translated errors and bounded,
    backoff-respecting retry for rate limits.

    Used by ``guardrail``/``planner`` -- small, tool-free calls where a
    single ``invoke()`` is enough. ``_stream_and_merge`` has its own copy of
    this retry loop because it also has to redo the streaming/chunk-merge
    dance on each attempt.
    """
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return model.invoke(messages)
        except Exception as exc:
            translated = translate_provider_error(exc, provider_name)
            if not isinstance(translated, RateLimitError) or attempt >= _MAX_RETRIES:
                raise translated from exc
            _wait_before_retry(writer, exc, provider_name, attempt)
    raise CellSenseError(f"{provider_name}: exhausted retries.")  # pragma: no cover


def _stream_and_merge(
    model: ChatModelLike,
    messages: list[BaseMessage],
    *,
    writer: Callable[[Any], None],
    channel: str,
    subtask_id: str | None,
    provider_name: str,
) -> BaseMessage:
    """Stream a model response, emitting a ``text_delta`` per chunk, merge the
    chunks into one final message, and translate/retry provider errors.

    Falls back to a plain ``invoke()`` if the model/provider combination
    doesn't support streaming. Rate-limit errors (translated via
    ``providers.base.translate_provider_error``) are retried with backoff,
    honouring the provider's own "try again in Ns" hint when present, up to
    ``_MAX_RETRIES`` times; any other translated error (bad request, auth,
    ...) is raised immediately -- retrying a request the provider has already
    rejected outright would just waste the same error three more times.
    """
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = _one_stream_attempt(model, messages, writer, channel, subtask_id)
        except Exception as exc:
            translated = translate_provider_error(exc, provider_name)
            if not isinstance(translated, RateLimitError) or attempt >= _MAX_RETRIES:
                raise translated from exc
            _wait_before_retry(writer, exc, provider_name, attempt)
            continue
        _emit_usage(writer, response)
        return response
    raise CellSenseError(f"{provider_name}: exhausted retries.")  # pragma: no cover


def _one_stream_attempt(
    model: ChatModelLike,
    messages: list[BaseMessage],
    writer: Callable[[Any], None],
    channel: str,
    subtask_id: str | None,
) -> BaseMessage:
    """One streaming attempt -- separated out so ``_stream_and_merge`` can
    retry it wholesale (chunks already emitted as ``TextDelta`` on a failed
    attempt are just UI noise; the retried attempt re-streams from scratch).
    """
    try:
        # `ChatModelLike` is typed generically as `Runnable[LanguageModelInput,
        # BaseMessage]` (the common supertype of a raw chat model and a
        # `.bind_tools(...)`-bound one -- see the alias's docstring), so mypy
        # sees `.stream()` yielding plain `BaseMessage`. At runtime a chat
        # model's `.stream()` always yields `BaseMessageChunk` instances --
        # the only reason `merged + chunk` below merges chunks at all, rather
        # than resolving to `BaseMessage.__add__`'s unrelated "build a
        # ChatPromptTemplate" overload.
        chunks: list[BaseMessageChunk] = []
        for raw_chunk in model.stream(messages):
            chunk = cast(BaseMessageChunk, raw_chunk)
            text = _text(chunk)
            if text:
                writer(
                    {
                        "type": "text_delta",
                        "text": text,
                        "channel": channel,
                        "subtask_id": subtask_id,
                    }
                )
            chunks.append(chunk)
    except NotImplementedError:
        return model.invoke(messages)
    if not chunks:
        return model.invoke(messages)
    merged = chunks[0]
    for chunk in chunks[1:]:
        merged = merged + chunk
    return merged


def _emit_usage(writer: Callable[[Any], None], response: BaseMessage) -> None:
    """Extract token usage off one model response and, if any is present,
    emit it as a custom event -- ``engine.py`` is the only thing that turns
    this into a :class:`~cellsense.events.UsageUpdated` event and feeds it to
    a :class:`~cellsense.observability.usage.UsageTracker`. Centralizing this
    here (rather than in ``engine.py`` alone) is what lets usage from
    ``worker`` branches -- whose messages never touch shared ``TurnState``,
    by design -- still reach the tracker.
    """
    input_tokens, output_tokens = extract_usage(response)
    if input_tokens or output_tokens:
        writer({"type": "usage", "input_tokens": input_tokens, "output_tokens": output_tokens})


def _last_message(state: TurnState) -> BaseMessage | None:
    """The most recent message on the top-level branch, or ``None`` if the
    transcript is (unexpectedly) still empty.

    Written this way -- rather than ``(state.get("messages") or [None])[-1]``
    -- because that pattern, while never actually wrong at runtime (only ever
    used through ``getattr(last, ..., default)``, which handles ``None``
    safely), makes mypy infer the literal ``[None]`` as ``list[BaseMessage]``
    and report a spurious "incompatible type None" on the list item. In
    practice ``guardrail`` always seeds at least a ``HumanMessage`` before
    ``agent``/``tools`` ever run, so ``messages`` is never actually empty here;
    this stays defensive anyway rather than assuming a graph invariant holds.
    """
    messages = state.get("messages")
    return messages[-1] if messages else None


def _text(message: BaseMessage) -> str:
    """Extract plain text from a message whose ``content`` may be a string or
    a list of provider-specific content blocks (Anthropic-style dict blocks).
    """
    content = message.content
    if isinstance(content, str):
        return content
    # BaseMessage.content's declared type is exactly str | list[str | dict[...]],
    # so the branch above and the one below are exhaustive -- mypy proves it (a
    # trailing fallback here is genuinely unreachable, not merely undetected).
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)
