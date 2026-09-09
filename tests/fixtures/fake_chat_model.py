"""A scripted ``BaseChatModel`` for graph/engine integration tests -- no
network, and no real ``langchain_*`` provider package required.

Every node factory in ``cellsense.graph.build.build_graph`` closes over the
*same* ``chat_model`` instance for the lifetime of one compiled graph
(guardrail, planner, agent, worker, and synthesize all share it -- see
``build.py``'s docstring), so :class:`ScriptedChatModel` dispatches on the
*shape* of the incoming message list (system-prompt wording, whether a
``ToolMessage`` is already present, ...) rather than an ordered response
queue. That is both robust to whichever node happens to call it next and
inherently thread-safe for concurrent fan-out ``worker`` calls, since a
script function never mutates shared state to decide what to say.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field

__all__ = [
    "Script",
    "ScriptedChatModel",
    "ai_text",
    "ai_tool_call",
    "has_tool_message",
    "is_forced_final_call",
    "is_guardrail_call",
    "is_planner_call",
    "is_synthesize_call",
]

Script = Callable[[list[BaseMessage]], BaseMessage]


class ScriptedChatModel(BaseChatModel):
    """A fake ``BaseChatModel`` driven by a plain Python callable.

    ``bind_tools()`` returns ``self`` unchanged -- CellSense's graph never
    actually invokes a call through a LangChain-bound tool object
    (``tools/registry.py``'s module docstring: the ``StructuredTool`` export
    exists only to give ``model.bind_tools()`` something schema-shaped to
    bind), it only needs *something* to bind so tool-call-shaped
    ``AIMessage``s round-trip; the script itself decides when to emit one.

    ``.stream()`` is inherited unmodified from ``BaseChatModel``: since this
    class does not override ``_stream``, ``BaseChatModel._should_stream``
    reports "no native streaming support" and ``.stream()`` transparently
    falls back to one ``.invoke()`` call yielding a single chunk -- exactly
    what ``cellsense.graph.nodes._one_stream_attempt`` expects either way.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    script: Script
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    def bind_tools(
        self, tools: Any, *, tool_choice: str | None = None, **kwargs: Any
    ) -> ScriptedChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        response = self.script(list(messages))
        return ChatResult(generations=[ChatGeneration(message=response)])

    @property
    def _llm_type(self) -> str:
        return "scripted-chat-model"


# ── response builders ────────────────────────────────────────────────────────


def ai_text(text: str) -> AIMessage:
    """A final, tool-free answer."""
    return AIMessage(content=text)


def ai_tool_call(name: str, args: dict[str, Any], call_id: str = "call1") -> AIMessage:
    """A single tool-call request."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


# ── call-shape detection (so one script can serve every node) ───────────────


def is_guardrail_call(messages: list[BaseMessage]) -> bool:
    return (
        bool(messages)
        and isinstance(messages[0], SystemMessage)
        and "relevance classifier" in str(messages[0].content)
    )


def is_planner_call(messages: list[BaseMessage]) -> bool:
    return (
        bool(messages)
        and isinstance(messages[0], SystemMessage)
        and "query decomposition planner" in str(messages[0].content)
    )


def is_synthesize_call(messages: list[BaseMessage]) -> bool:
    return (
        bool(messages)
        and isinstance(messages[0], HumanMessage)
        and "Independent sub-analyses were run in parallel" in str(messages[0].content)
    )


def is_forced_final_call(messages: list[BaseMessage]) -> bool:
    """True once ``nodes._force_final_answer`` has appended its
    "stop calling tools" instruction as the last message.
    """
    return (
        bool(messages)
        and isinstance(messages[-1], HumanMessage)
        and "maximum number of tool calls" in str(messages[-1].content)
    )


def has_tool_message(messages: list[BaseMessage]) -> bool:
    return any(isinstance(m, ToolMessage) for m in messages)
