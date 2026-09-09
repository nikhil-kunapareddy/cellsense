"""Wires the six nodes in ``graph.nodes`` into one compiled LangGraph state
machine.

::

    START -> guardrail --off-topic--> END
               |
               v (relevant)
            planner
               |
       +-------+--------------------------+
       | 1 sub-task                        | N sub-tasks (Send fan-out)
       v                                   v
     agent <----------+                 worker (x N)
       |               |                   |
       v (tool_calls)  |                   v
     tools ------------+              synthesize
       |                                   |
       v (no tool_calls)                   v
      END <--------------------------------+

The single-subtask path never visits ``synthesize`` -- ARCHITECTURE.md §6
calls this out explicitly as a latency optimization, and it falls out
naturally here from ``route_after_planner`` returning the literal node name
``"agent"`` instead of a list of ``Send`` objects.

``build_graph`` takes fully-constructed dependencies (a bound chat model, a
live ``ToolRegistry``/``Workspace``/``PermissionPolicy``) rather than a
``cellsense.config.Config`` object, so this module -- and the tests that
exercise it with a stub chat model -- never needs to import ``cellsense.config``
or ``cellsense.providers`` at all. ``graph.engine.Engine`` is the one place
that unpacks a ``Config`` into these primitives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from cellsense.graph.nodes import (
    make_agent_node,
    make_guardrail_node,
    make_planner_node,
    make_route_after_agent,
    make_synthesize_node,
    make_tools_node,
    make_worker_node,
    route_after_guardrail,
    route_after_planner,
)
from cellsense.graph.state import TurnState
from cellsense.io.schema import Workspace
from cellsense.permissions import PermissionPolicy
from cellsense.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

__all__ = ["build_graph"]


def build_graph(
    *,
    chat_model: BaseChatModel,
    tool_registry: ToolRegistry,
    workspace: Workspace,
    permissions: PermissionPolicy,
    provider_name: str = "unknown",
    checkpointer: BaseCheckpointSaver | None = None,
    max_tool_rounds: int = 8,
    max_subtasks: int = 4,
    guardrail_enabled: bool = True,
    planner_enabled: bool = True,
) -> CompiledStateGraph:
    """Build and compile the turn graph.

    All node dependencies are passed in explicitly (rather than resolved from
    a ``Config`` here) so this function -- and therefore the whole graph --
    can be exercised in tests against a stub ``BaseChatModel`` with no real
    provider, API key, or network access. ``provider_name`` is used purely
    for error labelling/translation (``providers.base.translate_provider_error``)
    inside the nodes -- it defaults to ``"unknown"`` so graph-only tests that
    don't care about provider-specific error text don't have to supply it.
    """
    graph = StateGraph(TurnState)

    # mypy cannot match any of `add_node`'s overloads here, and it is a stub
    # limitation rather than a real type error: a minimal repro (a bare TypedDict
    # state + a plain `def node(state: S) -> dict[str, Any]` passed directly)
    # type-checks cleanly, but as soon as the callable is typed through a
    # `Callable[[S], dict[str, Any]]` alias -- exactly what `nodes.Node` is, and
    # what every `make_x_node(...) -> Node` factory here returns -- overload
    # resolution against `_Node[NodeInputT] | ... | Runnable[NodeInputT, Any]`
    # fails even though the value is structurally identical to one that passes.
    # `[call-overload]` (never bare) on each call below, not a broader ignore.
    graph.add_node(
        "guardrail",
        make_guardrail_node(chat_model, enabled=guardrail_enabled, provider_name=provider_name),
    )  # type: ignore[call-overload]
    graph.add_node(
        "planner",
        make_planner_node(
            chat_model,
            enabled=planner_enabled,
            max_subtasks=max_subtasks,
            provider_name=provider_name,
        ),
    )  # type: ignore[call-overload]
    graph.add_node(
        "agent",
        make_agent_node(
            chat_model, tool_registry, max_tool_rounds=max_tool_rounds, provider_name=provider_name
        ),
    )  # type: ignore[call-overload]
    graph.add_node(  # type: ignore[call-overload]
        "tools", make_tools_node(tool_registry, workspace, permissions)
    )
    graph.add_node(
        "worker",
        make_worker_node(
            chat_model,
            tool_registry,
            workspace,
            permissions,
            max_tool_rounds=max_tool_rounds,
            provider_name=provider_name,
        ),
    )  # type: ignore[call-overload]
    graph.add_node(  # type: ignore[call-overload]
        "synthesize", make_synthesize_node(chat_model, provider_name=provider_name)
    )

    graph.add_edge(START, "guardrail")
    graph.add_conditional_edges("guardrail", route_after_guardrail, ["planner", END])
    graph.add_conditional_edges("planner", route_after_planner, ["agent", "worker"])
    graph.add_conditional_edges("agent", make_route_after_agent(max_tool_rounds), ["tools", END])
    graph.add_edge("tools", "agent")
    graph.add_edge("worker", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile(checkpointer=checkpointer)
