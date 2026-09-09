"""Tool discovery, schema export for every provider shape, and the one real
execution path.

Each tool module exposes a module-level ``SPEC: ToolSpec`` (see
``tools/base.py``). ``ToolRegistry`` collects the modules listed in
``_MODULES``, and exposes three read-only views of them plus one verb:

* ``specs()``            -- the raw ``ToolSpec`` list
* ``openai_schemas()``   -- OpenAI function-calling format (Groq/Gemini/Llama)
* ``anthropic_schemas()``-- Anthropic native tool-use format
* ``langchain_tools()``  -- ``StructuredTool`` objects for ``model.bind_tools()``
* ``run(name, args, workspace)`` -- **the** execution path

``langchain_tools()`` deliberately does not wire the handler into the
``StructuredTool``'s callable. LangChain's own tool-calling loop would run
the handler directly and hand back a plain string, which throws away the
``ToolResult`` (its DataFrame, its citations) the graph needs to attach
provenance to the final answer. So the LangChain tool objects exist only to
give ``model.bind_tools()`` something schema-shaped to bind; the graph reads
the model's tool call, then calls ``ToolRegistry.run()`` itself.

To add a tool: write a module with ``SPEC = tool_spec(...)(handler)``, then
add it to ``_MODULES``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, create_model

from cellsense.errors import ToolError
from cellsense.io.schema import ToolResult, Workspace
from cellsense.observability.logging import LOGGER_NAME
from cellsense.tools import (
    aggregate,
    describe,
    filter_rows,
    find_files,
    join,
    list_directory,
    plot,
)
from cellsense.tools.base import ToolSpec

_MODULES = [aggregate, describe, filter_rows, join, plot, list_directory, find_files]

__all__ = ["ToolRegistry"]

logger = logging.getLogger(LOGGER_NAME)


class ToolRegistry:
    """The set of tools available to the agent for one run.

    Takes an explicit ``specs`` list (rather than always using every built-in
    tool) so tests, and callers that want a restricted tool set for a given
    permission mode, can construct a registry without monkeypatching module
    globals.
    """

    def __init__(self, specs: list[ToolSpec] | None = None) -> None:
        self._specs: list[ToolSpec] = specs if specs is not None else [m.SPEC for m in _MODULES]
        names = [s.name for s in self._specs]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"Duplicate tool name(s) in registry: {', '.join(dupes)}")
        self._by_name: dict[str, ToolSpec] = {s.name: s for s in self._specs}

    def specs(self) -> list[ToolSpec]:
        return list(self._specs)

    def get(self, name: str) -> ToolSpec:
        try:
            return self._by_name[name]
        except KeyError:
            raise ToolError(
                f"Unknown tool {name!r}.",
                hint=f"Available tools: {', '.join(self._by_name)}",
            ) from None

    def run(self, name: str, args: dict[str, Any], workspace: Workspace) -> ToolResult:
        """Validate ``args`` against the tool's schema, coerce obvious LLM
        type slips, then call its handler. The only path that actually
        executes a tool -- LangChain's own calling convention is bypassed
        entirely (see module docstring).

        Every call is logged here -- the one choke point both the top-level
        ``tools`` node and every fan-out ``worker`` invocation run through --
        at INFO with the tool name, duration, and result row count on
        success, or WARNING with the error on failure. Argument *values* can
        carry user data, so they only ever appear at DEBUG, and even there as
        a ``repr()`` string (so the formatter's length-truncation applies to
        it as a single field) -- never at INFO, and the result's DataFrame
        itself is never logged at any level.
        """
        spec = self.get(name)
        start = time.monotonic()
        try:
            coerced = spec.validate_args(args)
            logger.debug("tool args", extra={"tool": name, "tool_args": repr(coerced)})
            result = spec.handler(coerced, workspace)
        except Exception as exc:
            duration_s = time.monotonic() - start
            logger.warning(
                "tool failed: %s",
                name,
                extra={"duration_s": round(duration_s, 4), "error": str(exc)},
            )
            raise
        duration_s = time.monotonic() - start
        logger.info(
            "tool ran: %s",
            name,
            extra={"duration_s": round(duration_s, 4), "row_count": len(result.data)},
        )
        return result

    def openai_schemas(self) -> list[dict[str, Any]]:
        """OpenAI function-calling format, used by the Groq/Gemini/Llama
        OpenAI-compatible backends.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": s.name,
                    "description": s.description,
                    "parameters": s.parameters,
                },
            }
            for s in self._specs
        ]

    def anthropic_schemas(self) -> list[dict[str, Any]]:
        """Anthropic native tool-use format."""
        return [
            {"name": s.name, "description": s.description, "input_schema": s.parameters}
            for s in self._specs
        ]

    def langchain_tools(self) -> list[StructuredTool]:
        """``StructuredTool`` objects shaped for ``model.bind_tools()``.
        Never actually invoked -- see the module docstring.
        """
        return [_to_structured_tool(s) for s in self._specs]


# ── LangChain export ─────────────────────────────────────────────────────────

_JSON_TYPE_MAP: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict[str, Any],
}


def _json_type_to_python(schema: dict[str, Any]) -> Any:
    json_type = schema.get("type")
    if json_type == "array":
        item_schema = schema.get("items") or {}
        if not item_schema:
            return list[Any]
        # Dynamic, pydantic-model-only use: item_type is a runtime type object,
        # not a static type expression, which mypy can't verify.
        item_type = _json_type_to_python(item_schema)
        return list[item_type]  # type: ignore[valid-type]
    if isinstance(json_type, str):
        return _JSON_TYPE_MAP.get(json_type, Any)
    return Any


def _args_model(tool_name: str, parameters: dict[str, Any]) -> type[BaseModel]:
    """Build a pydantic v2 model from a tool's JSON Schema ``parameters`` so
    LangChain can generate its own tool-call schema off it. Every field is
    made ``Optional`` at the pydantic level and required-ness is enforced
    separately by ``ToolSpec.validate_args`` -- duplicating "required" here
    would just be a second, easier-to-drift source of truth.
    """
    properties: dict[str, Any] = parameters.get("properties", {})
    required = set(parameters.get("required", []))
    fields: dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        py_type = _json_type_to_python(prop_schema)
        description = prop_schema.get("description", "")
        if prop_name in required:
            fields[prop_name] = (py_type, Field(..., description=description))
        else:
            fields[prop_name] = (py_type | None, Field(default=None, description=description))
    model_name = f"{tool_name.title().replace('_', '')}Args"
    return create_model(model_name, __config__=ConfigDict(extra="allow"), **fields)


def _to_structured_tool(spec: ToolSpec) -> StructuredTool:
    args_model = _args_model(spec.name, spec.parameters)

    def _not_executed(**_kwargs: Any) -> str:
        # Never actually called: CellSense's graph executes tools itself via
        # ToolRegistry.run() so it can attach citations before turning the
        # result into a ToolMessage. This closure exists only because
        # StructuredTool.from_function requires *some* callable to build the
        # schema-bound tool object model.bind_tools() expects.
        raise ToolError(
            f"{spec.name} was invoked through the LangChain tool-calling path, "
            "which CellSense never executes directly -- this indicates a wiring bug."
        )

    return StructuredTool.from_function(
        func=_not_executed,
        name=spec.name,
        description=spec.description,
        args_schema=args_model,
    )
