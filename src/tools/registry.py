"""Tool registry for CellSense.

Each tool lives in its own module exposing a uniform contract:

    SCHEMA : dict                              # Anthropic-style tool schema
    handle(inputs, file_data) -> ToolResult    # the executor

This module collects them into the three names agents import:

    ALL_SCHEMAS      — Anthropic tool schemas
    ALL_OAI_SCHEMAS  — the same schemas in OpenAI function-calling format
    TOOL_HANDLERS    — {tool name: handler}, keyed off each SCHEMA["name"]

To add a tool: drop a new module in src/tools/ with SCHEMA + handle, then
append it to _MODULES below.
"""
from __future__ import annotations

from typing import Any, Dict

from src.types import ToolResult
from src.tools import (
    filter_rows,
    aggregate,
    join_files,
    plot,
    list_directory,
    find_files,
)

_MODULES = [filter_rows, aggregate, join_files, plot, list_directory, find_files]


def _to_oai(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an Anthropic-style tool schema to OpenAI function-calling format."""
    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema["description"],
            "parameters": schema["input_schema"],
        },
    }


ALL_SCHEMAS = [m.SCHEMA for m in _MODULES]
ALL_OAI_SCHEMAS = [_to_oai(m.SCHEMA) for m in _MODULES]
TOOL_HANDLERS = {m.SCHEMA["name"]: m.handle for m in _MODULES}

__all__ = ["ToolResult", "ALL_SCHEMAS", "ALL_OAI_SCHEMAS", "TOOL_HANDLERS"]
