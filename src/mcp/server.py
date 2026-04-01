"""Standalone MCP server exposing CellSense filesystem tools.

Runs over stdio so any MCP-compatible client (Claude Desktop, etc.) can
connect to it and use the list_directory / find_files tools.

Usage:
    python -m src.mcp.server

Add to Claude Desktop's MCP config (claude_desktop_config.json):
    {
      "mcpServers": {
        "cellsense-fs": {
          "command": "python",
          "args": ["-m", "src.mcp.server"],
          "cwd": "/path/to/cellsense"
        }
      }
    }
"""
from __future__ import annotations

import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from src.mcp import (
    FIND_FILES_SCHEMA,
    LIST_DIRECTORY_SCHEMA,
    handle_find_files,
    handle_list_directory,
)

app = Server("cellsense-filesystem")

_TOOL_MAP = {
    "list_directory": handle_list_directory,
    "find_files": handle_find_files,
}


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name=s["name"],
            description=s["description"],
            inputSchema=s["input_schema"],
        )
        for s in [LIST_DIRECTORY_SCHEMA, FIND_FILES_SCHEMA]
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    handler = _TOOL_MAP.get(name)
    if handler is None:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    # MCP server has no live file_data — pass empty dict (discovery only)
    file_data: dict = {}
    try:
        result = handler(arguments, file_data)
        return [TextContent(type="text", text=result.to_text())]
    except Exception as exc:
        return [TextContent(type="text", text=f"Error: {exc}")]


async def _main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(_main())
