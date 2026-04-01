"""Claude agent using the Anthropic SDK with native tool calling."""
from __future__ import annotations

import os
from typing import Dict, List

import anthropic

from src.data import FileData
from src.tools import ALL_SCHEMAS, TOOL_HANDLERS, ToolResult
from src.utils.citations import Citation, merge_citations
from src.agents.base import orchestrate

MODEL = "claude-sonnet-4-5"
AGENT_LABEL = f"{MODEL} · Anthropic"
MAX_TOOL_ROUNDS = 8

_SYSTEM_PROMPT = """\
You are CellSense, an expert data analyst assistant embedded in a CLI tool.
A summary of any loaded Excel/CSV files is provided below. If no files are shown, \
use the list_directory or find_files tools to discover and load files before answering. \
Answer questions by calling the available tools to filter, aggregate, join, or find data.

Rules:
- Always use tool calls to retrieve actual data before answering — do not hallucinate numbers.
- After receiving tool results, synthesise a clear, concise answer.
- Every answer MUST end with a citation block in exactly this format:
  Sources: <filename> [Sheet: <sheet>, Rows: <i, j, ...>] | <filename2> [Rows: ...]
  (For CSV files omit the Sheet field.)
- If a question cannot be answered with the available data, say so clearly.

File context:
{context}
"""


def ask_question(
    question: str,
    file_data: Dict[str, FileData],
    context: str,
    history: List[dict],
) -> tuple[str, List[Citation]]:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def llm_call(messages: List[dict]) -> str:
        system = None
        filtered = []
        for m in messages:
            if m.get("role") == "system":
                system = m["content"]
            else:
                filtered.append(m)
        kwargs: dict = {"model": MODEL, "max_tokens": 1024, "messages": filtered}
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        return response.content[0].text

    def worker_fn(
        q: str, fd: Dict[str, FileData], ctx: str, hist: List[dict]
    ) -> tuple[str, List[Citation]]:
        return _tool_loop(client, q, fd, ctx, hist)

    return orchestrate(
        question, file_data, context, history, llm_call=llm_call, worker_fn=worker_fn
    )


def _tool_loop(
    client: anthropic.Anthropic,
    question: str,
    file_data: Dict[str, FileData],
    context: str,
    history: List[dict],
) -> tuple[str, List[Citation]]:
    system = _SYSTEM_PROMPT.format(context=context)
    history.append({"role": "user", "content": question})
    all_citations: List[Citation] = []

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=system,
            tools=ALL_SCHEMAS,
            messages=history,
        )

        tool_calls = []
        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(block)

        history.append({"role": "assistant", "content": response.content})

        if not tool_calls:
            return "\n".join(text_parts), merge_citations(all_citations)

        tool_results = []
        for tc in tool_calls:
            result = _execute_tool(tc.name, tc.input, file_data)
            if isinstance(result, ToolResult):
                all_citations.extend(result.citations)
                text = result.to_text()
            else:
                text = str(result)
            tool_results.append({"type": "tool_result", "tool_use_id": tc.id, "content": text})

        history.append({"role": "user", "content": tool_results})

    return (
        "I was unable to complete the analysis within the allowed steps.",
        merge_citations(all_citations),
    )


def _execute_tool(name: str, inputs: dict, file_data: Dict[str, FileData]) -> ToolResult | str:
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return f"Error: unknown tool {name!r}"
    try:
        return handler(inputs, file_data)
    except Exception as exc:
        return f"Tool error ({name}): {exc}"
