"""Llama agent using Meta's Llama API (api.llama.com).

Meta's API is NOT OpenAI-compatible in its response format. Responses use
`completion_message` instead of `choices`. This module handles that format
directly while still using the openai SDK for the request layer.

Required env var: LLAMA_API_KEY
Optional env vars:
  LLAMA_MODEL    — model ID (default: Llama-4-Maverick-17B-128E-Instruct-FP8)
  LLAMA_BASE_URL — API base URL (default: https://api.llama.com/v1)
"""
from __future__ import annotations

import json
import os
from typing import Any

import openai

from src.data import FileData
from src.tools import ALL_OAI_SCHEMAS, TOOL_HANDLERS, ToolResult
from src.utils.citations import Citation, merge_citations
from src.agents.base import orchestrate

BASE_URL = os.environ.get("LLAMA_BASE_URL", "https://api.llama.com/v1")
MODEL = os.environ.get("LLAMA_MODEL", "Llama-3.3-70B-Instruct")
AGENT_LABEL = f"{MODEL} · Meta Llama API"

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


def _completion_message_text(content: object) -> str:
    """Meta returns final content as ``{"type": "text", "text": "..."}`` or a string."""
    if isinstance(content, dict):
        return str(content.get("text", ""))
    return str(content) if content else ""


def ask_question(
    question: str,
    file_data: dict[str, FileData],
    context: str,
    history: list[dict],
) -> tuple[str, list[Citation]]:
    client = openai.OpenAI(
        api_key=os.environ["LLAMA_API_KEY"],
        base_url=BASE_URL,
    )

    def llm_call(messages: list[dict]) -> str:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=1024,
        )
        data = response.model_dump()
        # Meta uses completion_message; fall back to standard OpenAI choices format
        if "completion_message" in data:
            return _completion_message_text(data["completion_message"].get("content", {}))
        choices = data.get("choices") or []
        if choices:
            return choices[0].get("message", {}).get("content") or ""
        return ""

    def worker_fn(
        q: str, fd: dict[str, FileData], ctx: str, hist: list[dict]
    ) -> tuple[str, list[Citation]]:
        return _tool_loop(client, q, fd, ctx, hist)

    return orchestrate(
        question, file_data, context, history, llm_call=llm_call, worker_fn=worker_fn
    )


def _normalize_args(arguments_str: str) -> str:
    """Re-serialize tool arguments, coercing string 'true'/'false' to actual booleans.

    Llama sometimes emits boolean fields as strings (e.g. "recursive": "true").
    The API validates these against the tool schema on the next round-trip and
    rejects them with a 400 if the type doesn't match.
    """
    try:
        args = json.loads(arguments_str)
    except json.JSONDecodeError:
        return arguments_str

    coerced = {}
    for k, v in args.items():
        if isinstance(v, str) and v.lower() == "true":
            coerced[k] = True
        elif isinstance(v, str) and v.lower() == "false":
            coerced[k] = False
        else:
            coerced[k] = v
    return json.dumps(coerced)


def _tool_loop(
    client: openai.OpenAI,
    question: str,
    file_data: dict[str, FileData],
    context: str,
    history: list[dict],
) -> tuple[str, list[Citation]]:
    system_content = _SYSTEM_PROMPT.format(context=context)
    if history and history[0].get("role") == "system":
        history[0]["content"] = system_content
    else:
        history.insert(0, {"role": "system", "content": system_content})

    history.append({"role": "user", "content": question})

    all_citations: list[Citation] = []

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=history,
            tools=ALL_OAI_SCHEMAS,
            tool_choice="auto",
        )

        msg: dict[str, Any] = response.model_dump()["completion_message"]
        stop_reason = str(msg.get("stop_reason", "stop"))

        if stop_reason == "tool_calls":
            tool_calls: list[dict[str, Any]] = msg.get("tool_calls", [])

            history.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tc.get("id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": _normalize_args(tc["function"]["arguments"]),
                        },
                    }
                    for i, tc in enumerate(tool_calls)
                ],
            })

            for i, tc in enumerate(tool_calls):
                tc_id = tc.get("id", f"call_{i}")
                try:
                    inputs = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError as exc:
                    tool_text = f"Error parsing tool arguments: {exc}"
                else:
                    result = _execute_tool(tc["function"]["name"], inputs, file_data)
                    if isinstance(result, ToolResult):
                        all_citations.extend(result.citations)
                        tool_text = result.to_text()
                    else:
                        tool_text = str(result)

                history.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": tool_text,
                })

        else:
            text = _completion_message_text(msg.get("content", {}))
            return text, merge_citations(all_citations)

    fallback = "I was unable to complete the analysis within the allowed steps."
    return fallback, merge_citations(all_citations)


def _execute_tool(
    name: str, inputs: dict, file_data: dict[str, FileData]
) -> ToolResult | str:
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return f"Error: unknown tool {name!r}"
    try:
        return handler(inputs, file_data)
    except Exception as exc:
        return f"Tool error ({name}): {exc}"
