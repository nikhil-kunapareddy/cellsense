"""Groq agent using the Groq API (api.groq.com).

Groq's API is fully OpenAI-compatible (uses standard choices format).

Required env var: GROQ_API_KEY
Optional env vars:
  GROQ_MODEL    — model ID (default: llama-3.3-70b-versatile)
  GROQ_BASE_URL — API base URL (default: https://api.groq.com/openai/v1)
"""
from __future__ import annotations

import json
import os

import openai

from src.data import FileData
from src.tools import ALL_OAI_SCHEMAS, TOOL_HANDLERS, ToolResult
from src.utils.citations import Citation, merge_citations
from src.agents.base import orchestrate

BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
AGENT_LABEL = f"{MODEL} · Groq"

MAX_TOOL_ROUNDS = 8

_SYSTEM_PROMPT = """\
You are CellSense, an expert data analyst assistant embedded in a CLI tool.
The user has loaded one or more Excel/CSV files. A compressed summary of those files \
is provided below. Answer the user's questions by calling the available tools to filter, \
aggregate, or join the data as needed.

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
    file_data: dict[str, FileData],
    context: str,
    history: list[dict],
) -> tuple[str, list[Citation]]:
    client = openai.OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url=BASE_URL,
    )

    def llm_call(messages: list[dict]) -> str:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""

    def worker_fn(
        q: str, fd: dict[str, FileData], ctx: str, hist: list[dict]
    ) -> tuple[str, list[Citation]]:
        return _tool_loop(client, q, fd, ctx, hist)

    return orchestrate(
        question, file_data, context, history, llm_call=llm_call, worker_fn=worker_fn
    )


def _tool_loop(
    client: openai.OpenAI,
    question: str,
    file_data: dict[str, FileData],
    context: str,
    history: list[dict],
) -> tuple[str, list[Citation]]:
    if not history:
        history.append(
            {"role": "system", "content": _SYSTEM_PROMPT.format(context=context)}
        )

    history.append({"role": "user", "content": question})

    all_citations: list[Citation] = []

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=history,
            tools=ALL_OAI_SCHEMAS,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        history.append(msg.model_dump(exclude_unset=True))

        if not msg.tool_calls:
            return msg.content or "", merge_citations(all_citations)

        for tc in msg.tool_calls:
            try:
                inputs = json.loads(tc.function.arguments)
            except json.JSONDecodeError as exc:
                tool_text = f"Error parsing tool arguments: {exc}"
            else:
                result = _execute_tool(tc.function.name, inputs, file_data)
                if isinstance(result, ToolResult):
                    all_citations.extend(result.citations)
                    tool_text = result.to_text()
                else:
                    tool_text = str(result)

            history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_text,
            })

    return (
        "I was unable to complete the analysis within the allowed steps.",
        merge_citations(all_citations),
    )


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
