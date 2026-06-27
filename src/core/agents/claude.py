"""Claude agent using the Anthropic SDK with native tool calling."""
from __future__ import annotations

import os
from typing import Dict, List

import anthropic

from src.data import FileData
from src.types import ToolResult
from src.tools.registry import ALL_SCHEMAS
from src.utils.citations import Citation, merge_citations
from src.core.orchestrator import orchestrate, execute_tool, SYSTEM_PROMPT, MAX_TOOL_ROUNDS

MODEL = "claude-sonnet-4-5"
AGENT_LABEL = f"{MODEL} · Anthropic"


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
    system = SYSTEM_PROMPT.format(context=context)
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
            result = execute_tool(tc.name, tc.input, file_data)
            if isinstance(result, ToolResult):
                all_citations.extend(result.citations)
                text = result.to_text()
            else:
                text = str(result)
            tool_results.append({"type": "tool_result", "tool_use_id": tc.id, "content": text})

        history.append({"role": "user", "content": tool_results})

    return "I was unable to complete the analysis within the allowed steps.", merge_citations(all_citations)
