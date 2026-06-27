"""OpenAI-compatible agent backend for Groq, Gemini, and Llama.

All three providers expose an OpenAI-compatible HTTP interface via the openai SDK.
Differences are captured in AgentConfig so the tool-calling loop stays DRY.

Required env vars (per config):
  GROQ_API_KEY / GEMINI_API_KEY / LLAMA_API_KEY

Optional env vars:
  GROQ_MODEL / GEMINI_MODEL / LLAMA_MODEL  — override the default model
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable

import openai

from src.data import FileData
from src.types import ToolResult
from src.tools.registry import ALL_OAI_SCHEMAS
from src.utils.citations import Citation, merge_citations
from src.core.orchestrator import orchestrate, execute_tool, SYSTEM_PROMPT, MAX_TOOL_ROUNDS


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    api_key_env: str
    base_url: str
    model_env: str
    default_model: str
    label_suffix: str
    meta_response_format: bool = False  # Meta returns completion_message instead of choices
    normalize_bool_args: bool = False   # Meta sometimes emits booleans as strings


GROQ = AgentConfig(
    api_key_env="GROQ_API_KEY",
    base_url="https://api.groq.com/openai/v1",
    model_env="GROQ_MODEL",
    default_model="llama-3.3-70b-versatile",
    label_suffix="Groq",
)

GEMINI = AgentConfig(
    api_key_env="GEMINI_API_KEY",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    model_env="GEMINI_MODEL",
    default_model="gemini-2.0-flash",
    label_suffix="Google Gemini",
)

LLAMA = AgentConfig(
    api_key_env="LLAMA_API_KEY",
    base_url=os.environ.get("LLAMA_BASE_URL", "https://api.llama.com/v1"),
    model_env="LLAMA_MODEL",
    default_model="Llama-3.3-70B-Instruct",
    label_suffix="Meta Llama API",
    meta_response_format=True,
    normalize_bool_args=True,
)


# ── Public factory ─────────────────────────────────────────────────────────────

def get_label(config: AgentConfig) -> str:
    model = os.environ.get(config.model_env, config.default_model)
    return f"{model} · {config.label_suffix}"


def make_ask_fn(config: AgentConfig) -> Callable:
    """Return an ask_question function bound to the given provider config."""
    model = os.environ.get(config.model_env, config.default_model)

    def ask_question(
        question: str,
        file_data: dict[str, FileData],
        context: str,
        history: list[dict],
    ) -> tuple[str, list[Citation]]:
        client = openai.OpenAI(
            api_key=os.environ[config.api_key_env],
            base_url=config.base_url,
        )

        def llm_call(messages: list[dict]) -> str:
            response = client.chat.completions.create(
                model=model, messages=messages, max_tokens=1024
            )
            if config.meta_response_format:
                data = response.model_dump()
                if "completion_message" in data:
                    return _meta_text(data["completion_message"].get("content", {}))
                choices = data.get("choices") or []
                return choices[0].get("message", {}).get("content") or "" if choices else ""
            return response.choices[0].message.content or ""

        def worker_fn(q, fd, ctx, hist):
            return _tool_loop(client, model, config, q, fd, ctx, hist)

        return orchestrate(question, file_data, context, history, llm_call=llm_call, worker_fn=worker_fn)

    return ask_question


# ── Shared tool loop ───────────────────────────────────────────────────────────

def _tool_loop(
    client: openai.OpenAI,
    model: str,
    config: AgentConfig,
    question: str,
    file_data: dict[str, FileData],
    context: str,
    history: list[dict],
) -> tuple[str, list[Citation]]:
    system_content = SYSTEM_PROMPT.format(context=context)

    if config.meta_response_format:
        # Meta expects system prompt at history[0] and mutates it each call
        if history and history[0].get("role") == "system":
            history[0]["content"] = system_content
        else:
            history.insert(0, {"role": "system", "content": system_content})
    else:
        if not history:
            history.append({"role": "system", "content": system_content})

    history.append({"role": "user", "content": question})
    all_citations: list[Citation] = []

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=model,
            messages=history,
            tools=ALL_OAI_SCHEMAS,
            tool_choice="auto",
        )

        if config.meta_response_format:
            msg = response.model_dump()["completion_message"]
            if str(msg.get("stop_reason", "stop")) != "tool_calls":
                return _meta_text(msg.get("content", {})), merge_citations(all_citations)

            tool_calls = msg.get("tool_calls", [])
            history.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tc.get("id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": (
                                _normalize_args(tc["function"]["arguments"])
                                if config.normalize_bool_args
                                else tc["function"]["arguments"]
                            ),
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
                    result = execute_tool(tc["function"]["name"], inputs, file_data)
                    if isinstance(result, ToolResult):
                        all_citations.extend(result.citations)
                        tool_text = result.to_text()
                    else:
                        tool_text = str(result)
                history.append({"role": "tool", "tool_call_id": tc_id, "content": tool_text})

        else:
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
                    result = execute_tool(tc.function.name, inputs, file_data)
                    if isinstance(result, ToolResult):
                        all_citations.extend(result.citations)
                        tool_text = result.to_text()
                    else:
                        tool_text = str(result)
                history.append({"role": "tool", "tool_call_id": tc.id, "content": tool_text})

    return "I was unable to complete the analysis within the allowed steps.", merge_citations(all_citations)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _meta_text(content: object) -> str:
    """Extract text from Meta's completion_message content format."""
    if isinstance(content, dict):
        return str(content.get("text", ""))
    return str(content) if content else ""


def _normalize_args(arguments_str: str) -> str:
    """Coerce string 'true'/'false' to booleans — Meta sometimes emits these incorrectly."""
    try:
        args = json.loads(arguments_str)
    except json.JSONDecodeError:
        return arguments_str
    return json.dumps({
        k: (True if isinstance(v, str) and v.lower() == "true"
            else False if isinstance(v, str) and v.lower() == "false"
            else v)
        for k, v in args.items()
    })
