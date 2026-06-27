"""Shared orchestration logic and constants for all CellSense agents.

Every agent backend (Claude, Llama, …) plugs in two callbacks:

    llm_call(messages) -> str
        A plain chat call with no tools — used by the planner and synthesizer.
        ``messages`` is a list of {"role": ..., "content": ...} dicts.

    worker_fn(question, file_data, context, history) -> (answer, citations)
        The backend's own tool-calling loop — used to answer each sub-task.

Flow
----
1. Planner  — one llm_call; returns a JSON list of independent sub-questions.
2. Workers  — worker_fn called concurrently, each with an isolated history.
3. Synthesizer — one llm_call; merges sub-results into a single answer.

For simple questions the planner returns a single sub-task and the call falls
through directly to worker_fn with no added latency.
"""
from __future__ import annotations

import concurrent.futures
import json
from dataclasses import dataclass
from typing import Callable, Dict, List

from src.data import FileData
from src.types import ToolResult
from src.tools.registry import TOOL_HANDLERS
from src.utils.citations import Citation, merge_citations
from src.core.guardrails import check_relevance, rejection_message

# ── Shared constants ───────────────────────────────────────────────────────────

MAX_TOOL_ROUNDS = 8

SYSTEM_PROMPT = """\
You are CellSense, an expert data analyst assistant embedded in a CLI tool.
The user has loaded one or more Excel/CSV files. A compressed summary of those files \
is provided below. Answer the user's questions by calling the available tools to filter, \
aggregate, join, or find data as needed.

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

# ── Types ──────────────────────────────────────────────────────────────────────

LLMCallFn = Callable[[List[dict]], str]
WorkerFn = Callable[[str, Dict[str, FileData], str, List[dict]], tuple[str, List[Citation]]]

# ── Prompts ────────────────────────────────────────────────────────────────────

_PLANNER_SYSTEM = """\
You are a query decomposition planner for a data analysis CLI tool.
Decide whether the user's question should be split into independent parallel
sub-questions, then return a JSON array describing those sub-tasks.

Rules:
- Only decompose if the sub-questions are truly independent — different
  aggregations, different filters, or different files that don't depend on
  each other's results.
- If the question is simple, or sub-questions must be chained (output of one
  feeds another), return a single sub-task containing the original question.
- Maximum 4 sub-tasks.
- Return ONLY a JSON array — no prose, no markdown fences.
  Schema: [{{"id": "t1", "question": "..."}}, ...]

File context:
{context}
"""

_SYNTHESIZER_USER = """\
Independent sub-analyses have been run in parallel to answer this question:
"{question}"

Sub-results:
{sub_results}

Synthesize the above into a single concise answer. Summarise key findings —
do not repeat raw numbers already explained. End with a combined citation block:
  Sources: <filename> [Sheet: <sheet>, Rows: <i, j, ...>] | <filename2> [Rows: ...]
  (Omit Sheet for CSV files.)
"""


# ── Shared tool executor ───────────────────────────────────────────────────────

def execute_tool(name: str, inputs: dict, file_data: Dict[str, FileData]) -> ToolResult | str:
    """Dispatch a tool call to the appropriate handler."""
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return f"Error: unknown tool {name!r}"
    try:
        return handler(inputs, file_data)
    except Exception as exc:
        return f"Tool error ({name}): {exc}"


# ── Public entry point ─────────────────────────────────────────────────────────

def orchestrate(
    question: str,
    file_data: Dict[str, FileData],
    context: str,
    history: List[dict],
    *,
    llm_call: LLMCallFn,
    worker_fn: WorkerFn,
) -> tuple[str, List[Citation]]:
    """Orchestrate a user question via plan → parallel workers → synthesize."""

    # Step 0: Guardrail — reject off-topic questions before any tool calls
    if not check_relevance(question, context, llm_call):
        msg = rejection_message()
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": msg})
        return msg, []

    subtasks = _plan(question, context, llm_call)

    # Single sub-task: skip orchestration overhead entirely
    if len(subtasks) == 1:
        return worker_fn(question, file_data, context, history)

    # Fan out workers in parallel
    ordered: list[tuple[_SubTask, str, List[Citation]]] = [None] * len(subtasks)  # type: ignore[list-item]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(subtasks)) as pool:
        future_to_idx = {
            pool.submit(worker_fn, st.question, file_data, context, []): i
            for i, st in enumerate(subtasks)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            answer, citations = future.result()
            ordered[idx] = (subtasks[idx], answer, citations)

    final_answer, all_citations = _synthesize(question, ordered, llm_call)

    # Record a single clean turn in the shared history
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": final_answer})

    return final_answer, merge_citations(all_citations)


# ── Internal helpers ───────────────────────────────────────────────────────────

@dataclass
class _SubTask:
    id: str
    question: str


def _plan(question: str, context: str, llm_call: LLMCallFn) -> list[_SubTask]:
    system_msg = {"role": "system", "content": _PLANNER_SYSTEM.format(context=context)}
    user_msg = {"role": "user", "content": question}
    try:
        raw = llm_call([system_msg, user_msg])
        items = json.loads(raw)
        return [_SubTask(id=item["id"], question=item["question"]) for item in items]
    except Exception:
        return [_SubTask(id="t1", question=question)]


def _synthesize(
    original_question: str,
    sub_results: list[tuple[_SubTask, str, List[Citation]]],
    llm_call: LLMCallFn,
) -> tuple[str, List[Citation]]:
    sub_results_text = "\n\n".join(
        f"[{st.id}] Sub-question: {st.question}\nAnswer: {answer}"
        for st, answer, _ in sub_results
    )
    all_citations = [c for _, _, cits in sub_results for c in cits]
    user_msg = {"role": "user", "content": _SYNTHESIZER_USER.format(
        question=original_question,
        sub_results=sub_results_text,
    )}
    answer = llm_call([user_msg])
    return answer, all_citations
