"""Input guardrails for CellSense.

Runs before the agent loop to reject questions that are not relevant to
the loaded data files. Uses the same llm_call as the active backend so no
extra API credentials are needed.
"""
from __future__ import annotations

from typing import Callable, List

LLMCallFn = Callable[[List[dict]], str]

_RELEVANCE_PROMPT = """\
You are a relevance classifier for a data analysis CLI tool.
The user has loaded spreadsheet/CSV files described below. Your job is to
decide whether the user's question is about analyzing that data.

Loaded file context:
{context}

Rules:
- Answer ONLY with "yes" or "no" — no explanation.
- "yes"  → the question asks about the data contents, statistics, filters,
           comparisons, trends, or any analysis of the loaded files;
           OR it asks to find, locate, search for, or load data files
           (e.g. "find files in ~/Downloads", "look for CSVs in the data folder");
           OR it is a greeting, farewell, or simple polite message
           (e.g. "hi", "hello", "thanks", "goodbye").
- "no"   → the question is about general knowledge, coding help, current
           events, or anything unrelated to the loaded files or finding data files.
"""

_REJECTION_MESSAGE = (
    "I can only answer questions about your loaded data files. "
    "Try asking something like:\n"
    "  • \"What is the total revenue by region?\"\n"
    "  • \"Show rows where sales > 10000\"\n"
    "  • \"Which product had the highest average price?\""
)


def check_relevance(question: str, context: str, llm_call: LLMCallFn) -> bool:
    """Return True if the question is relevant to the loaded data, False otherwise."""
    try:
        system_msg = {"role": "system", "content": _RELEVANCE_PROMPT.format(context=context)}
        user_msg = {"role": "user", "content": question}
        raw = llm_call([system_msg, user_msg]).strip().lower()
        return raw.startswith("yes")
    except Exception:
        # If the guardrail itself fails, let the question through
        return True


def rejection_message() -> str:
    return _REJECTION_MESSAGE
