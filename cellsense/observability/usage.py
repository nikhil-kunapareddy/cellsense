"""Token and cost accounting: per-turn and per-session, thread-safe.

The graph fans out worker threads for multi-subtask turns (ARCHITECTURE.md §6),
each making its own model call, so :class:`UsageTracker` is written for concurrent
``record()`` calls from the start rather than bolting on a lock later.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from cellsense.providers.base import ModelSpec
from cellsense.providers.registry import estimate_cost

__all__ = ["UsageTotals", "UsageTracker", "extract_usage", "format_usage"]


@dataclass(frozen=True)
class UsageTotals:
    """A point-in-time snapshot from :class:`UsageTracker`.

    ``cost_usd`` is ``None`` if *any* contributing call used a model with unknown
    pricing -- a partial sum would understate the true cost without saying so, and
    silently pretending we know the total is worse than admitting we don't.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0


class _Accumulator:
    """Internal, lock-free totals holder -- callers must hold ``UsageTracker._lock``
    while touching one of these.
    """

    __slots__ = ("_unknown_seen", "cost_usd", "input_tokens", "output_tokens")

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self._unknown_seen = False

    def add(self, input_tokens: int, output_tokens: int, cost_usd: float | None) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        if cost_usd is None:
            self._unknown_seen = True
        else:
            self.cost_usd += cost_usd

    def snapshot(self) -> UsageTotals:
        return UsageTotals(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cost_usd=None if self._unknown_seen else self.cost_usd,
        )


class UsageTracker:
    """Accumulates token and cost totals across a session, with a resettable
    per-turn window layered on top.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._turn = _Accumulator()
        self._session = _Accumulator()

    def record(self, model_spec: ModelSpec, input_tokens: int, output_tokens: int) -> None:
        """Record one model call's usage against both the turn and session totals."""
        cost = estimate_cost(model_spec, input_tokens, output_tokens)
        with self._lock:
            self._turn.add(input_tokens, output_tokens, cost)
            self._session.add(input_tokens, output_tokens, cost)

    def turn_totals(self) -> UsageTotals:
        with self._lock:
            return self._turn.snapshot()

    def session_totals(self) -> UsageTotals:
        with self._lock:
            return self._session.snapshot()

    def reset_turn(self) -> None:
        """Start a fresh per-turn window without touching the session totals --
        call this at the top of every new turn.
        """
        with self._lock:
            self._turn = _Accumulator()


def extract_usage(message: Any) -> tuple[int, int]:
    """Pull ``(input_tokens, output_tokens)`` out of a LangChain ``AIMessage``.

    Providers disagree on where usage lives:

    * Most current integrations populate ``message.usage_metadata`` (a
      ``{"input_tokens": int, "output_tokens": int, "total_tokens": int}`` dict) --
      this is LangChain's own unified shape and is checked first.
    * Some populate ``message.response_metadata["token_usage"]`` instead, in
      either the unified shape or OpenAI's raw ``prompt_tokens`` /
      ``completion_tokens`` shape.
    * Others nest it under ``message.response_metadata["usage"]`` (Anthropic's raw
      shape uses ``input_tokens`` / ``output_tokens`` here too).

    Returns ``(0, 0)`` if none of the above are present or usable, rather than
    raising -- a missing usage block should never fail a turn.
    """
    usage_metadata = getattr(message, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        input_tokens = usage_metadata.get("input_tokens") or 0
        output_tokens = usage_metadata.get("output_tokens") or 0
        if input_tokens or output_tokens:
            return int(input_tokens), int(output_tokens)

    response_metadata = getattr(message, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage")
        if not isinstance(token_usage, dict):
            token_usage = response_metadata.get("usage")
        if isinstance(token_usage, dict):
            input_tokens = token_usage.get("input_tokens") or token_usage.get("prompt_tokens") or 0
            output_tokens = (
                token_usage.get("output_tokens") or token_usage.get("completion_tokens") or 0
            )
            if input_tokens or output_tokens:
                return int(input_tokens), int(output_tokens)

    return 0, 0


def _format_token_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}m"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def format_usage(input_tokens: int, output_tokens: int, cost_usd: float | None) -> str:
    """Render usage for the UI status line, e.g. ``"1.2k tok · $0.0013"``. Cost
    renders as ``"-"`` when unknown, never ``"$0.00"``.
    """
    total = input_tokens + output_tokens
    cost_str = f"${cost_usd:.4f}" if cost_usd is not None else "-"
    return f"{_format_token_count(total)} tok · {cost_str}"
