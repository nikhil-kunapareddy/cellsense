"""Pins cellsense.observability.usage: extract_usage across every provider
metadata shape, UsageTracker thread-safety, reset semantics, and
format_usage's known/unknown-cost rendering.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from cellsense.observability.usage import UsageTracker, extract_usage, format_usage
from cellsense.providers.base import ModelSpec

PRICED = ModelSpec(
    id="priced-model",
    provider="fake",
    display="Priced",
    context_window=1000,
    max_output_tokens=100,
    input_usd_per_mtok=1.0,
    output_usd_per_mtok=2.0,
)
UNPRICED = ModelSpec(
    id="unpriced-model",
    provider="fake",
    display="Unpriced",
    context_window=1000,
    max_output_tokens=100,
    input_usd_per_mtok=None,
    output_usd_per_mtok=None,
)


class TestExtractUsage:
    def test_unified_usage_metadata_shape(self) -> None:
        message = SimpleNamespace(usage_metadata={"input_tokens": 10, "output_tokens": 20})
        assert extract_usage(message) == (10, 20)

    def test_openai_raw_token_usage_shape(self) -> None:
        message = SimpleNamespace(
            usage_metadata=None,
            response_metadata={"token_usage": {"prompt_tokens": 5, "completion_tokens": 7}},
        )
        assert extract_usage(message) == (5, 7)

    def test_anthropic_raw_usage_shape(self) -> None:
        message = SimpleNamespace(
            response_metadata={"usage": {"input_tokens": 3, "output_tokens": 4}}
        )
        assert extract_usage(message) == (3, 4)

    def test_absent_usage_returns_zero_zero(self) -> None:
        message = SimpleNamespace()
        assert extract_usage(message) == (0, 0)

    def test_usage_metadata_present_but_both_zero_falls_through_to_response_metadata(self) -> None:
        message = SimpleNamespace(
            usage_metadata={"input_tokens": 0, "output_tokens": 0},
            response_metadata={"token_usage": {"prompt_tokens": 9, "completion_tokens": 1}},
        )
        assert extract_usage(message) == (9, 1)

    def test_response_metadata_present_but_not_a_dict_is_ignored(self) -> None:
        message = SimpleNamespace(usage_metadata=None, response_metadata="not a dict")
        assert extract_usage(message) == (0, 0)


class TestUsageTracker:
    def test_record_accumulates_turn_and_session_totals(self) -> None:
        tracker = UsageTracker()
        tracker.record(PRICED, 100, 50)
        tracker.record(PRICED, 200, 25)
        totals = tracker.turn_totals()
        assert totals.input_tokens == 300
        assert totals.output_tokens == 75
        assert totals.cost_usd == pytest.approx((300 * 1.0 + 75 * 2.0) / 1_000_000)
        assert tracker.session_totals() == totals

    def test_reset_turn_clears_turn_but_not_session(self) -> None:
        tracker = UsageTracker()
        tracker.record(PRICED, 100, 50)
        tracker.reset_turn()
        tracker.record(PRICED, 10, 5)
        assert tracker.turn_totals().input_tokens == 10
        assert tracker.session_totals().input_tokens == 110

    def test_any_unpriced_call_makes_the_whole_window_cost_unknown(self) -> None:
        tracker = UsageTracker()
        tracker.record(PRICED, 100, 50)
        tracker.record(UNPRICED, 100, 50)
        assert tracker.turn_totals().cost_usd is None
        # token counts are still summed even though cost is unknown
        assert tracker.turn_totals().input_tokens == 200

    def test_concurrent_record_calls_lose_no_updates(self) -> None:
        tracker = UsageTracker()
        n_threads = 20
        calls_per_thread = 50

        def _worker() -> None:
            for _ in range(calls_per_thread):
                tracker.record(PRICED, 100, 10)

        threads = [threading.Thread(target=_worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        totals = tracker.turn_totals()
        expected_input = n_threads * calls_per_thread * 100
        expected_output = n_threads * calls_per_thread * 10
        assert totals.input_tokens == expected_input
        assert totals.output_tokens == expected_output
        assert totals.cost_usd == pytest.approx(
            (expected_input * 1.0 + expected_output * 2.0) / 1_000_000
        )


class TestFormatUsage:
    def test_known_cost_is_rendered_with_four_decimals(self) -> None:
        assert format_usage(100, 50, 0.0013) == "150 tok · $0.0013"

    def test_unknown_cost_renders_a_dash_never_zero_dollars(self) -> None:
        out = format_usage(100, 50, None)
        assert out.endswith("· -")
        assert "$0.00" not in out

    def test_token_count_under_1000_is_shown_bare(self) -> None:
        assert format_usage(10, 10, 0.0) == "20 tok · $0.0000"

    def test_token_count_in_thousands_uses_k_suffix(self) -> None:
        assert format_usage(1200, 0, 0.0).startswith("1.2k tok")

    def test_token_count_in_millions_uses_m_suffix(self) -> None:
        assert format_usage(2_000_000, 0, 0.0).startswith("2.0m tok")
