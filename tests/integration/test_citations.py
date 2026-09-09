"""Citations end-to-end (ARCHITECTURE.md SS2's rendering rule + SS6's turn
lifecycle): a tool that touches every row of a large table must render an
honest row *count*, and the engine's own computed citations are authoritative
over anything the model writes into its own answer text.
"""

from __future__ import annotations

from cellsense.events import TurnFinished
from cellsense.graph.engine import _strip_model_citations
from cellsense.io.schema import Citation, format_citations, merge_citations
from tests.fixtures.fake_chat_model import (
    ai_text,
    ai_tool_call,
    has_tool_message,
    is_guardrail_call,
)


def test_a_tool_touching_all_240_rows_renders_an_honest_row_count(
    make_chat_model, make_engine, make_workspace, big_sales_frame
) -> None:
    df = big_sales_frame  # 240 rows, > io.schema._MAX_ROW_INDICES (20)
    assert len(df) == 240

    def script(messages):
        if is_guardrail_call(messages):
            return ai_text("yes")
        if has_tool_message(messages):
            # The model appends its own (now-stale, and differently
            # formatted) citation line -- the engine's own computed
            # citations must win, and this line must be stripped entirely.
            return ai_text(
                "Total revenue summed across everything.\n\nSources: sales_2024.csv [Rows: 0]"
            )
        return ai_tool_call("aggregate", {"aggregations": {"revenue": "sum"}}, call_id="agg-1")

    ws = make_workspace(sales_2024=df)
    model = make_chat_model(script)
    engine = make_engine(ws, model)

    events = list(
        engine.stream_turn("total revenue?", thread_id="cite-1", on_approval=lambda _r: "deny")
    )
    terminal = next(e for e in events if isinstance(e, TurnFinished))

    assert terminal.answer == "Total revenue summed across everything."
    assert terminal.citations == ["sales_2024.csv [240 rows]"]


def test_turn_finished_citations_equal_the_single_renderer_output(
    make_chat_model, make_engine, make_workspace, big_sales_frame
) -> None:
    """Guard the single-renderer invariant directly: whatever
    ``TurnFinished.citations`` carries must be exactly
    ``"Sources: " + " | ".join(citations)`` reconstructible into
    ``format_citations(merged)`` -- ``_citation_strings`` (engine.py) must
    never duplicate or drift from ``format_citations`` (io/schema.py).
    """
    df = big_sales_frame

    def script(messages):
        if is_guardrail_call(messages):
            return ai_text("yes")
        if has_tool_message(messages):
            return ai_text("Done.")
        return ai_tool_call("aggregate", {"aggregations": {"revenue": "sum"}}, call_id="agg-1")

    ws = make_workspace(sales_2024=df)
    model = make_chat_model(script)
    engine = make_engine(ws, model)

    events = list(
        engine.stream_turn("total revenue?", thread_id="cite-2", on_approval=lambda _r: "deny")
    )
    terminal = next(e for e in events if isinstance(e, TurnFinished))

    # Recompute independently, from the real Workspace/tool contract: an
    # aggregate() with no group_by cites every source row.
    expected_citation = Citation(filename="sales_2024.csv", sheet=None, rows=tuple(df.index))
    merged = merge_citations([expected_citation])
    expected_block = format_citations(merged)

    assert expected_block == "Sources: " + " | ".join(terminal.citations)


class TestStripModelCitations:
    """Unit-style checks of ``_strip_model_citations`` directly -- it must
    strip a model-authored ``Sources:`` line regardless of markdown emphasis,
    but never truncate ordinary prose that merely contains the word
    "sources" mid-sentence.
    """

    def test_strips_a_plain_sources_line(self) -> None:
        answer = "The total is 42.\n\nSources: a.csv [Rows: 1]"
        assert _strip_model_citations(answer) == "The total is 42."

    def test_strips_a_bold_markdown_sources_line(self) -> None:
        answer = "The total is 42.\n\n**Sources:** a.csv [Rows: 1]"
        assert _strip_model_citations(answer) == "The total is 42."

    def test_does_not_truncate_prose_that_mentions_sources_mid_sentence(self) -> None:
        answer = (
            "This report pulls from several data sources: a.csv and b.csv, "
            "both loaded successfully."
        )
        assert _strip_model_citations(answer) == answer
