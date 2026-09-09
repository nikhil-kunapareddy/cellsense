"""Permission gates end-to-end: ``plot`` is ``side_effect=True`` and not on
the default ``permissions.allow`` list, so under the default ``"prompt"``
mode every call must pause for approval (ARCHITECTURE.md SS6/SS3).

The subtlest case is "allow after resume executes the side effect exactly
once": ``nodes.py``'s module docstring explains that ``interrupt()`` sits
*immediately before* the one call that performs the side effect, specifically
because LangGraph re-executes the whole node from the top on resume -- any
code before that point (permission decisions, the ``tool_started`` event)
reruns, but the aborted pre-interrupt pass never reaches ``tool_registry.run()``.
These tests assert on the actual PNG file on disk, not just on events, per
the task brief.
"""

from __future__ import annotations

from pathlib import Path

import cellsense.tools.plot as plot_module
from cellsense.events import (
    ApprovalDecision,
    ApprovalRequested,
    ToolFinished,
    ToolStarted,
    TurnFinished,
)
from tests.fixtures.fake_chat_model import (
    ai_text,
    ai_tool_call,
    has_tool_message,
    is_guardrail_call,
)


def _plot_script(final_text: str):
    def script(messages):
        if is_guardrail_call(messages):
            return ai_text("yes")
        if has_tool_message(messages):
            return ai_text(final_text)
        return ai_tool_call(
            "plot",
            {"chart_type": "bar", "x_column": "region", "y_column": "revenue"},
            call_id="plot-1",
        )

    return script


def _pngs(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob("*.png")) if output_dir.exists() else []


class TestDeny:
    def test_deny_means_no_png_is_written(
        self, make_chat_model, make_engine, make_workspace, sales_df, monkeypatch, tmp_path
    ) -> None:
        output_dir = tmp_path / "output"
        monkeypatch.setattr(plot_module, "OUTPUT_DIR", output_dir)

        ws = make_workspace(sales=sales_df)
        model = make_chat_model(_plot_script("Declined to plot."))
        engine = make_engine(ws, model)

        events = list(
            engine.stream_turn(
                "plot revenue by region", thread_id="deny-1", on_approval=lambda _r: "deny"
            )
        )

        assert _pngs(output_dir) == []
        requests = [e for e in events if isinstance(e, ApprovalRequested)]
        decisions = [e for e in events if isinstance(e, ApprovalDecision)]
        assert len(requests) == 1
        assert requests[0].tool == "plot"
        assert len(decisions) == 1
        assert decisions[0].outcome == "deny"
        terminal = [e for e in events if isinstance(e, TurnFinished)]
        assert len(terminal) == 1
        assert terminal[0].answer == "Declined to plot."


class TestAllow:
    def test_allow_writes_exactly_one_png(
        self, make_chat_model, make_engine, make_workspace, sales_df, monkeypatch, tmp_path
    ) -> None:
        output_dir = tmp_path / "output"
        monkeypatch.setattr(plot_module, "OUTPUT_DIR", output_dir)

        ws = make_workspace(sales=sales_df)
        model = make_chat_model(_plot_script("Plotted it."))
        engine = make_engine(ws, model)

        events = list(
            engine.stream_turn(
                "plot revenue by region", thread_id="allow-1", on_approval=lambda _r: "allow"
            )
        )

        pngs = _pngs(output_dir)
        assert len(pngs) == 1
        assert pngs[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

        terminal = [e for e in events if isinstance(e, TurnFinished)]
        assert len(terminal) == 1
        assert terminal[0].answer == "Plotted it."

    def test_allow_after_resume_does_not_run_the_side_effect_twice(
        self, make_chat_model, make_engine, make_workspace, sales_df, monkeypatch, tmp_path
    ) -> None:
        """The node re-executes from the top on ``Command(resume=...)`` --
        this is the exact scenario ``nodes.py`` calls out as the one the
        ``interrupt()``-placement discipline exists to protect. Assert on the
        filesystem, not just on the event count, since a duplicated event
        alone wouldn't prove a duplicated *write*.
        """
        output_dir = tmp_path / "output"
        monkeypatch.setattr(plot_module, "OUTPUT_DIR", output_dir)

        ws = make_workspace(sales=sales_df)
        model = make_chat_model(_plot_script("Plotted it once."))
        engine = make_engine(ws, model)

        events = list(
            engine.stream_turn(
                "plot revenue by region", thread_id="allow-2", on_approval=lambda _r: "allow"
            )
        )

        assert len(_pngs(output_dir)) == 1

        # tool_started/tool_finished for the plot call must appear exactly
        # once each in the surfaced event stream too, even though the tools
        # node body actually executed twice (once aborted at the interrupt,
        # once for real on resume) -- Engine._translate_custom's per-
        # (kind, call_id) dedup is what makes this true.
        starts = [e for e in events if isinstance(e, ToolStarted) and e.name == "plot"]
        finishes = [e for e in events if isinstance(e, ToolFinished) and e.name == "plot"]
        assert len(starts) == 1
        assert len(finishes) == 1

    def test_allow_always_is_remembered_for_a_later_call_in_the_same_session(
        self, make_chat_model, make_engine, make_workspace, sales_df, monkeypatch, tmp_path
    ) -> None:
        output_dir = tmp_path / "output"
        monkeypatch.setattr(plot_module, "OUTPUT_DIR", output_dir)

        ws = make_workspace(sales=sales_df)
        model = make_chat_model(_plot_script("Plotted it."))
        engine = make_engine(ws, model)

        decisions_issued: list[str] = []

        def on_approval(_request):
            decisions_issued.append("allow_always")
            return "allow_always"

        list(
            engine.stream_turn(
                "plot revenue by region", thread_id="allow-3", on_approval=on_approval
            )
        )
        assert len(_pngs(output_dir)) == 1
        assert len(decisions_issued) == 1

        # A second turn, same Engine (same PermissionPolicy.session_allow),
        # different thread -- must NOT pause for approval again.
        def on_approval_should_not_be_called(_request):
            raise AssertionError("plot should already be session-allowed")

        second_events = list(
            engine.stream_turn(
                "plot revenue by region again",
                thread_id="allow-4",
                on_approval=on_approval_should_not_be_called,
            )
        )
        assert len(_pngs(output_dir)) == 2
        second_requests = [e for e in second_events if isinstance(e, ApprovalRequested)]
        assert second_requests == []
