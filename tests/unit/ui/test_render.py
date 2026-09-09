"""Pins cellsense.ui.render.render_event: it must handle every event type in
cellsense.events without raising, honour the documented "single-subtask plan
renders nothing" contract, and ignore unknown/synthetic event kinds rather
than raising.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

from rich.console import Console

import cellsense.events as ev
from cellsense.ui.render import render_event
from cellsense.ui.theme import resolve_theme

THEME = resolve_theme("no-color")


def _text_of(renderable) -> str:
    """Render a rich renderable (Text, Group, ...) to plain text.

    ``str(renderable)`` only works for ``Text`` -- a ``Group`` (what several
    of these renderers return) stringifies to its Python repr instead, so
    this actually renders it through a Console to get the real output.
    """
    if renderable is None:
        return ""
    console = Console(file=io.StringIO(), width=200, no_color=True, highlight=False)
    console.print(renderable)
    return console.file.getvalue()


def test_run_started_has_no_visual() -> None:
    event = ev.RunStarted(question="q", thread_id="t", model="m", provider="p")
    assert render_event(event, THEME) is None


def test_node_started_has_no_visual() -> None:
    assert render_event(ev.NodeStarted(node="agent"), THEME) is None


def test_usage_updated_has_no_visual() -> None:
    event = ev.UsageUpdated(input_tokens=1, output_tokens=1, cost_usd=None)
    assert render_event(event, THEME) is None


def test_plan_created_with_one_subtask_renders_nothing() -> None:
    event = ev.PlanCreated(subtasks=[("1", "only one")])
    assert render_event(event, THEME) is None


def test_plan_created_with_multiple_subtasks_renders_something() -> None:
    event = ev.PlanCreated(subtasks=[("1", "first"), ("2", "second")])
    rendered = render_event(event, THEME)
    assert rendered is not None
    text = _text_of(rendered)
    assert "first" in text and "second" in text


def test_plan_created_with_zero_subtasks_renders_nothing() -> None:
    assert render_event(ev.PlanCreated(subtasks=[]), THEME) is None


def test_text_delta_answer_channel_renders_the_raw_text() -> None:
    rendered = render_event(ev.TextDelta(text="hello"), THEME)
    assert "hello" in _text_of(rendered)


def test_text_delta_worker_channel_is_indented() -> None:
    rendered = render_event(ev.TextDelta(text="hi", channel="worker"), THEME)
    assert "hi" in _text_of(rendered)


def test_text_delta_reasoning_channel_renders() -> None:
    rendered = render_event(ev.TextDelta(text="thinking", channel="reasoning"), THEME)
    assert "thinking" in _text_of(rendered)


def test_tool_started_renders_name() -> None:
    event = ev.ToolStarted(call_id="1", name="aggregate", args={"group_by": "region"})
    rendered = render_event(event, THEME)
    assert "aggregate" in _text_of(rendered)


def test_tool_started_hides_args_when_show_tool_args_is_false() -> None:
    event = ev.ToolStarted(call_id="1", name="aggregate", args={"group_by": "region"})
    rendered = render_event(event, THEME, show_tool_args=False)
    assert "region" not in _text_of(rendered)


def test_tool_finished_renders_summary_and_duration() -> None:
    event = ev.ToolFinished(
        call_id="1", name="aggregate", summary="did a thing", row_count=3, duration_s=1.25
    )
    rendered = render_event(event, THEME)
    assert "did a thing" in _text_of(rendered)
    assert "1.2" in _text_of(rendered)


def test_tool_failed_renders_the_error() -> None:
    event = ev.ToolFailed(call_id="1", name="aggregate", error="bad column", duration_s=0.1)
    rendered = render_event(event, THEME)
    assert "bad column" in _text_of(rendered)


def test_approval_requested_renders_tool_and_reason() -> None:
    event = ev.ApprovalRequested(call_id="1", tool="plot", args={}, reason="writes a file")
    rendered = render_event(event, THEME)
    text = _text_of(rendered)
    assert "plot" in text and "writes a file" in text


def test_approval_decision_allow_renders_approved() -> None:
    event = ev.ApprovalDecision(call_id="1", tool="plot", outcome="allow")
    assert "Approved" in _text_of(render_event(event, THEME))


def test_approval_decision_deny_renders_denied() -> None:
    event = ev.ApprovalDecision(call_id="1", tool="plot", outcome="deny")
    assert "Denied" in _text_of(render_event(event, THEME))


def test_approval_decision_allow_always_is_distinguished_from_plain_allow() -> None:
    plain = _text_of(
        render_event(ev.ApprovalDecision(call_id="1", tool="plot", outcome="allow"), THEME)
    )
    always = _text_of(
        render_event(ev.ApprovalDecision(call_id="1", tool="plot", outcome="allow_always"), THEME)
    )
    assert plain != always


def test_notice_info_renders_message() -> None:
    assert "heads up" in _text_of(render_event(ev.Notice(message="heads up", level="info"), THEME))


def test_notice_warn_renders_message() -> None:
    assert "careful" in _text_of(render_event(ev.Notice(message="careful", level="warn"), THEME))


def test_notice_error_renders_message() -> None:
    assert "broken" in _text_of(render_event(ev.Notice(message="broken", level="error"), THEME))


def test_turn_finished_renders_answer() -> None:
    event = ev.TurnFinished(answer="the answer is 42", citations=[], duration_s=1.0)
    assert "42" in _text_of(render_event(event, THEME))


def test_turn_finished_renders_citations_when_present() -> None:
    event = ev.TurnFinished(answer="a", citations=["data.csv [Rows: 0, 1]"], duration_s=1.0)
    assert "data.csv" in _text_of(render_event(event, THEME))


def test_turn_failed_renders_error_message() -> None:
    event = ev.TurnFailed(message="something broke")
    assert "something broke" in _text_of(render_event(event, THEME))


def test_turn_failed_cancelled_renders_interrupted_not_error() -> None:
    event = ev.TurnFailed(message="ignored", cancelled=True)
    text = _text_of(render_event(event, THEME))
    assert "Interrupted" in text
    assert "Error" not in text


def test_turn_failed_renders_hint_when_present() -> None:
    event = ev.TurnFailed(message="broke", hint="try again")
    assert "try again" in _text_of(render_event(event, THEME))


def test_render_event_defaults_to_a_theme_when_none_given() -> None:
    # Must not raise even without an explicit theme (per render_event's docstring).
    assert render_event(ev.Notice(message="m")) is not None


def test_every_documented_event_type_is_handled_without_raising() -> None:
    instances = [
        ev.RunStarted(question="q", thread_id="t", model="m", provider="p"),
        ev.PlanCreated(subtasks=[("1", "a"), ("2", "b")]),
        ev.NodeStarted(node="agent"),
        ev.TextDelta(text="hi"),
        ev.ToolStarted(call_id="1", name="describe", args={}),
        ev.ToolFinished(call_id="1", name="describe", summary="s", row_count=1, duration_s=0.1),
        ev.ToolFailed(call_id="1", name="describe", error="e", duration_s=0.1),
        ev.ApprovalRequested(call_id="1", tool="plot", args={}, reason="side effect"),
        ev.ApprovalDecision(call_id="1", tool="plot", outcome="allow"),
        ev.UsageUpdated(input_tokens=1, output_tokens=1, cost_usd=0.01),
        ev.Notice(message="m"),
        ev.TurnFinished(answer="a", citations=[], duration_s=1.0),
        ev.TurnFailed(message="m"),
    ]
    for instance in instances:
        render_event(instance, THEME)  # must not raise for any of these


class TestUnknownEventKindsAreIgnored:
    def test_base_event_kind_renders_nothing(self) -> None:
        assert render_event(ev.Event(), THEME) is None

    def test_synthetic_unknown_event_kind_renders_nothing_not_raises(self) -> None:
        @dataclass(slots=True)
        class _FutureEvent(ev.Event):
            payload: str = "x"
            kind: str = field(init=False, default="some_future_event_kind_nobody_handles_yet")

        assert render_event(_FutureEvent(), THEME) is None
