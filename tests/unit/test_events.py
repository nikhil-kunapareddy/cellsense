"""Pins cellsense.events: every event dataclass has the documented 'kind'
discriminator, and the module's own zero-cellsense-imports contract (also
checked structurally by test_architecture.py's AST-based scan).
"""

from __future__ import annotations

import cellsense.events as ev


def test_every_public_event_type_has_a_distinct_kind() -> None:
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
    kinds = [instance.kind for instance in instances]
    assert len(kinds) == len(set(kinds))
    assert "event" not in kinds  # base Event's default kind is never used by a real event


def test_base_event_kind_is_event() -> None:
    assert ev.Event().kind == "event"


def test_kind_is_not_settable_via_constructor() -> None:
    # kind uses field(init=False, ...) on every subclass -- passing it
    # positionally/by keyword must not be possible.
    notice = ev.Notice(message="m")
    assert notice.kind == "notice"


def test_text_delta_defaults_to_answer_channel() -> None:
    assert ev.TextDelta(text="hi").channel == "answer"


def test_notice_defaults_to_info_level() -> None:
    assert ev.Notice(message="m").level == "info"


def test_turn_failed_defaults_are_not_cancelled_and_no_hint() -> None:
    failure = ev.TurnFailed(message="m")
    assert failure.cancelled is False
    assert failure.hint is None


def test_all_exports_are_importable_from_the_module() -> None:
    for name in ev.__all__:
        assert hasattr(ev, name)
