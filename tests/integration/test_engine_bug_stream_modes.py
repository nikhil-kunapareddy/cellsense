"""Regression test for a bug that breaks every real ``Engine.stream_turn()``
call against the installed langgraph (1.2.11, satisfying this project's own
``langgraph>=1.0`` pin).

``cellsense/graph/engine.py:80`` declares::

    STREAM_MODES: Sequence[_StreamMode] = ("custom", "updates", "messages")

a **tuple**, with a comment explaining that a ``list[str]`` literal was
rejected as "too wide" for mypy. But ``langgraph.pregel.main.Pregel.stream``'s
internal ``_output`` helper only tags each yielded chunk with its
``(mode, payload)`` pair when ``isinstance(stream_mode, list)`` is true; for
any other sequence type (a tuple, in this case) it falls through to
``yield payload`` -- the bare, untagged chunk. ``Engine.stream_turn`` then
does::

    for mode, chunk in self._app.stream(stream_input, config, stream_mode=STREAM_MODES):

which, fed a bare dict with more than two keys (e.g. the very first event of
every turn, ``{"type": "node_started", "node": "guardrail", "subtask_id": None}``,
three keys), raises ``ValueError: too many values to unpack (expected 2)``.
That escapes the inner loop, is caught by ``stream_turn``'s own
``except Exception`` handler, translated, and reported as a ``TurnFailed`` --
so the *hard contract* (exactly one terminal event) technically survives, but
every single real turn fails instead of answering.

Confirmed directly against the installed langgraph:

>>> isinstance(("a", "b"), list)
False
>>> # langgraph/pregel/main.py, class Pregel's _output(...) generator:
>>> #     elif isinstance(stream_mode, list):
>>> #         yield (mode, payload)
>>> #     ...
>>> #     else:
>>> #         yield payload

This test builds a completely benign, on-topic, tool-free turn (guardrail
says "yes", planner declines to decompose, the agent answers directly) --
exactly the case that should produce a plain ``TurnFinished`` -- against the
the real shipped ``STREAM_MODES``,
i.e. the real shipped behaviour). Every other test in this package passes
documented workaround so it can exercise the graph's intended behaviour
instead of tripping this same unrelated crash on every scenario -- see that
fixture's docstring in ``tests/integration/conftest.py``.
"""

from __future__ import annotations

from cellsense.events import TurnFailed, TurnFinished
from tests.fixtures.fake_chat_model import ai_text, is_guardrail_call, is_planner_call


def test_stream_turn_completes_a_trivial_turn_with_the_real_stream_modes(
    make_chat_model, make_engine, make_workspace, sales_df
) -> None:
    def script(messages):
        if is_guardrail_call(messages):
            return ai_text("yes")
        if is_planner_call(messages):
            return ai_text('[{"id": "t1", "question": "hello"}]')
        return ai_text("Hello yourself.")

    ws = make_workspace(sales=sales_df)
    model = make_chat_model(script)
    engine = make_engine(ws, model)

    events = list(engine.stream_turn("hello", thread_id="bug-repro", on_approval=lambda _r: "deny"))

    terminals = [e for e in events if isinstance(e, (TurnFinished, TurnFailed))]
    # The hard contract (exactly one terminal event) actually survives this
    # bug -- stream_turn's top-level except-Exception still yields exactly
    # one TurnFailed. What should NOT survive, but does today, is that it is
    # a TurnFailed instead of a TurnFinished for a totally benign question.
    assert len(terminals) == 1, f"hard contract violated: {terminals!r}"
    terminal = terminals[0]
    assert isinstance(terminal, TurnFinished), (
        f"expected a normal TurnFinished, got {terminal!r}; "
        "see this test's module docstring for the root cause"
    )
