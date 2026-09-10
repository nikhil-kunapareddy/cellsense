"""Pins cellsense.ui.stream's consumption of the engine iterator.

``run_turn`` stops iterating the moment it sees a terminal event, which leaves
``Engine.stream_turn`` suspended inside its ``with bind(...)`` block. These
tests pin that it is closed deterministically rather than abandoned to the
garbage collector -- see ``test_logging.py`` for why that distinction matters.
"""

from __future__ import annotations

import io
import threading
from collections.abc import Iterator

import pytest
from rich.console import Console

from cellsense.events import Event, RunStarted, TextDelta, TurnFailed, TurnFinished
from cellsense.ui.stream import run_turn
from cellsense.ui.theme import resolve_theme


class _RecordingEngine:
    """Engine stub whose ``stream_turn`` generator we keep a handle on."""

    model_label = "fake-model"
    provider_label = "Fake"

    def __init__(self, tail: list[Event] | None = None) -> None:
        self.generator: Iterator[Event] | None = None
        self.entered = False
        self.exited = False
        self._tail = tail or []

    def stream_turn(self, question, *, thread_id, on_approval, cancel=None):
        def _gen() -> Iterator[Event]:
            self.entered = True
            try:
                yield RunStarted(
                    question=question,
                    thread_id=thread_id,
                    model=self.model_label,
                    provider=self.provider_label,
                )
                yield TextDelta(text="hello")
                yield TurnFinished(answer="hello", citations=[], duration_s=0.01)
                yield from self._tail  # never reached: run_turn stops at the terminal event
            finally:
                self.exited = True

        self.generator = _gen()
        return self.generator


def _console() -> Console:
    # Not a terminal -> run_turn takes its plain, non-Live path, which is what
    # makes this runnable headlessly.
    return Console(file=io.StringIO(), force_terminal=False, width=80)


def _run(engine: _RecordingEngine) -> Event:
    return run_turn(
        engine,
        "a question",
        thread_id="t1",
        console=_console(),
        theme=resolve_theme("no-color"),
    )


def test_run_turn_returns_the_terminal_event() -> None:
    engine = _RecordingEngine()
    assert isinstance(_run(engine), TurnFinished)


def test_the_engine_generator_is_closed_not_abandoned() -> None:
    """The regression. ``run_turn`` breaks on the terminal event, so without an
    explicit close the generator stays suspended mid-``with`` until the garbage
    collector finalizes it on some other thread -- which raised
    ``ValueError: <Token ...> was created in a different Context`` and printed
    an ignored-exception traceback after the answer.
    """
    engine = _RecordingEngine(tail=[TextDelta(text="unreached")])
    _run(engine)

    assert engine.entered, "precondition: the generator actually started"
    assert engine.exited, "the generator's finally never ran -- it was abandoned"
    assert engine.generator is not None
    assert engine.generator.gi_frame is None, "generator still suspended after run_turn"


def test_closing_is_done_on_the_pump_thread_so_context_unwinds_cleanly() -> None:
    """Closing must happen on the thread that resumed the generator, because a
    ContextVar token is only valid in the Context that created it.
    """
    seen: dict[str, int | None] = {"resumed_on": None, "closed_on": None}

    class _ThreadAwareEngine(_RecordingEngine):
        def stream_turn(self, question, *, thread_id, on_approval, cancel=None):
            def _gen() -> Iterator[Event]:
                try:
                    seen["resumed_on"] = threading.get_ident()
                    yield TurnFinished(answer="", citations=[], duration_s=0.0)
                finally:
                    seen["closed_on"] = threading.get_ident()

            self.generator = _gen()
            return self.generator

    _run(_ThreadAwareEngine())
    assert seen["resumed_on"] is not None
    assert seen["closed_on"] == seen["resumed_on"]
    assert seen["closed_on"] != threading.get_ident(), "should not unwind on the main thread"


@pytest.mark.parametrize(
    "terminal",
    [
        TurnFinished(answer="ok", citations=[], duration_s=0.0),
        TurnFailed(message="boom"),
    ],
    ids=["finished", "failed"],
)
def test_generator_is_closed_for_either_terminal_event(terminal: Event) -> None:
    engine = _RecordingEngine()

    def stream_turn(question, *, thread_id, on_approval, cancel=None):
        def _gen() -> Iterator[Event]:
            try:
                yield terminal
                yield TextDelta(text="unreached")
            finally:
                engine.exited = True

        engine.generator = _gen()
        return engine.generator

    engine.stream_turn = stream_turn  # type: ignore[method-assign]
    _run(engine)
    assert engine.exited
    assert engine.generator is not None and engine.generator.gi_frame is None


def test_a_generator_that_raises_on_close_does_not_break_the_turn() -> None:
    """Cleanup must never mask a completed turn."""
    engine = _RecordingEngine()

    def stream_turn(question, *, thread_id, on_approval, cancel=None):
        def _gen() -> Iterator[Event]:
            try:
                yield TurnFinished(answer="ok", citations=[], duration_s=0.0)
                yield TextDelta(text="unreached")
            except GeneratorExit:
                raise RuntimeError("cleanup exploded") from None

        engine.generator = _gen()
        return engine.generator

    engine.stream_turn = stream_turn  # type: ignore[method-assign]
    assert isinstance(_run(engine), TurnFinished)


def test_print_mode_closes_the_generator_when_interrupted() -> None:
    """``run_print`` returns 130 straight out of its KeyboardInterrupt branch,
    which would otherwise abandon the generator mid-``bind()``.
    """
    from cellsense.ui.printer import run_print

    state = {"closed": False}

    class _InterruptingEngine:
        model_label = "fake-model"
        provider_label = "Fake"

        def stream_turn(self, question, *, thread_id, on_approval, cancel=None):
            def _gen() -> Iterator[Event]:
                try:
                    yield RunStarted(
                        question=question, thread_id=thread_id, model="m", provider="p"
                    )
                    raise KeyboardInterrupt
                finally:
                    state["closed"] = True

            self.generator = _gen()
            return self.generator

    engine = _InterruptingEngine()
    code = run_print(
        engine,
        "q",
        thread_id="t1",
        console=Console(file=io.StringIO(), width=80),
        error_console=Console(file=io.StringIO(), width=80),
    )
    assert code == 130
    assert state["closed"], "generator abandoned on the interrupt path"
    assert engine.generator.gi_frame is None
