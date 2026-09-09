"""Rate-limit retry/backoff (``cellsense.graph.nodes._stream_and_merge`` /
``_invoke_with_retry`` -- "BUG 2" in that module's docstring): a live Groq run
once surfaced a raw, untranslated 429. Every model call now goes through
``providers.base.translate_provider_error`` and a bounded, backoff-respecting
retry that honours the provider's own "try again in Ns" hint.

Both tests disable the guardrail and planner (``AgentConfig(guardrail=False,
planner=False)``) so the *only* model call in the graph is the top-level
``agent`` node's -- isolating the retry loop under test from any other call
site. ``time.sleep`` is monkeypatched so a bounded (3 retries) but real
backoff schedule costs nothing in wall-clock time.
"""

from __future__ import annotations

import cellsense.graph.nodes as nodes_module
from cellsense.config import AgentConfig, Config
from cellsense.events import Notice, TurnFailed, TurnFinished
from tests.fixtures.fake_chat_model import ai_text


class _FakeRateLimitError(Exception):
    """Shaped so ``providers.base.translate_provider_error`` recognizes it as
    a 429 (via ``.status_code``) -- exactly how real SDK exceptions are
    inspected: no import of any real provider SDK required.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.status_code = 429


def _no_guardrail_no_planner() -> Config:
    return Config(agent=AgentConfig(guardrail=False, planner=False))


def test_retries_with_backoff_honouring_the_retry_after_hint_then_succeeds(
    make_chat_model, make_engine, make_workspace, sales_df, assert_one_terminal_event, monkeypatch
) -> None:
    waits: list[float] = []
    monkeypatch.setattr(nodes_module.time, "sleep", lambda s: waits.append(s))

    calls = {"n": 0}

    def script(_messages):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _FakeRateLimitError("Rate limited, try again in 2.5s")
        return ai_text("Done.")

    ws = make_workspace(sales=sales_df)
    model = make_chat_model(script)
    engine = make_engine(ws, model, config=_no_guardrail_no_planner())

    events = list(
        engine.stream_turn("total revenue?", thread_id="r1", on_approval=lambda _r: "deny")
    )

    terminal = assert_one_terminal_event(events)
    assert isinstance(terminal, TurnFinished)
    assert terminal.answer == "Done."
    assert calls["n"] == 3  # 2 failures + 1 success, no more

    # One Notice per wait, and the sleep duration honours the "try again in
    # Ns" hint from the (translated) provider error, not some other default.
    notices = [e for e in events if isinstance(e, Notice)]
    assert len(notices) == 2
    assert all(n.level == "warn" for n in notices)
    assert all("rate-limited" in n.message.lower() for n in notices)
    assert waits == [2.5, 2.5]


def test_eventually_fails_cleanly_with_a_hint_instead_of_hanging_forever(
    make_chat_model, make_engine, make_workspace, sales_df, assert_one_terminal_event, monkeypatch
) -> None:
    waits: list[float] = []
    monkeypatch.setattr(nodes_module.time, "sleep", lambda s: waits.append(s))

    calls = {"n": 0}

    def script(_messages):
        calls["n"] += 1
        raise _FakeRateLimitError("Rate limited, try again in 1s")

    ws = make_workspace(sales=sales_df)
    model = make_chat_model(script)
    engine = make_engine(ws, model, config=_no_guardrail_no_planner())

    events = list(
        engine.stream_turn("total revenue?", thread_id="r2", on_approval=lambda _r: "deny")
    )

    terminal = assert_one_terminal_event(events)
    assert isinstance(terminal, TurnFailed)
    assert not terminal.cancelled
    # A helpful, provider-attributed message -- not a bare traceback, and
    # not a silent hang (the whole test just ran, monkeypatched, in well
    # under a second despite the "instant" sleeps standing in for real waits).
    assert "fake" in terminal.message.lower()  # provider_name is echoed in
    assert "rate limited" in terminal.message.lower()

    # nodes._MAX_RETRIES == 3: three waits, then the fourth attempt raises
    # for good instead of retrying a fourth time.
    assert len(waits) == 3
    assert calls["n"] == 4
    notices = [e for e in events if isinstance(e, Notice)]
    assert len(notices) == 3
