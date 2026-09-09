"""Persistence: conversation history and session metadata live entirely in
the SQLite checkpointer (``graph/checkpoint.py``), not in the ``Engine``
instance -- dropping an ``Engine`` and rebuilding a new one against the same
``db_path`` must recover both.
"""

from __future__ import annotations

from cellsense.events import TurnFinished
from tests.fixtures.fake_chat_model import ai_text, is_guardrail_call, is_planner_call


def _simple_script(answer: str):
    def script(messages):
        if is_guardrail_call(messages):
            return ai_text("yes")
        if is_planner_call(messages):
            return ai_text('[{"id": "t1", "question": "q"}]')
        return ai_text(answer)

    return script


def test_history_and_sessions_survive_dropping_and_rebuilding_the_engine(
    make_chat_model, make_engine, make_workspace, sales_df, tmp_path
) -> None:
    db_path = tmp_path / "sessions.db"
    ws = make_workspace(sales=sales_df)

    engine1 = make_engine(ws, make_chat_model(_simple_script("42.")), db_path=db_path)
    events = list(
        engine1.stream_turn(
            "what is total revenue?", thread_id="persist-1", on_approval=lambda _r: "deny"
        )
    )
    assert isinstance(events[-1], TurnFinished)

    del engine1  # the point: nothing about recovery may depend on this object

    engine2 = make_engine(ws, make_chat_model(_simple_script("unused")), db_path=db_path)

    history = engine2.history("persist-1")
    assert history, "history() found nothing after rebuilding the Engine"
    roles = [role for role, _text in history]
    assert "human" in roles
    assert "ai" in roles
    texts = " ".join(text for _role, text in history)
    assert "what is total revenue?" in texts
    assert "42." in texts

    sessions = engine2.sessions()
    assert any(s.thread_id == "persist-1" for s in sessions)


def test_session_info_exposes_the_four_names_the_ui_reads(
    make_chat_model, make_engine, make_workspace, sales_df, tmp_path
) -> None:
    """ARCHITECTURE.md SS7 pins ``SessionInfo`` as a structural contract: the
    UI reads exactly ``thread_id``, ``updated_at``, ``preview``, ``turns``.
    """
    db_path = tmp_path / "sessions.db"
    ws = make_workspace(sales=sales_df)
    engine = make_engine(ws, make_chat_model(_simple_script("Answer.")), db_path=db_path)

    list(
        engine.stream_turn(
            "what is total revenue?", thread_id="persist-2", on_approval=lambda _r: "deny"
        )
    )

    sessions = engine.sessions()
    info = next(s for s in sessions if s.thread_id == "persist-2")

    assert isinstance(info.thread_id, str) and info.thread_id == "persist-2"
    assert isinstance(info.updated_at, str) and info.updated_at
    assert isinstance(info.preview, str) and "total revenue" in info.preview
    assert isinstance(info.turns, int) and info.turns > 0


def test_multiple_sessions_are_ordered_most_recently_updated_first(
    make_chat_model, make_engine, make_workspace, sales_df, tmp_path
) -> None:
    db_path = tmp_path / "sessions.db"
    ws = make_workspace(sales=sales_df)
    engine = make_engine(ws, make_chat_model(_simple_script("Answer.")), db_path=db_path)

    list(engine.stream_turn("first?", thread_id="older", on_approval=lambda _r: "deny"))
    list(engine.stream_turn("second?", thread_id="newer", on_approval=lambda _r: "deny"))

    sessions = engine.sessions()
    thread_ids = [s.thread_id for s in sessions]
    assert thread_ids.index("newer") < thread_ids.index("older")
