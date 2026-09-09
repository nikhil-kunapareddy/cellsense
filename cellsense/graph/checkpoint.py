"""SQLite-backed persistence: opening the checkpointer and reading it back as
session metadata, independent of any live graph or ``Engine`` instance.

Conversation history for every thread lives entirely in the checkpointer (see
``TurnState.messages`` in ``state.py``); nothing in the UI or the ``Engine``
keeps its own copy. This module is what lets ``cellsense session list``, a
future ``/resume`` slash command, and the "recover after restart" test in the
verification suite all work off the same on-disk database.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

# CellSense's own dataclasses that end up inside a checkpointed TurnState
# (cellsense.io.schema.Citation, cellsense.graph.state.SubtaskResult) aren't
# built-in msgpack types, so the default serializer only deserializes them
# with a "this will be blocked in a future version" warning. Registering them
# explicitly makes checkpoint round-trips silent and future-proof instead of
# relying on a deprecated fallback.
_ALLOWED_MSGPACK_MODULES = [
    ("cellsense.io.schema", "Citation"),
    ("cellsense.graph.state", "SubtaskResult"),
]

__all__ = [
    "SessionInfo",
    "delete_session",
    "list_sessions",
    "load_history",
    "open_checkpointer",
]


@dataclass(frozen=True, slots=True)
class SessionInfo:
    """One row of ``cellsense session list``: enough to let a human pick a
    thread to resume without loading its full history.
    """

    thread_id: str
    created_at: str
    """ISO 8601 timestamp of the thread's first checkpoint."""
    updated_at: str
    """ISO 8601 timestamp of the thread's most recent checkpoint."""
    question_preview: str
    """The most recent turn's question, truncated to 80 characters."""
    message_count: int
    """Length of the top-level ``messages`` transcript as of the last turn."""

    @property
    def preview(self) -> str:
        """Alias for ``question_preview``.

        ``cellsense.ui.repl`` defines its own structural ``SessionInfo``
        Protocol (``thread_id``, ``updated_at``, ``preview``, ``turns``) so
        that package never has to import ``graph``. Exposing both names here
        -- the descriptive ones this module's own callers use, plus these
        short aliases -- is what lets one dataclass satisfy both without the
        UI reading a field that doesn't exist and silently rendering blank.
        """
        return self.question_preview

    @property
    def turns(self) -> int:
        """Alias for ``message_count`` -- see :attr:`preview`."""
        return self.message_count


def open_checkpointer(path: Path) -> SqliteSaver:
    """Open (creating the file and schema if needed) a long-lived SQLite
    checkpointer at ``path``.

    ``SqliteSaver.from_conn_string`` is a context manager in LangGraph 1.x
    (LANGGRAPH_NOTES.md mechanic #4), which does not fit an ``Engine`` that
    outlives many ``stream_turn()`` calls across a whole REPL session -- so
    this builds the ``sqlite3.Connection`` directly instead, with
    ``check_same_thread=False`` per the notes' thread-safety gotcha (fan-out
    workers and the checkpointer's own writes can happen from different
    threads within a single ``app.stream()`` call).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    serde = JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_MSGPACK_MODULES)
    saver = SqliteSaver(conn, serde=serde)
    saver.setup()
    return saver


def list_sessions(saver: SqliteSaver) -> list[SessionInfo]:
    """Every thread with at least one checkpoint, most recently updated first."""
    cursor = saver.conn.execute("SELECT DISTINCT thread_id FROM checkpoints")
    thread_ids = [row[0] for row in cursor.fetchall()]
    sessions = [_session_info(saver, thread_id) for thread_id in thread_ids]
    sessions.sort(key=lambda info: info.updated_at, reverse=True)
    return sessions


def load_history(saver: SqliteSaver, thread_id: str) -> list[tuple[str, str]]:
    """The full ``[(role, text), ...]`` transcript for one thread, oldest
    first, read straight back out of the checkpointer -- no separate history
    store exists anywhere in CellSense.
    """
    tup = saver.get_tuple(_thread_config(thread_id))
    if tup is None:
        return []
    messages: list[BaseMessage] = tup.checkpoint.get("channel_values", {}).get("messages") or []
    return [(_role(message), _content_text(message)) for message in messages]


def delete_session(saver: SqliteSaver, thread_id: str) -> None:
    """Permanently remove every checkpoint and pending write for ``thread_id``."""
    saver.delete_thread(thread_id)


# ── private helpers ──────────────────────────────────────────────────────────


def _thread_config(thread_id: str) -> RunnableConfig:
    return RunnableConfig(configurable={"thread_id": thread_id})


def _session_info(saver: SqliteSaver, thread_id: str) -> SessionInfo:
    # SqliteSaver.list() yields most-recent-first; a session's full checkpoint
    # history is small enough in practice (one CLI session) that reading all of
    # it to find the earliest entry is simpler, and safer, than a bespoke
    # ORDER BY query against the serialized checkpoint blob's embedded ts.
    checkpoints = list(saver.list(_thread_config(thread_id)))
    latest, earliest = checkpoints[0], checkpoints[-1]

    values = latest.checkpoint.get("channel_values", {})
    messages = values.get("messages") or []
    question = str(values.get("question", ""))
    preview = question if len(question) <= 80 else f"{question[:77]}..."

    return SessionInfo(
        thread_id=thread_id,
        created_at=str(earliest.checkpoint.get("ts", "")),
        updated_at=str(latest.checkpoint.get("ts", "")),
        question_preview=preview,
        message_count=len(messages),
    )


def _role(message: BaseMessage) -> str:
    return str(getattr(message, "type", message.__class__.__name__.lower()))


def _content_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    # BaseMessage.content's declared type is exactly str | list[str | dict[...]],
    # so the branch above and the one below are exhaustive -- mypy proves it (a
    # trailing fallback here is genuinely unreachable, not merely undetected).
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)
