"""Pins cellsense.observability.logging: JSONL output, redaction,
truncation, "never writes to stdout", and idempotent handler setup.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from cellsense.config import Config
from cellsense.observability.logging import LOGGER_NAME, bind, setup_logging


@pytest.fixture(autouse=True)
def _cleanup_logger():
    """setup_logging() mutates the process-wide "cellsense" logger; always
    tear its handlers down so tests don't leak file handles or bleed into
    each other (or into the rest of the suite).
    """
    yield
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def _make_config(tmp_path) -> Config:
    config = Config()
    config.telemetry.log_dir = str(tmp_path / "logs")
    return config


def _log_file(tmp_path) -> Path:
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    return tmp_path / "logs" / f"cellsense-{today}.jsonl"


def test_setup_logging_creates_a_jsonl_file_with_valid_json_lines(tmp_path) -> None:
    config = _make_config(tmp_path)
    logger = setup_logging(config)
    logger.info("hello world")

    log_path = _log_file(tmp_path)
    assert log_path.is_file()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["message"] == "hello world"
    assert record["level"] == "INFO"
    assert record["logger"] == "cellsense"
    assert "ts" in record


def test_bound_fields_are_attached_to_every_record_in_the_block(tmp_path) -> None:
    config = _make_config(tmp_path)
    logger = setup_logging(config)
    with bind(thread_id="abc123"):
        logger.info("inside")
    logger.info("outside")

    lines = _log_file(tmp_path).read_text().strip().splitlines()
    inside, outside = (json.loads(line) for line in lines)
    assert inside["thread_id"] == "abc123"
    assert "thread_id" not in outside


def test_nested_bind_merges_and_unwinds(tmp_path) -> None:
    config = _make_config(tmp_path)
    logger = setup_logging(config)
    with bind(thread_id="t1"):
        with bind(node="agent"):
            logger.info("nested")
        logger.info("after inner")

    lines = _log_file(tmp_path).read_text().strip().splitlines()
    nested, after_inner = (json.loads(line) for line in lines)
    assert nested["thread_id"] == "t1"
    assert nested["node"] == "agent"
    assert after_inner["thread_id"] == "t1"
    assert "node" not in after_inner


def test_token_count_fields_are_not_mistaken_for_secrets(tmp_path) -> None:
    """``input_tokens``/``output_tokens`` (usage *counts*, logged on every
    ``turn finished`` record) must never be redacted -- only an actual secret
    named plain ``token`` or ``*_token`` (singular) should be. A naive
    ``"token"`` substring match would also catch the plural ``"tokens"`` in
    these two legitimate, non-secret field names.
    """
    config = _make_config(tmp_path)
    logger = setup_logging(config)
    logger.info(
        "turn finished",
        extra={"input_tokens": 123, "output_tokens": 45, "auth_token": "sk-should-be-redacted"},
    )

    record = json.loads(_log_file(tmp_path).read_text().strip().splitlines()[0])
    assert record["input_tokens"] == 123
    assert record["output_tokens"] == 45
    assert record["auth_token"] == "***redacted***"


def test_sensitive_field_names_are_redacted(tmp_path) -> None:
    config = _make_config(tmp_path)
    logger = setup_logging(config)
    with bind(api_key="sk-super-secret-value"):
        logger.info("call made")

    raw = _log_file(tmp_path).read_text()
    assert "sk-super-secret-value" not in raw
    record = json.loads(raw.strip().splitlines()[0])
    assert record["api_key"] == "***redacted***"


def test_long_field_values_are_truncated(tmp_path) -> None:
    config = _make_config(tmp_path)
    logger = setup_logging(config)
    huge = "x" * 3000
    logger.info("payload", extra={"payload": huge})

    record = json.loads(_log_file(tmp_path).read_text().strip().splitlines()[0])
    assert len(record["payload"]) < 3000
    assert "truncated" in record["payload"]


def test_long_message_itself_is_truncated(tmp_path) -> None:
    config = _make_config(tmp_path)
    logger = setup_logging(config)
    logger.info("y" * 3000)

    record = json.loads(_log_file(tmp_path).read_text().strip().splitlines()[0])
    assert len(record["message"]) < 3000
    assert "truncated" in record["message"]


def test_never_writes_to_stdout(tmp_path, capsys) -> None:
    config = _make_config(tmp_path)
    logger = setup_logging(config)
    logger.info("an info line")
    logger.warning("a warning line")

    captured = capsys.readouterr()
    assert captured.out == ""


def test_warning_and_above_go_to_stderr_but_info_does_not(tmp_path, capsys) -> None:
    config = _make_config(tmp_path)
    logger = setup_logging(config)
    logger.info("quiet")
    captured = capsys.readouterr()
    assert captured.err == ""

    logger.warning("loud")
    captured = capsys.readouterr()
    assert "loud" in captured.err


def test_debug_flag_sends_everything_to_stderr(tmp_path, capsys) -> None:
    config = _make_config(tmp_path)
    logger = setup_logging(config, debug=True)
    logger.debug("debug line")
    captured = capsys.readouterr()
    assert "debug line" in captured.err


def test_repeated_setup_does_not_accumulate_handlers(tmp_path) -> None:
    config = _make_config(tmp_path)
    setup_logging(config)
    setup_logging(config)
    logger = logging.getLogger(LOGGER_NAME)
    assert len(logger.handlers) == 2  # one file handler + one stderr handler


def test_level_is_read_from_config(tmp_path) -> None:
    config = _make_config(tmp_path)
    config.telemetry.level = "warning"
    logger = setup_logging(config)
    assert logger.level == logging.WARNING


# ── regression tests pinning the guarantees graph/nodes.py, graph/engine.py,
# cli.py, and tools/registry.py rely on for their instrumentation ──────────


def test_a_logging_call_never_raises_even_if_formatting_fails(tmp_path, capsys) -> None:
    """Logging must never break a turn: every call site added across the
    engine/nodes/tools/cli instrumentation trusts that a bad `logger.x(...)`
    call (e.g. a mismatched ``%s``) cannot itself raise. The stdlib's own
    ``Handler.handleError`` swallows formatting failures (reporting them on
    stderr, never stdout, and never propagating) -- this pins that behavior
    rather than assuming it.
    """
    config = _make_config(tmp_path)
    logger = setup_logging(config)
    logger.info("boom %s")  # %s wants an arg; none supplied -- must not raise
    assert capsys.readouterr().out == ""


def test_none_valued_bound_fields_serialize_as_json_null(tmp_path) -> None:
    """``_node_span``/``bind()`` call sites always pass ``subtask_id`` (often
    ``None`` on the top-level branch, never omitted) so every record has a
    consistent field set; ``None`` must round-trip as JSON ``null``, not
    crash truncation/redaction or get silently dropped.
    """
    config = _make_config(tmp_path)
    logger = setup_logging(config)
    with bind(thread_id="t1", subtask_id=None):
        logger.info("node entry")

    record = json.loads(_log_file(tmp_path).read_text().strip().splitlines()[0])
    assert record["thread_id"] == "t1"
    assert record["subtask_id"] is None


def test_non_string_extra_values_pass_through_unchanged(tmp_path) -> None:
    """Duration/row-count/boolean fields (``duration_s``, ``row_count``,
    ``fell_back``, ...) are logged as real JSON numbers/booleans, not
    stringified -- ``_truncate`` only touches ``str`` values, so everything
    else must survive ``_sanitize`` untouched.
    """
    config = _make_config(tmp_path)
    logger = setup_logging(config)
    logger.info(
        "tool ran: %s",
        "aggregate",
        extra={"duration_s": 0.1234, "row_count": 42, "fell_back": False},
    )

    record = json.loads(_log_file(tmp_path).read_text().strip().splitlines()[0])
    assert record["duration_s"] == 0.1234
    assert record["row_count"] == 42
    assert record["fell_back"] is False


def test_extra_dict_must_not_reuse_a_logrecord_attribute_name(tmp_path) -> None:
    """``extra={"args": ...}`` collides with ``LogRecord``'s own ``args``
    attribute and raises ``KeyError`` at the call site -- exactly why
    ``tools.registry.ToolRegistry.run()`` logs a tool call's arguments under
    the key ``tool_args``, never ``args``. Pinned here so a future call site
    doesn't reintroduce the collision.
    """
    config = _make_config(tmp_path)
    logger = setup_logging(config)
    with pytest.raises(KeyError):
        logger.info("tool args", extra={"args": "()"})


def test_bind_unwinds_even_if_the_block_raises(tmp_path) -> None:
    """``graph.nodes._node_span`` and ``Engine.stream_turn`` both wrap code
    that can raise (a model/provider error, a cancelled turn) inside
    ``with bind(...):``. The context must still unwind -- via ``bind``'s
    ``finally`` -- so a failed turn's bound fields never leak into whatever
    is logged next.
    """
    config = _make_config(tmp_path)
    logger = setup_logging(config)
    with pytest.raises(ValueError, match="boom"), bind(thread_id="t1"):
        logger.info("inside")
        raise ValueError("boom")
    logger.info("after")

    lines = _log_file(tmp_path).read_text().strip().splitlines()
    inside, after = (json.loads(line) for line in lines)
    assert inside["thread_id"] == "t1"
    assert "thread_id" not in after


@pytest.mark.parametrize("tz", ["Etc/GMT+12", "Pacific/Kiritimati"], ids=["utc-12", "utc+14"])
def test_log_filename_uses_utc_not_local_time(tmp_path, monkeypatch, tz: str) -> None:
    """The log file is named for the UTC date, matching the UTC ``ts`` on every
    record it contains.

    Regression guard. The filename was originally built from a naive
    ``datetime.now()``, so between 17:00 and midnight Pacific the process wrote
    UTC-stamped records into a file named for the previous day. The existing
    tests only caught it because they happened to run during those hours --
    they were green the rest of the day.

    These two offsets are 26 hours apart, so whatever the current instant is, at
    least one of them puts the local date on a different day from UTC. That
    makes this deterministic at any time of day rather than only near midnight.
    """
    import time

    monkeypatch.setenv("TZ", tz)
    time.tzset()
    try:
        setup_logging(_make_config(tmp_path)).info("hello")
        expected = f"cellsense-{datetime.now(UTC).strftime('%Y-%m-%d')}.jsonl"
        written = sorted(p.name for p in (tmp_path / "logs").iterdir())
        assert written == [expected], f"under TZ={tz}, expected {expected}, got {written}"
    finally:
        monkeypatch.undo()
        time.tzset()
