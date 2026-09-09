"""Structured logging: JSON lines to disk, a quiet stderr tap for warnings+.

The UI owns stdout entirely (rich renders the conversation there); this module
never writes to stdout, only to a rotating-by-date JSONL file under
``~/.cellsense/logs`` and, for anything WARNING or above (or everything, under
``--debug``), a plain-text line on stderr.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cellsense.config import Config

__all__ = ["LOGGER_NAME", "bind", "setup_logging"]

LOGGER_NAME = "cellsense"

_MAX_FIELD_LEN = 2000
# `token(?!s)` -- not just `token` -- so this does not also swallow the very
# fields observability.usage/graph.engine need to log: `input_tokens` and
# `output_tokens` are LLM usage *counts*, never a secret, but a bare `token`
# substring match would redact them purely because they end in "tokens".
# Every genuinely sensitive field observed in this codebase (`api_key`,
# `auth_token`, `access_token`, a bare `token`) still matches: none of them
# has a trailing "s" right after "token".
_SENSITIVE_KEY_PATTERN = re.compile(
    r"(key|token(?!s)|secret|password|authorization)", re.IGNORECASE
)

# Fields bound via bind() for the current call stack. A ContextVar rather than a
# plain dict/threading.local is deliberate: it is copy-on-write per logical task,
# so concurrent graph fan-out workers never see (or clobber) each other's bound
# fields, and nothing needs an explicit lock to read it. Note ContextVars do not
# propagate into a *new* thread automatically -- call bind() again inside each
# worker thread's own body with the fields you want attached there.
_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "cellsense_log_context", default=None
)


def _current_context() -> dict[str, Any]:
    return _context.get() or {}


# Computed once: the attribute names a bare LogRecord carries, so the formatter
# can tell "extra=" fields apart from LogRecord's own bookkeeping attributes.
_STANDARD_RECORD_KEYS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)


def _truncate(value: Any) -> Any:
    """Never let one field balloon a log line -- also the one line of defense
    against accidentally logging a full file's contents.
    """
    if isinstance(value, str) and len(value) > _MAX_FIELD_LEN:
        omitted = len(value) - _MAX_FIELD_LEN
        return f"{value[:_MAX_FIELD_LEN]}...<truncated {omitted} chars>"
    return value


def _sanitize(fields: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in fields.items():
        if _SENSITIVE_KEY_PATTERN.search(key):
            out[key] = "***redacted***"
        else:
            out[key] = _truncate(value)
    return out


class _JsonLinesFormatter(logging.Formatter):
    """One JSON object per line: timestamp, level, logger, message, bound context
    fields, any ``extra=`` fields on the record, and a truncated traceback.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": _truncate(record.getMessage()),
        }
        payload.update(_sanitize(_current_context()))

        # logging.Logger.info(msg, extra={...}) lands as individual attributes on
        # the record, not as a single dict -- pull out anything that isn't a
        # standard LogRecord attribute.
        extra_fields = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_RECORD_KEYS and key not in ("message", "asctime")
        }
        if extra_fields:
            payload.update(_sanitize(extra_fields))

        if record.exc_info:
            payload["exc_info"] = _truncate(self.formatException(record.exc_info))

        return json.dumps(payload, default=str, sort_keys=True)


def _level_from_name(name: str) -> int:
    return getattr(logging, name.upper(), logging.INFO)


def setup_logging(config: Config, *, debug: bool = False) -> logging.Logger:
    """Configure and return the ``"cellsense"`` logger.

    Idempotent-ish: calling it again replaces the handlers rather than stacking
    duplicates, so re-invoking after a config reload (e.g. ``/config reload`` in
    a future UI command) is safe.
    """
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    level = logging.DEBUG if debug else _level_from_name(config.telemetry.level)
    logger.setLevel(level)
    logger.propagate = False

    log_dir = Path(config.telemetry.log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    file_path = log_dir / f"cellsense-{datetime.now().strftime('%Y-%m-%d')}.jsonl"

    # FileHandler.emit() is protected by a per-handler lock, so concurrent writes
    # from graph fan-out worker threads interleave safely at the line level.
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setFormatter(_JsonLinesFormatter())
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.setLevel(logging.DEBUG if debug else logging.WARNING)
    stderr_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(stderr_handler)

    return logger


@contextmanager
def bind(**fields: Any) -> Iterator[None]:
    """Attach fields (``thread_id``, ``node``, ``subtask_id``, ...) to every log
    record emitted within this ``with`` block, merged with any already-bound
    fields in the current context.

    Usage in a graph node::

        with bind(thread_id=state.thread_id, node="agent"):
            logger.info("starting tool round")
    """
    merged = {**_current_context(), **fields}
    token = _context.set(merged)
    try:
        yield
    finally:
        _context.reset(token)
