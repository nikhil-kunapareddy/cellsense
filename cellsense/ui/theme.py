"""Single source of styling truth for :mod:`cellsense.ui`.

Every other module in this package looks up styles by *token name* through a
:class:`Theme` instance instead of writing literal colour strings. That keeps
``render.py``/``stream.py``/``slash.py`` renderer code readable (``theme.style
("tool.error")`` instead of ``"bold red"`` sprinkled everywhere) and means a new
palette -- or ``NO_COLOR`` compliance -- is a one-file change.

Why a token table rather than a handful of named colours: the renderers need
semantically distinct styles (a failed tool vs. a denied approval vs. a plain
warning) that may coincide today but should be free to diverge later without
touching call sites.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from rich.console import Console
from rich.theme import Theme as RichTheme

ThemeMode = Literal["auto", "light", "dark", "no-color"]

#: Every token a renderer in this package is allowed to ask for. Keeping this
#: list explicit means a typo'd token name reliably falls back to "" (default
#: style) instead of silently drifting -- see :meth:`Theme.style`.
TOKENS: tuple[str, ...] = (
    "banner.title",
    "banner.meta",
    "banner.rule",
    "banner.hint",
    "prompt",
    "prompt.marker",
    "muted",
    "answer.citation",
    "worker",
    "reasoning",
    "plan.title",
    "plan.item",
    "tool.name",
    "tool.result",
    "tool.error",
    "notice.info",
    "notice.warn",
    "notice.error",
    "status.text",
    "status.interrupt",
    "approval.title",
    "approval.arg",
    "error",
    "warning",
    "success",
    "table.header",
)

_DARK: dict[str, str] = {
    "banner.title": "bold cyan",
    "banner.meta": "dim",
    "banner.rule": "cyan dim",
    "banner.hint": "dim",
    "prompt": "bold green",
    "prompt.marker": "dim",
    "muted": "dim",
    "answer.citation": "dim",
    "worker": "dim",
    "reasoning": "dim italic",
    "plan.title": "bold",
    "plan.item": "",
    "tool.name": "bold cyan",
    "tool.result": "dim",
    "tool.error": "bold red",
    "notice.info": "dim",
    "notice.warn": "yellow",
    "notice.error": "bold red",
    "status.text": "dim",
    "status.interrupt": "dim",
    "approval.title": "bold yellow",
    "approval.arg": "dim",
    "error": "bold red",
    "warning": "yellow",
    "success": "green",
    "table.header": "bold",
}

_LIGHT: dict[str, str] = {
    **_DARK,
    "banner.title": "bold blue",
    "banner.rule": "blue dim",
    "prompt": "bold dark_green",
    "tool.name": "bold blue",
}

_NO_COLOR: dict[str, str] = dict.fromkeys(TOKENS, "")


@dataclass(frozen=True)
class Theme:
    """A resolved palette: a mode label plus a token -> rich style-string map.

    Frozen and immutable so it can be shared across threads (the live status
    line and the tool-trace renderer both read it concurrently during a turn).
    """

    mode: ThemeMode
    styles: dict[str, str]

    def style(self, token: str) -> str:
        """Look up a token, defaulting to the empty (unstyled) string.

        Unknown tokens degrade to no styling rather than raising, matching the
        "never raise on unexpected input" rule the rest of the UI follows.
        """
        return self.styles.get(token, "")

    def rich_theme(self) -> RichTheme:
        """Build a :class:`rich.theme.Theme` for ``Console(theme=...)``."""
        return RichTheme(self.styles)


def _auto_mode() -> Literal["light", "dark"]:
    """Best-effort light/dark guess from ``COLORFGBG`` (set by many terminals).

    The format is ``"<fg>;<bg>"`` in the 0-15 ANSI palette; background codes
    7 and 15 are light. Anything else -- including the common case of the
    variable being absent -- defaults to dark, which is the safer bet for a
    developer terminal.
    """
    colorfgbg = os.environ.get("COLORFGBG")
    if colorfgbg:
        parts = colorfgbg.split(";")
        if parts and parts[-1] in ("7", "15"):
            return "light"
    return "dark"


def resolve_theme(mode: ThemeMode = "auto") -> Theme:
    """Resolve a requested mode into a concrete :class:`Theme`.

    ``NO_COLOR`` (https://no-color.org) always wins regardless of ``mode``:
    any non-``None`` value forces the no-color palette. This is checked here,
    once, so every call site gets the same answer.
    """
    if os.environ.get("NO_COLOR") is not None:
        return Theme(mode="no-color", styles=dict(_NO_COLOR))
    if mode == "no-color":
        return Theme(mode="no-color", styles=dict(_NO_COLOR))
    if mode == "light":
        return Theme(mode="light", styles=dict(_LIGHT))
    if mode == "dark":
        return Theme(mode="dark", styles=dict(_DARK))
    resolved = _auto_mode()
    return Theme(mode=resolved, styles=dict(_LIGHT if resolved == "light" else _DARK))


def make_console(theme: Theme, *, file: object | None = None, record: bool = False) -> Console:
    """Build the one :class:`~rich.console.Console` a caller should print through.

    Centralised so every entry point (REPL, printer, tests) gets identical
    ``no_color``/``highlight`` behaviour tied to the resolved theme rather than
    re-deriving it ad hoc.
    """
    return Console(
        theme=theme.rich_theme(),
        no_color=theme.mode == "no-color",
        highlight=False,
        file=file,  # type: ignore[arg-type]
        record=record,
    )
