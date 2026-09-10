"""Drive one engine turn: threaded consumption + a live status line.

This module owns the concurrency for a turn:

* the engine iterator (``Engine.stream_turn``) is pumped on a worker thread,
  because iterating it *is* the blocking work (model calls, tool execution);
* the main thread renders an animated ``rich.live.Live`` status line --
  spinner, elapsed time, running token count, cost -- while permanent output
  (tool trace, plan, notices, the final answer) is printed to normal
  scrollback *above* that transient line, exactly like Claude Code;
* Esc/Ctrl-C during a turn set a ``threading.Event`` that the engine contract
  (docs/ARCHITECTURE.md §7) guarantees it will notice at the next node
  boundary, terminating the iterator with ``TurnFailed(cancelled=True)``.

Approval prompts are handled here too (not in ``repl.py``): only this module
holds the live reference that needs pausing while the human answers, so it
builds the default ``on_approval`` passed to ``Engine.stream_turn`` itself.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Literal

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

from cellsense.events import (
    ApprovalRequested,
    Event,
    TextDelta,
    TurnFailed,
    TurnFinished,
    UsageUpdated,
)
from cellsense.ui.render import render_event
from cellsense.ui.theme import Theme

if TYPE_CHECKING:  # pragma: no cover - typing only, see the hard import rule below.
    # cellsense.ui must not import cellsense.graph at runtime; EngineLike is a
    # typing.Protocol defined in repl.py that captures the Engine surface we use.
    from cellsense.ui.repl import EngineLike, OnApproval

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_STATUS_VERBS = ("Analyzing", "Thinking", "Crunching numbers", "Synthesizing")
_SECONDS_PER_VERB = 4.0

_SENTINEL = object()


def _fmt_tokens(n: int) -> str:
    if n >= 1000:
        return f"{n / 1000:.1f}k tok"
    return f"{n} tok"


def _fmt_cost(cost_usd: float | None) -> str:
    return f"${cost_usd:.4f}" if cost_usd is not None else "$-"


def _status_line(
    theme: Theme,
    *,
    frame: int,
    elapsed_s: float,
    tokens: int,
    cost_usd: float | None,
    cancelling: bool,
) -> Text:
    spinner = _SPINNER_FRAMES[frame % len(_SPINNER_FRAMES)]
    if cancelling:
        text = f"{spinner} Cancelling… ({elapsed_s:.0f}s)"
        return Text(text, style=theme.style("status.interrupt"))
    verb = _STATUS_VERBS[int(elapsed_s // _SECONDS_PER_VERB) % len(_STATUS_VERBS)]
    return Text(
        f"{spinner} {verb}… ({elapsed_s:.0f}s · {_fmt_tokens(tokens)} · "
        f"{_fmt_cost(cost_usd)} · esc to interrupt)",
        style=theme.style("status.text"),
    )


def _live_region(answer_text: str, status: Text) -> RenderableType:
    if answer_text:
        return Group(Markdown(answer_text), Text(""), status)
    return status


def _stdin_is_tty() -> bool:
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


class _EscWatcher:
    """Best-effort background reader that turns a raw Esc/Ctrl-C byte into ``cancel.set()``.

    Ctrl-C is already handled portably via Python's normal ``KeyboardInterrupt``
    delivery (see ``run_turn``'s poll loop); this class exists purely for Esc,
    which is just an ordinary byte and never raises anything on its own. It
    requires POSIX ``termios``/``tty``/``select`` and a real TTY; construction
    is skipped entirely otherwise (piped/test input), which is the graceful
    non-TTY degradation the spec asks for.
    """

    def __init__(self, cancel: threading.Event) -> None:
        self._cancel = cancel
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._fd = sys.stdin.fileno()
        self._old_settings: Any | None = None

    def start(self) -> None:
        try:
            import termios
            import tty

            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except Exception:
            self._old_settings = None
        self._thread.start()

    def _run(self) -> None:
        import select

        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([self._fd], [], [], 0.05)
            except (OSError, ValueError):
                return
            if not ready:
                continue
            try:
                chunk = os.read(self._fd, 1)
            except OSError:
                return
            if chunk in (b"\x1b", b"\x03"):
                self._cancel.set()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.2)
        if self._old_settings is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)


def _default_on_approval(
    console: Console,
    theme: Theme,
    live_holder: list[Live | None],
    approval_lock: threading.Lock,
) -> OnApproval:
    """Build the ``on_approval`` passed to ``Engine.stream_turn`` when the
    caller (repl.py) doesn't supply one: pause the live spinner, print the
    request, read a line, resume.

    Runs on the *pump* thread (the engine calls it synchronously while
    iterating) -- guarded by ``approval_lock`` so it can't race the main
    thread's ``live.update()`` calls.
    """

    def _ask(request: ApprovalRequested) -> Literal["allow", "allow_always", "deny"]:
        with approval_lock:
            live = live_holder[0]
            if live is not None:
                live.stop()
            try:
                renderable = render_event(request, theme, show_tool_args=True)
                if renderable is not None:
                    console.print(renderable)
                choice = console.input("Approve? [y]es / [a]lways / [N]o: ").strip().lower()
            finally:
                if live is not None:
                    live.start(refresh=True)
        if choice in ("y", "yes"):
            return "allow"
        if choice in ("a", "always"):
            return "allow_always"
        return "deny"

    return _ask


def run_turn(
    engine: EngineLike,
    question: str,
    *,
    thread_id: str,
    console: Console,
    theme: Theme,
    show_tool_args: bool = True,
    on_approval: OnApproval | None = None,
    cancel: threading.Event | None = None,
) -> Event:
    """Run one turn to completion and return its terminal event.

    Consumes ``engine.stream_turn(...)`` on a worker thread. While it runs,
    the main thread renders permanent output (tool trace, plan, notices)
    directly to ``console`` -- which coexists correctly with an active
    ``Live`` region, see rich's docs on printing during ``Live`` -- and keeps
    a transient status line (spinner/elapsed/tokens/cost) plus the
    in-progress answer text inside that ``Live`` region. On
    ``TurnFinished``/``TurnFailed`` the region is cleared and the final,
    permanent rendering is printed once.

    ``cancel`` may be supplied by the caller (tests do this to trigger
    cancellation deterministically without simulating keystrokes); if omitted
    one is created. Setting it -- from any thread -- stops the turn; the
    process and the REPL are never killed.
    """
    cancel = cancel if cancel is not None else threading.Event()
    live_holder: list[Live | None] = [None]
    approval_lock = threading.Lock()
    approval = on_approval or _default_on_approval(console, theme, live_holder, approval_lock)

    events = engine.stream_turn(question, thread_id=thread_id, on_approval=approval, cancel=cancel)

    event_queue: queue.Queue[object] = queue.Queue()

    def _pump() -> None:
        try:
            for item in events:
                event_queue.put(item)
                if isinstance(item, (TurnFinished, TurnFailed)):
                    break
        except Exception as exc:
            event_queue.put(TurnFailed(message=f"Internal error: {exc}"))
        finally:
            # Breaking out of the loop above leaves the generator suspended
            # inside `Engine.stream_turn`'s `with bind(...)` block. Close it
            # here so that block unwinds on *this* thread, in the same Context
            # its ContextVar token was created in. Left to the garbage
            # collector instead, it gets finalized on whatever thread happens
            # to collect, and the mismatched `reset()` raises
            # "ValueError: <Token ...> was created in a different Context",
            # printed as an ignored exception some time after the answer.
            close = getattr(events, "close", None)
            if callable(close):
                with suppress(Exception):  # cleanup must never mask a turn
                    close()
            event_queue.put(_SENTINEL)

    pump = threading.Thread(target=_pump, daemon=True)
    pump.start()

    interactive = console.is_terminal
    esc_watcher = _EscWatcher(cancel) if interactive and _stdin_is_tty() else None
    if esc_watcher is not None:
        esc_watcher.start()

    terminal_event: Event = TurnFailed(message="turn ended without a terminal event")
    start = time.monotonic()
    answer_text = ""
    input_tokens = output_tokens = 0
    cost_usd: float | None = None
    frame = 0

    try:
        if interactive:
            with Live(console=console, transient=True, refresh_per_second=10) as live:
                live_holder[0] = live
                while True:
                    try:
                        item = event_queue.get(timeout=0.08)
                    except queue.Empty:
                        item = None
                    except KeyboardInterrupt:
                        cancel.set()
                        continue

                    if item is _SENTINEL:
                        break
                    if item is not None:
                        assert isinstance(item, Event)
                        if isinstance(item, TextDelta) and item.channel == "answer":
                            answer_text += item.text
                        elif isinstance(item, UsageUpdated):
                            input_tokens, output_tokens = item.input_tokens, item.output_tokens
                            cost_usd = item.cost_usd
                        elif isinstance(item, (TurnFinished, TurnFailed)):
                            terminal_event = item
                            if isinstance(item, TurnFinished):
                                input_tokens, output_tokens = item.input_tokens, item.output_tokens
                                cost_usd = item.cost_usd
                        else:
                            renderable = render_event(item, theme, show_tool_args=show_tool_args)
                            if renderable is not None:
                                console.print(renderable)

                    frame += 1
                    elapsed = time.monotonic() - start
                    status = _status_line(
                        theme,
                        frame=frame,
                        elapsed_s=elapsed,
                        tokens=input_tokens + output_tokens,
                        cost_usd=cost_usd,
                        cancelling=cancel.is_set(),
                    )
                    live.update(_live_region(answer_text, status))
        else:
            # Non-TTY: no animation, no Esc watcher -- plain, synchronous-looking
            # output as each permanent event arrives. Ctrl-C still cancels.
            while True:
                try:
                    item = event_queue.get(timeout=0.08)
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    cancel.set()
                    continue

                if item is _SENTINEL:
                    break
                assert isinstance(item, Event)
                if isinstance(item, TextDelta) and item.channel == "answer":
                    answer_text += item.text
                elif isinstance(item, UsageUpdated):
                    input_tokens, output_tokens = item.input_tokens, item.output_tokens
                    cost_usd = item.cost_usd
                elif isinstance(item, (TurnFinished, TurnFailed)):
                    terminal_event = item
                    if isinstance(item, TurnFinished):
                        input_tokens, output_tokens = item.input_tokens, item.output_tokens
                        cost_usd = item.cost_usd
                else:
                    renderable = render_event(item, theme, show_tool_args=show_tool_args)
                    if renderable is not None:
                        console.print(renderable)
    finally:
        if esc_watcher is not None:
            esc_watcher.stop()
        pump.join(timeout=1.0)

    final_renderable = render_event(terminal_event, theme, show_tool_args=show_tool_args)
    if final_renderable is not None:
        console.print(final_renderable)
    return terminal_event
