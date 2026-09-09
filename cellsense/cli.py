"""The ``cellsense`` command-line entry point.

This module is the wiring layer: it owns nothing about *how* questions get
answered (that's ``cellsense.graph.Engine``) or *how* the terminal looks
(that's ``cellsense.ui``) -- it resolves configuration, builds the workspace
and tool registry, constructs one ``Engine``, and hands it to either the
interactive REPL or the one-shot print renderer. See ``docs/ARCHITECTURE.md``
for the module boundaries this respects.

``main()`` -- the console-script target in ``pyproject.toml``
(``cellsense = "cellsense.cli:main"``) -- does one extra thing before handing
off to the Typer app: it rewrites ``sys.argv`` so that ``cellsense FILE -q Q``
works without a subcommand. Typer/Click has no clean way to make a group's
*default* action accept the same free-form positional/option mix as a real
subcommand while still routing ``cellsense models`` etc. correctly, so the
default action lives in a real subcommand named ``ask`` and ``main()`` simply
prepends ``"ask"`` to argv when the first token isn't a known subcommand name
or a help/version flag. This keeps all the actual argument parsing inside
ordinary Typer/Click code -- no custom ``click.Group`` subclass required.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import shlex
import sqlite3
import subprocess
import sys
import tomllib
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, cast

import typer
from dotenv import load_dotenv
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cellsense import __version__
from cellsense.config import Config, load_config, write_default_config
from cellsense.errors import CellSenseError, ConfigError, DataError
from cellsense.graph.checkpoint import delete_session, list_sessions, open_checkpointer
from cellsense.graph.engine import Engine
from cellsense.io.loaders import SUPPORTED_EXTENSIONS, load_workspace
from cellsense.io.schema import Workspace
from cellsense.observability.logging import LOGGER_NAME, setup_logging
from cellsense.providers.registry import PROVIDERS, available_providers, list_models
from cellsense.tools.registry import ToolRegistry
from cellsense.ui import (
    FileSummary,
    ModelChoice,
    PermissionsSummary,
    SheetSummary,
    make_console,
    resolve_theme,
    run_print,
    run_repl,
)
from cellsense.ui.repl import EngineLike
from cellsense.ui.theme import Theme, ThemeMode

__all__ = ["app", "main"]

# Subcommands `main()`'s argv rewrite must not shadow, plus the flags that
# should short-circuit straight to Click's own handling (top-level --help
# so it lists every subcommand, not just `ask`'s options).
_KNOWN_COMMANDS = frozenset({"ask", "models", "sessions", "init", "doctor"})
_PASSTHROUGH_TOKENS = frozenset({"--help", "-h"})

_DEFAULT_DB_PATH = Path.home() / ".cellsense" / "sessions.db"

app = typer.Typer(
    name="cellsense",
    help="Ask natural-language questions about your CSV/Excel files from the terminal.",
    add_completion=False,
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


# ── shared console/theme helpers ─────────────────────────────────────────────


def _err_console(theme_mode: ThemeMode) -> Console:
    return make_console(resolve_theme(theme_mode), file=sys.stderr)


def _load_env() -> None:
    """Load ``.env`` from the current directory into ``os.environ``.

    ``load_config`` does this itself as its first step (so the ``ask`` command
    never needs to call it separately), but ``models``/``doctor`` report on
    provider API keys without going through ``load_config`` at all, so they
    call this directly to see the same environment ``ask`` would.
    """
    load_dotenv(Path.cwd() / ".env")


def _render_error(console: Console, exc: CellSenseError) -> None:
    """Render a deliberate CellSense failure as a clean panel -- never a traceback."""
    lines: list[RenderableType] = [Text(exc.message, style="bold red")]
    if exc.hint:
        lines.append(Text(exc.hint, style="dim"))
    console.print(Panel(Group(*lines), title="Error", border_style="red", expand=False))


def _render_unexpected(console: Console, exc: Exception, *, log_path: Path, debug: bool) -> None:
    """Render an exception nobody deliberately raised: a short apology, the log
    path, and -- only under ``--debug`` -- the full traceback.
    """
    lines: list[RenderableType] = [
        Text("An unexpected error occurred.", style="bold red"),
        Text(f"Details were logged to: {log_path}", style="dim"),
    ]
    console.print(Panel(Group(*lines), title="Error", border_style="red", expand=False))
    if debug:
        console.print_exception(show_locals=False)


def _log_file_path(config: Config) -> Path:
    """Mirror ``observability.logging.setup_logging``'s file-naming formula so
    the top-level error handler can point at the right file without that
    module exposing the path itself.
    """
    log_dir = Path(config.telemetry.log_dir).expanduser()
    return log_dir / f"cellsense-{datetime.now().strftime('%Y-%m-%d')}.jsonl"


# ── config plumbing ───────────────────────────────────────────────────────────


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Generic recursive dict merge (``override`` wins). Deliberately reimplemented
    here rather than imported: ``config._deep_merge`` is a private helper of
    ``cellsense.config``, and this is a plain data-merge utility, not a second
    copy of the precedence *policy* itself (``load_config`` still owns that).
    """
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_extra_config_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Malformed TOML in {path}: {exc}") from exc


def _build_cli_overrides(
    *,
    model: str | None,
    max_tool_rounds: int | None,
    yes: bool,
    deny: bool,
    config_path: Path | None,
) -> dict[str, Any]:
    """Turn the subset of CLI flags that map onto ``Config`` fields into the
    nested-dict shape ``load_config`` expects, omitting any flag the user
    didn't actually pass so it never shadows a lower-precedence value.

    ``--config <path>`` is folded in here too: its contents are treated as
    sitting *below* individual flags in precedence (a flag always wins over
    the file it's layered on top of) but are still passed through
    ``load_config``'s own ``cli_overrides`` parameter -- and therefore its
    own key validation -- rather than read some other way.
    """
    if yes and deny:
        raise ConfigError("--yes and --deny are mutually exclusive.")

    overrides: dict[str, Any] = {}
    if model is not None:
        overrides.setdefault("model", {})["default"] = model
    if max_tool_rounds is not None:
        overrides.setdefault("agent", {})["max_tool_rounds"] = max_tool_rounds
    if yes:
        overrides.setdefault("permissions", {})["mode"] = "allow"
    elif deny:
        overrides.setdefault("permissions", {})["mode"] = "deny"

    if config_path is not None:
        file_data = _load_extra_config_file(config_path)
        overrides = _deep_merge(file_data, overrides)

    return overrides


# ── file validation ───────────────────────────────────────────────────────────


def _validate_paths(raw_paths: list[Path]) -> list[Path]:
    """Check every path up front so loading fails with one clear message
    instead of a pandas/OS-level error partway through.
    """
    resolved: list[Path] = []
    problems: list[str] = []
    for p in raw_paths:
        if not p.exists():
            problems.append(f"File not found: {p}")
        elif p.is_dir():
            problems.append(f"{p} is a directory, not a file.")
        elif p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            problems.append(
                f"Unsupported file type {p.suffix!r}: {p} "
                f"(supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))})"
            )
        else:
            resolved.append(p)
    if problems:
        raise DataError("Cannot load the given file(s).", hint="\n".join(problems))
    return resolved


def _workspace_to_file_summaries(workspace: Workspace) -> list[FileSummary]:
    """Project ``Workspace`` onto the UI's own ``FileSummary``/``SheetSummary``
    shapes -- ``cellsense.ui`` deliberately never imports ``cellsense.io``
    (see ``ui/repl.py``'s module docstring), so this mapping lives here.
    """
    summaries: list[FileSummary] = []
    for fd in workspace.files.values():
        sheets = tuple(
            SheetSummary(
                name=None if fd.file_type == "csv" else name,
                rows=len(df),
                cols=len(df.columns),
                columns=tuple(str(c) for c in df.columns),
            )
            for name, df in fd.sheets.items()
        )
        summaries.append(FileSummary(filename=fd.filename, file_type=fd.file_type, sheets=sheets))
    return summaries


# ── session id resolution ─────────────────────────────────────────────────────


def _resolve_thread_id(engine: Engine, *, resume: str | None, session: str | None) -> str:
    if resume and session:
        raise ConfigError("--resume and --session are mutually exclusive.")
    if resume:
        known = {s.thread_id for s in engine.sessions()}
        if resume not in known:
            raise ConfigError(
                f"No saved session {resume!r}.",
                hint="Run `cellsense sessions` to list saved session ids.",
            )
        return resume
    if session:
        return session
    return uuid.uuid4().hex


def _read_stdin_question() -> str:
    if sys.stdin.isatty():
        raise ConfigError(
            "--print requires a question.",
            hint='Pass -q/--query "..." or pipe a question on stdin.',
        )
    question = sys.stdin.read().strip()
    if not question:
        raise ConfigError("No question was provided on stdin.")
    return question


# ── REPL callback wiring ──────────────────────────────────────────────────────


class _EngineProxy:
    """A stable ``EngineLike`` handle that can swap its underlying ``Engine``.

    ``run_repl`` is handed one engine reference for the life of the loop, but
    ``/model <selector>`` needs to change which provider/model backs the
    session -- the ``Engine`` API has no setter for that (ARCHITECTURE.md §7
    fixes the model at construction), so switching models means building a
    *new* ``Engine`` and swapping it in behind this proxy rather than
    mutating one in place.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def replace(self, engine: Engine) -> None:
        self._engine = engine

    @property
    def model_label(self) -> str:
        return self._engine.model_label

    @property
    def provider_label(self) -> str:
        return self._engine.provider_label

    def stream_turn(self, question: str, **kwargs: Any) -> Any:
        return self._engine.stream_turn(question, **kwargs)

    def sessions(self) -> Any:
        return self._engine.sessions()

    def history(self, thread_id: str) -> list[tuple[str, str]]:
        return self._engine.history(thread_id)


def _make_on_attach(
    workspace: Workspace, file_summaries: list[FileSummary], console: Console, theme: Theme
) -> Any:
    def _attach(path: str) -> None:
        try:
            fd = workspace.add(Path(path))
        except CellSenseError as exc:
            _render_error(console, exc)
            return
        sheets = tuple(
            SheetSummary(
                name=None if fd.file_type == "csv" else name,
                rows=len(df),
                cols=len(df.columns),
                columns=tuple(str(c) for c in df.columns),
            )
            for name, df in fd.sheets.items()
        )
        file_summaries.append(
            FileSummary(filename=fd.filename, file_type=fd.file_type, sheets=sheets)
        )
        console.print(
            Text(
                f"Attached {fd.filename} ({fd.total_rows:,} rows)",
                style=theme.style("success"),
            )
        )

    return _attach


def _make_on_model_change(
    proxy: _EngineProxy,
    workspace: Workspace,
    config: Config,
    registry: ToolRegistry,
    console: Console,
    theme: Theme,
) -> Any:
    def _switch(selector: str) -> None:
        new_config = deepcopy(config)
        new_config.model.default = selector
        try:
            new_engine = Engine(workspace, new_config, registry=registry)
        except CellSenseError as exc:
            _render_error(console, exc)
            return
        proxy.replace(new_engine)
        config.model.default = selector
        console.print(
            Text(
                f"Switched to {new_engine.provider_label} · {new_engine.model_label}",
                style=theme.style("success"),
            )
        )

    return _switch


def _list_models_callback() -> list[ModelChoice]:
    available = set(available_providers())
    return [
        ModelChoice(
            selector=f"{spec.provider}:{spec.id}",
            provider=spec.provider,
            has_key=spec.provider in available,
        )
        for spec in list_models()
    ]


# ── the `ask` command: the default action ────────────────────────────────────


def _version_callback(value: bool) -> None:
    if value:
        Console().print(f"cellsense {__version__}")
        raise typer.Exit()


@app.command("ask", help="Ask a question about your files (the default action).")
def ask(
    files: Annotated[
        list[Path] | None,
        typer.Argument(help="CSV/XLSX files to load. Omit to let the agent find files itself."),
    ] = None,
    query: Annotated[
        str | None, typer.Option("--query", "-q", help="Ask one question and exit.")
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model", "-m", help="Provider:model selector, e.g. groq:openai/gpt-oss-120b."
        ),
    ] = None,
    print_mode: Annotated[
        bool,
        typer.Option(
            "--print",
            "-p",
            help="Non-interactive mode; reads the question from stdin if -q is absent.",
        ),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show the tool trace in --print mode.")
    ] = False,
    debug: Annotated[
        bool, typer.Option("--debug", help="Debug-level logging and full tracebacks on error.")
    ] = False,
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable ANSI styling.")] = False,
    yes: Annotated[
        bool, typer.Option("--yes", help="Auto-approve side-effect tools (permissions.mode=allow).")
    ] = False,
    deny: Annotated[
        bool, typer.Option("--deny", help="Auto-deny side-effect tools (permissions.mode=deny).")
    ] = False,
    resume: Annotated[
        str | None, typer.Option("--resume", help="Resume a saved session by thread id.")
    ] = None,
    session: Annotated[
        str | None, typer.Option("--session", help="Use a specific thread id for this run.")
    ] = None,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Load an additional TOML config file.")
    ] = None,
    max_tool_rounds: Annotated[
        int | None, typer.Option("--max-tool-rounds", help="Override agent.max_tool_rounds.")
    ] = None,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Print the version and exit.",
        ),
    ] = False,
) -> None:
    theme_mode: ThemeMode = "no-color" if no_color else "auto"
    err_console = _err_console(theme_mode)

    try:
        cli_overrides = _build_cli_overrides(
            model=model,
            max_tool_rounds=max_tool_rounds,
            yes=yes,
            deny=deny,
            config_path=config_path,
        )
        config = load_config(cli_overrides, cwd=Path.cwd())
    except CellSenseError as exc:
        _render_error(err_console, exc)
        raise typer.Exit(code=exc.exit_code) from None

    setup_logging(config, debug=debug)
    logger = logging.getLogger(LOGGER_NAME)
    # Resolved settings only -- model selector and permission mode are never
    # secrets (the API key itself lives in the environment, not in `config`,
    # and is never read here); file_count is a count, not the paths/contents.
    logger.info(
        "ask invoked",
        extra={
            "model": config.model.default,
            "permission_mode": config.permissions.mode,
            "file_count": len(files or []),
            "debug": debug,
        },
    )

    try:
        paths = _validate_paths(files or [])
        workspace = load_workspace(paths) if paths else Workspace()
        if workspace.files:
            logger.info(
                "workspace loaded",
                extra={
                    "filenames": [fd.filename for fd in workspace.files.values()],
                    "total_rows": sum(fd.total_rows for fd in workspace.files.values()),
                    "total_cols": sum(
                        len(df.columns)
                        for fd in workspace.files.values()
                        for df in fd.sheets.values()
                    ),
                },
            )
        else:
            logger.info("workspace loaded", extra={"file_count": 0})
        registry = ToolRegistry()
        engine = Engine(workspace, config, registry=registry)
        thread_id = _resolve_thread_id(engine, resume=resume, session=session)

        # cast: Engine structurally satisfies EngineLike at runtime (verified below
        # in the reconciliation report) -- mypy can't see it because
        # graph.checkpoint.SessionInfo (Engine.sessions()'s real return type) and
        # ui.repl.SessionInfo (the Protocol EngineLike declares) are two distinct
        # types with overlapping but non-identical fields. See the discrepancy
        # noted in the task report; not something cli.py can fix without editing
        # ui/repl.py or graph/checkpoint.py, which are out of scope here.
        engine_like = cast(EngineLike, engine)

        if query is not None:
            exit_code = run_print(
                engine_like, query, thread_id=thread_id, verbose=verbose, theme_mode=theme_mode
            )
            logger.info("ask exited", extra={"exit_code": exit_code, "thread_id": thread_id})
            raise typer.Exit(code=exit_code)

        if print_mode:
            question = _read_stdin_question()
            exit_code = run_print(
                engine_like,
                question,
                thread_id=thread_id,
                verbose=verbose,
                theme_mode=theme_mode,
            )
            logger.info("ask exited", extra={"exit_code": exit_code, "thread_id": thread_id})
            raise typer.Exit(code=exit_code)

        theme = resolve_theme(theme_mode)
        console = make_console(theme)
        proxy = _EngineProxy(engine)
        file_summaries = _workspace_to_file_summaries(workspace)

        run_repl(
            proxy,
            files=file_summaries,
            thread_id=thread_id,
            theme_mode=theme_mode,
            show_tool_args=config.ui.show_tool_args or verbose,
            on_attach=_make_on_attach(workspace, file_summaries, console, theme),
            on_model_change=_make_on_model_change(
                proxy, workspace, config, registry, console, theme
            ),
            list_models=_list_models_callback,
            permissions_summary=lambda: PermissionsSummary(
                mode=config.permissions.mode, allow=tuple(config.permissions.allow)
            ),
            console=console,
        )
        logger.info("ask exited", extra={"exit_code": 0, "thread_id": thread_id})
    except typer.Exit:
        raise
    except CellSenseError as exc:
        logger.warning(
            "ask failed",
            extra={"error_type": type(exc).__name__, "exit_code": exc.exit_code},
        )
        _render_error(err_console, exc)
        raise typer.Exit(code=exc.exit_code) from None
    except Exception as exc:  # last resort: never let a raw traceback reach the user
        logger.exception("Unhandled error in `cellsense ask`")
        _render_unexpected(err_console, exc, log_path=_log_file_path(config), debug=debug)
        raise typer.Exit(code=1) from None


# ── `cellsense models` ────────────────────────────────────────────────────────


@app.command(
    "models", help="List providers/models: id, context window, pricing, and API key status."
)
def models_cmd() -> None:
    _load_env()
    console = make_console(resolve_theme("auto"))
    available = set(available_providers())
    table = Table(title="Providers & models", header_style="bold")
    table.add_column("Selector")
    table.add_column("Context")
    table.add_column("Input $/Mtok")
    table.add_column("Output $/Mtok")
    table.add_column("Key")
    for spec in list_models():
        context = f"{spec.context_window:,}" if spec.context_window else "-"
        input_price = (
            f"{spec.input_usd_per_mtok:.2f}" if spec.input_usd_per_mtok is not None else "-"
        )
        output_price = (
            f"{spec.output_usd_per_mtok:.2f}" if spec.output_usd_per_mtok is not None else "-"
        )
        has_key = spec.provider in available
        key_status = "[green]present[/green]" if has_key else "[dim]missing[/dim]"
        table.add_row(f"{spec.provider}:{spec.id}", context, input_price, output_price, key_status)
    console.print(table)
    console.print(
        Text(
            "Selector syntax: provider:model_id, a bare provider (uses its default), "
            "or a bare model id if it's unambiguous. Unrecognized ids still work -- "
            "the table above is metadata, not a whitelist.",
            style="dim",
        )
    )


# ── `cellsense sessions` [rm] ─────────────────────────────────────────────────

sessions_app = typer.Typer(help="List or manage saved sessions.", add_completion=False)
app.add_typer(sessions_app, name="sessions", invoke_without_command=True)


def _render_sessions_table(console: Console) -> None:
    saver = open_checkpointer(_DEFAULT_DB_PATH)
    infos = list_sessions(saver)
    if not infos:
        console.print(Text("No saved sessions.", style="dim"))
        return
    table = Table(header_style="bold")
    table.add_column("Thread")
    table.add_column("Created")
    table.add_column("Updated")
    table.add_column("Messages")
    table.add_column("Last question")
    for info in infos:
        table.add_row(
            info.thread_id,
            info.created_at,
            info.updated_at,
            str(info.message_count),
            info.question_preview,
        )
    console.print(table)


@sessions_app.callback(invoke_without_command=True)
def sessions_main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is not None:
        return
    console = make_console(resolve_theme("auto"))
    try:
        _render_sessions_table(console)
    except CellSenseError as exc:
        _render_error(console, exc)
        raise typer.Exit(code=exc.exit_code) from None


@sessions_app.command("rm")
def sessions_rm(thread_id: Annotated[str, typer.Argument(help="Session id to delete.")]) -> None:
    console = make_console(resolve_theme("auto"))
    try:
        saver = open_checkpointer(_DEFAULT_DB_PATH)
        known = {info.thread_id for info in list_sessions(saver)}
        if thread_id not in known:
            raise ConfigError(
                f"No saved session {thread_id!r}.",
                hint="Run `cellsense sessions` to list saved session ids.",
            )
        delete_session(saver, thread_id)
    except CellSenseError as exc:
        _render_error(console, exc)
        raise typer.Exit(code=exc.exit_code) from None
    console.print(Text(f"Deleted session {thread_id}", style="green"))


# ── `cellsense init` ──────────────────────────────────────────────────────────


@app.command("init", help="Write a starter .cellsense/config.toml in the current directory.")
def init_cmd(
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing config file.")
    ] = False,
) -> None:
    console = make_console(resolve_theme("auto"))
    path = Path.cwd() / ".cellsense" / "config.toml"
    try:
        if path.exists() and not force:
            raise ConfigError(f"{path} already exists.", hint="Pass --force to overwrite it.")
        write_default_config(path)
    except CellSenseError as exc:
        _render_error(console, exc)
        raise typer.Exit(code=exc.exit_code) from None
    console.print(Text(f"Wrote {path}", style="green"))


# ── `cellsense doctor` ────────────────────────────────────────────────────────


def _mask_key(raw: str) -> str:
    prefix = raw[:4] if len(raw) > 4 else "*" * len(raw)
    return f"{prefix}…  ({len(raw)} chars)"


def _check_checkpoint_db(path: Path) -> tuple[bool, str]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return False, f"cannot create {path.parent}: {exc}"
    if not os.access(path.parent, os.W_OK):
        return False, f"{path.parent} is not writable"
    if path.exists():
        try:
            conn = sqlite3.connect(str(path))
            try:
                conn.execute("PRAGMA quick_check")
            finally:
                conn.close()
        except sqlite3.Error as exc:
            return False, f"{path} exists but failed a quick_check: {exc}"
        return True, f"{path} exists and is writable"
    return True, f"{path.parent} is writable (DB will be created on first use)"


def _dataless_check(timeout_s: float = 30.0) -> tuple[bool, str]:
    """Scan the *active* ``sys.prefix`` (never ``./.venv``) for iCloud-evicted
    "dataless" placeholder files -- see docs/DEVELOPMENT.md for exactly why
    this matters and how it was diagnosed. Uses ``stat`` metadata only, which
    never triggers the re-download hang that reading file *contents* does.
    """
    candidates = sorted(Path(sys.prefix).glob("lib/python*/site-packages"))
    if not candidates:
        return True, f"No site-packages directory found under {sys.prefix} (skipped)."
    target = candidates[0]
    command = (
        f"find {shlex.quote(str(target))} -type f -print0 "
        f"| xargs -0 stat -f '%Sf %N' 2>/dev/null | grep -c dataless"
    )
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except subprocess.TimeoutExpired:
        return False, (
            f"Scan of {target} timed out after {timeout_s:.0f}s -- treat this as a likely "
            "dataless-file trap. See docs/DEVELOPMENT.md."
        )
    raw_count = (result.stdout or "").strip()
    count = int(raw_count) if raw_count.isdigit() else 0
    if count > 0:
        return False, f"{count} dataless file(s) found under {target}. See docs/DEVELOPMENT.md."
    return True, f"No dataless files found under {target}."


@app.command(
    "doctor",
    help=(
        "Diagnose the local environment: Python version, provider keys/packages, "
        "checkpoint DB writability, and the iCloud dataless-file trap."
    ),
)
def doctor_cmd() -> None:
    _load_env()
    console = make_console(resolve_theme("auto"))
    console.print(Text("CellSense doctor", style="bold"))
    console.print()

    console.print(f"Python: {sys.version.split()[0]}  ({sys.executable})")

    try:
        config = load_config({}, cwd=Path.cwd())
        console.print("Config: [green]resolved OK[/green]")
    except CellSenseError as exc:
        config = None
        console.print(f"Config: [red]{exc.message}[/red]")
        if exc.hint:
            console.print(f"  [dim]{exc.hint}[/dim]")

    console.print()
    console.print(Text("Provider API keys", style="bold"))
    key_table = Table(box=None, pad_edge=False, show_header=True, header_style="bold")
    key_table.add_column("Provider")
    key_table.add_column("Env var")
    key_table.add_column("Status")
    for spec in PROVIDERS.values():
        raw = os.environ.get(spec.env_key)
        status = f"[green]set[/green]  {_mask_key(raw)}" if raw else "[dim]not set[/dim]"
        key_table.add_row(spec.label, spec.env_key, status)
    console.print(key_table)

    console.print()
    console.print(Text("Provider packages", style="bold"))
    pkg_table = Table(box=None, pad_edge=False, show_header=True, header_style="bold")
    pkg_table.add_column("Provider")
    pkg_table.add_column("Package")
    pkg_table.add_column("Status")
    for spec in PROVIDERS.values():
        installed = importlib.util.find_spec(spec.package) is not None
        status = (
            "[green]importable[/green]"
            if installed
            else f"[yellow]missing[/yellow]  pip install 'cellsense[{spec.extra}]'"
        )
        pkg_table.add_row(spec.label, spec.package, status)
    console.print(pkg_table)

    console.print()
    db_ok, db_detail = _check_checkpoint_db(_DEFAULT_DB_PATH)
    db_style = "green" if db_ok else "red"
    console.print(f"Checkpoint DB: [{db_style}]{db_detail}[/{db_style}]")

    console.print()
    dataless_ok, dataless_detail = _dataless_check()
    dataless_style = "green" if dataless_ok else "red"
    console.print(f"Dataless-file check: [{dataless_style}]{dataless_detail}[/{dataless_style}]")

    if config is not None:
        console.print()
        console.print(f"Logs: {_log_file_path(config)}")


# ── process entry point ──────────────────────────────────────────────────────


def main() -> None:
    """Console-script entry point (``pyproject.toml``: ``cellsense.cli:main``).

    Rewrites argv so the bare invocation ``cellsense [FILES]... [-q Q]`` -- with
    no subcommand -- routes to the ``ask`` command. See the module docstring
    for why this is a plain argv rewrite rather than a custom Click group.
    """
    argv = sys.argv[1:]
    if not argv:
        argv = ["ask"]
    elif argv[0] not in _KNOWN_COMMANDS and argv[0] not in _PASSTHROUGH_TOKENS:
        argv = ["ask", *argv]
    app(args=argv, prog_name="cellsense")


if __name__ == "__main__":
    main()
