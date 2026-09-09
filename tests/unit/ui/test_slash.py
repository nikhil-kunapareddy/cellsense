"""Pins cellsense.ui.slash: the command table's invariants, dispatch via
execute()/find_command(), and the completion menu.

Test doubles for SlashContext's structural collaborators (engine/session/
files) are plain SimpleNamespace objects built here, not imports from
cellsense.ui.repl -- slash.py only relies on getattr-based duck typing (see
its own docstrings), so there is no need to depend on repl.py's concrete
dataclasses to exercise it.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

from prompt_toolkit.document import Document
from rich.console import Console

from cellsense.ui.slash import (
    SLASH_TABLE,
    SlashCompleter,
    SlashContext,
    SlashOutcome,
    execute,
    find_command,
)
from cellsense.ui.theme import resolve_theme

THEME = resolve_theme("no-color")


def _console() -> Console:
    return Console(file=io.StringIO(), width=200, no_color=True, highlight=False)


def _ctx(**overrides) -> SlashContext:
    defaults = {
        "args": [],
        "console": _console(),
        "theme": THEME,
        "engine": SimpleNamespace(sessions=lambda: []),
        "session": SimpleNamespace(
            thread_id="t1",
            turns=0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost_usd=0.0,
            transcript=[],
        ),
        "files": (),
        "list_models": None,
        "permissions_summary": None,
    }
    defaults.update(overrides)
    return SlashContext(**defaults)


def _output(ctx: SlashContext) -> str:
    return ctx.console.file.getvalue()


class TestSlashTableInvariants:
    def test_command_names_are_unique(self) -> None:
        names = [cmd.name for cmd in SLASH_TABLE]
        assert len(names) == len(set(names))

    def test_names_and_aliases_never_collide(self) -> None:
        tokens = []
        for cmd in SLASH_TABLE:
            tokens.append(cmd.name)
            tokens.extend(cmd.aliases)
        assert len(tokens) == len(set(tokens))

    def test_every_handler_is_callable(self) -> None:
        for cmd in SLASH_TABLE:
            assert callable(cmd.handler)

    def test_every_command_starts_with_a_slash(self) -> None:
        for cmd in SLASH_TABLE:
            assert cmd.name.startswith("/")


class TestFindCommand:
    def test_finds_by_exact_name(self) -> None:
        assert find_command("/help") is not None
        assert find_command("/help").name == "/help"

    def test_finds_without_leading_slash(self) -> None:
        assert find_command("help").name == "/help"

    def test_is_case_insensitive(self) -> None:
        assert find_command("/HELP").name == "/help"

    def test_finds_by_alias(self) -> None:
        assert find_command("/quit").name == "/exit"

    def test_unknown_command_returns_none(self) -> None:
        assert find_command("/nope") is None


class TestExecuteDispatch:
    def test_empty_line_is_a_no_op(self) -> None:
        ctx = _ctx()
        outcome = execute("", ctx)
        assert outcome == SlashOutcome()
        assert _output(ctx) == ""

    def test_unknown_command_prints_a_hint_and_does_not_exit(self) -> None:
        ctx = _ctx()
        outcome = execute("/bogus", ctx)
        assert "Unknown command" in _output(ctx)
        assert outcome.exit_repl is False

    def test_args_are_split_and_passed_to_the_handler(self) -> None:
        ctx = _ctx()
        outcome = execute("/model claude:foo", ctx)
        assert outcome.set_model == "claude:foo"

    def test_exit_command_sets_exit_repl(self) -> None:
        ctx = _ctx()
        outcome = execute("/exit", ctx)
        assert outcome.exit_repl is True

    def test_quit_alias_also_exits(self) -> None:
        ctx = _ctx()
        outcome = execute("/quit", ctx)
        assert outcome.exit_repl is True

    def test_clear_command_clears_and_redraws(self) -> None:
        ctx = _ctx()
        outcome = execute("/clear", ctx)
        assert outcome.clear_screen is True
        assert outcome.redraw_banner is True


class TestCostCommand:
    def test_reports_turns_tokens_and_cost(self) -> None:
        ctx = _ctx(
            session=SimpleNamespace(
                thread_id="t1",
                turns=3,
                total_input_tokens=100,
                total_output_tokens=50,
                total_cost_usd=0.1234,
                transcript=[],
            )
        )
        execute("/cost", ctx)
        out = _output(ctx)
        assert "3 turn(s)" in out
        assert "150" in out
        assert "0.1234" in out


class TestFilesCommand:
    def test_no_files_prints_hint(self) -> None:
        ctx = _ctx(files=())
        execute("/files", ctx)
        assert "No files loaded" in _output(ctx)

    def test_lists_filename_sheet_rows_cols_and_columns(self) -> None:
        sheet = SimpleNamespace(name="Q1", rows=10, cols=2, columns=("region", "revenue"))
        file_summary = SimpleNamespace(filename="sales.xlsx", sheets=(sheet,))
        ctx = _ctx(files=(file_summary,))
        execute("/files", ctx)
        out = _output(ctx)
        assert "sales.xlsx[Q1]" in out
        assert "10" in out and "region" in out and "revenue" in out

    def test_csv_sheet_with_no_name_omits_bracket_clause(self) -> None:
        sheet = SimpleNamespace(name=None, rows=5, cols=1, columns=("a",))
        file_summary = SimpleNamespace(filename="data.csv", sheets=(sheet,))
        ctx = _ctx(files=(file_summary,))
        execute("/files", ctx)
        out = _output(ctx)
        assert "data.csv[" not in out
        assert "data.csv" in out


class TestModelCommand:
    def test_no_args_and_no_catalog_reports_unavailable(self) -> None:
        ctx = _ctx(list_models=None)
        execute("/model", ctx)
        assert "unavailable" in _output(ctx).lower()

    def test_no_args_lists_catalog(self) -> None:
        choice = SimpleNamespace(
            selector="anthropic:claude-sonnet-5", provider="anthropic", has_key=True
        )
        ctx = _ctx(list_models=lambda: [choice])
        execute("/model", ctx)
        assert "anthropic:claude-sonnet-5" in _output(ctx)

    def test_with_arg_returns_set_model_outcome(self) -> None:
        ctx = _ctx()
        outcome = execute("/model groq:foo", ctx)
        assert outcome.set_model == "groq:foo"


class TestPermissionsCommand:
    def test_unavailable_when_no_summary_provider(self) -> None:
        ctx = _ctx(permissions_summary=None)
        execute("/permissions", ctx)
        assert "unavailable" in _output(ctx).lower()

    def test_reports_mode_and_allow_list(self) -> None:
        summary = SimpleNamespace(mode="prompt", allow=("aggregate", "plot"))
        ctx = _ctx(permissions_summary=lambda: summary)
        execute("/permissions", ctx)
        out = _output(ctx)
        assert "prompt" in out
        assert "aggregate" in out and "plot" in out


class TestSessionsAndResume:
    def test_sessions_with_none_saved_prints_hint(self) -> None:
        ctx = _ctx(engine=SimpleNamespace(sessions=lambda: []))
        execute("/sessions", ctx)
        assert "No saved sessions" in _output(ctx)

    def test_sessions_lists_threads(self) -> None:
        info = SimpleNamespace(thread_id="abc", updated_at="2026-01-01", turns=2, preview="hi")
        ctx = _ctx(engine=SimpleNamespace(sessions=lambda: [info]))
        execute("/sessions", ctx)
        assert "abc" in _output(ctx)

    def test_resume_without_args_lists_and_hints(self) -> None:
        info = SimpleNamespace(thread_id="abc", updated_at="2026-01-01", turns=2, preview="hi")
        ctx = _ctx(engine=SimpleNamespace(sessions=lambda: [info]))
        execute("/resume", ctx)
        out = _output(ctx)
        assert "abc" in out
        assert "/resume <thread-id>" in out

    def test_resume_with_matching_id_switches(self) -> None:
        info = SimpleNamespace(thread_id="abc", updated_at="2026-01-01", turns=2, preview="hi")
        ctx = _ctx(engine=SimpleNamespace(sessions=lambda: [info]))
        outcome = execute("/resume abc", ctx)
        assert outcome.resume_thread_id == "abc"
        assert "Resumed" in _output(ctx)

    def test_resume_with_unknown_id_warns(self) -> None:
        info = SimpleNamespace(thread_id="abc", updated_at="2026-01-01", turns=2, preview="hi")
        ctx = _ctx(engine=SimpleNamespace(sessions=lambda: [info]))
        outcome = execute("/resume zzz", ctx)
        assert outcome.resume_thread_id is None
        assert "No session" in _output(ctx)

    def test_resume_with_no_saved_sessions_prints_hint(self) -> None:
        ctx = _ctx(engine=SimpleNamespace(sessions=lambda: []))
        execute("/resume", ctx)
        assert "No saved sessions" in _output(ctx)


class TestExportCommand:
    def test_writes_transcript_markdown_to_the_given_path(self, tmp_path) -> None:
        entry = SimpleNamespace(role="user", text="hello")
        target = tmp_path / "out.md"
        ctx = _ctx(
            session=SimpleNamespace(
                thread_id="t1",
                turns=1,
                total_input_tokens=0,
                total_output_tokens=0,
                total_cost_usd=0.0,
                transcript=[entry],
            )
        )
        execute(f"/export {target}", ctx)
        content = target.read_text()
        assert "# CellSense session `t1`" in content
        assert "**User:**" in content
        assert "hello" in content

    def test_default_path_uses_the_thread_id(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        ctx = _ctx(
            session=SimpleNamespace(
                thread_id="t9",
                turns=0,
                total_input_tokens=0,
                total_output_tokens=0,
                total_cost_usd=0.0,
                transcript=[],
            )
        )
        execute("/export", ctx)
        assert (tmp_path / "cellsense-session-t9.md").is_file()


class TestSlashCompleter:
    def _completions(self, text: str) -> list:
        completer = SlashCompleter()
        document = Document(text=text, cursor_position=len(text))
        return list(completer.get_completions(document, None))

    def test_completes_partial_command_name(self) -> None:
        completions = self._completions("/mo")
        assert any(c.text == "/model" for c in completions)

    def test_no_completions_once_a_space_is_typed(self) -> None:
        assert self._completions("/model x") == []

    def test_no_completions_without_a_leading_slash(self) -> None:
        assert self._completions("model") == []

    def test_alias_prefix_yields_the_canonical_command_name(self) -> None:
        # Typing a prefix of the alias "quit" still inserts the canonical "/exit".
        completions = self._completions("/qui")
        assert any(c.text == "/exit" for c in completions)

    def test_display_meta_is_the_command_summary(self) -> None:
        completions = self._completions("/hel")
        match = next(c for c in completions if c.text == "/help")
        assert match.display_meta_text == "Show this message"
