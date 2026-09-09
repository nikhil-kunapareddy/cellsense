"""``cellsense.cli`` via ``typer.testing.CliRunner``.

Every test isolates ``$HOME`` and ``cwd`` into fresh ``tmp_path`` directories
(no real ``~/.cellsense/config.toml``, no real ``~/.cellsense/sessions.db``,
and critically no real ``.env`` -- the repo root's ``.env`` holds real
provider keys, and ``load_config``/``doctor``'s ``_load_env()`` calls
``load_dotenv()``, which would otherwise leak them into ``os.environ`` for
the rest of the test process). Provider env vars are also explicitly cleared
so a test asserting "no key" behaviour can't be accidentally satisfied by
whatever happens to be set in the real environment.

Success-path turns (``ask -q``, ``--print``) monkeypatch ``cellsense.cli.Engine``
itself with a stub -- nothing here ever reaches a real provider or the
network, per the task brief.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import cellsense.cli as cli_module
from cellsense import __version__
from cellsense.cli import app
from cellsense.events import RunStarted, TurnFinished

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.chdir(tmp_path)
    for var in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "LLAMA_API_KEY",
        "LLAMA_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)


class _FakeCliEngine:
    """Stands in for ``cellsense.graph.engine.Engine`` in CLI tests: no
    provider resolution, no checkpointer, no network -- just enough surface
    for ``ask``/``run_print``/``run_repl`` to drive a turn.
    """

    def __init__(self, workspace, config, *, registry, db_path=None) -> None:
        del workspace, config, registry, db_path

    @property
    def model_label(self) -> str:
        return "fake-model"

    @property
    def provider_label(self) -> str:
        return "Fake"

    def stream_turn(self, question, *, thread_id, on_approval, cancel=None):
        del on_approval, cancel
        yield RunStarted(
            question=question, thread_id=thread_id, model="fake-model", provider="Fake"
        )
        yield TurnFinished(
            answer="The total revenue is 123.45.",
            citations=["sales.csv [Rows: 0-2]"],
            duration_s=0.01,
        )

    def sessions(self):
        return []

    def history(self, thread_id: str) -> list[tuple[str, str]]:
        del thread_id
        return []


@pytest.fixture
def stub_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module, "Engine", _FakeCliEngine)


def _write_sales_csv(tmp_path: Path) -> Path:
    path = tmp_path / "sales.csv"
    path.write_text("region,revenue\nNorth,100\nSouth,200\n")
    return path


# ── --version / --help ───────────────────────────────────────────────────────


def test_version_flag_prints_the_version() -> None:
    # CliRunner invokes the Typer `app` object directly, bypassing main()'s
    # argv rewrite (which prepends "ask" so `cellsense --version` works with
    # no subcommand) -- so the "ask" subcommand is named explicitly here.
    # test_main_argv_rewrite below exercises that rewrite itself.
    result = runner.invoke(app, ["ask", "--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_top_level_help_lists_every_subcommand() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("ask", "models", "sessions", "init", "doctor"):
        assert name in result.stdout


def test_main_argv_rewrite_routes_a_bare_file_argument_to_ask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_engine
) -> None:
    csv_path = _write_sales_csv(tmp_path)
    monkeypatch.setattr("sys.argv", ["cellsense", str(csv_path), "-q", "total revenue?"])
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()
    assert exc_info.value.code == 0


# ── `models` ──────────────────────────────────────────────────────────────────


def test_models_lists_every_provider_and_key_status() -> None:
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    for selector_prefix in ("anthropic:", "openai:", "gemini:", "groq:", "llama:"):
        assert selector_prefix in result.stdout
    assert "missing" in result.stdout  # no keys are set, thanks to the isolation fixture


# ── `doctor` ──────────────────────────────────────────────────────────────────


def test_doctor_reports_every_section_without_touching_real_keys() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "CellSense doctor" in result.stdout
    assert "Provider API keys" in result.stdout
    assert "Provider packages" in result.stdout
    assert "Checkpoint DB" in result.stdout
    # The isolation fixture clears every provider env var.
    assert "not set" in result.stdout


# ── `init` ────────────────────────────────────────────────────────────────────


def test_init_writes_the_starter_config(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    config_path = tmp_path / ".cellsense" / "config.toml"
    assert config_path.is_file()
    assert "[model]" in config_path.read_text()


def test_init_refuses_to_clobber_an_existing_config_without_force(tmp_path: Path) -> None:
    first = runner.invoke(app, ["init"])
    assert first.exit_code == 0

    second = runner.invoke(app, ["init"])
    assert second.exit_code == 2  # ConfigError.exit_code

    forced = runner.invoke(app, ["init", "--force"])
    assert forced.exit_code == 0


# ── `sessions` [rm] ───────────────────────────────────────────────────────────


def test_sessions_lists_and_removes_a_real_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_engine,
    make_chat_model,
    make_workspace,
    sales_df,
) -> None:
    from tests.fixtures.fake_chat_model import ai_text, is_guardrail_call, is_planner_call

    def script(messages):
        if is_guardrail_call(messages):
            return ai_text("yes")
        if is_planner_call(messages):
            return ai_text('[{"id": "t1", "question": "q"}]')
        return ai_text("An answer.")

    db_path = tmp_path / "sessions.db"
    ws = make_workspace(sales=sales_df)
    engine = make_engine(ws, make_chat_model(script), db_path=db_path)
    list(engine.stream_turn("hi", thread_id="cli-session-1", on_approval=lambda _r: "deny"))

    monkeypatch.setattr(cli_module, "_DEFAULT_DB_PATH", db_path)

    listing = runner.invoke(app, ["sessions"])
    assert listing.exit_code == 0
    assert "cli-session-1" in listing.stdout

    removal = runner.invoke(app, ["sessions", "rm", "cli-session-1"])
    assert removal.exit_code == 0
    assert "Deleted session cli-session-1" in removal.stdout

    after = runner.invoke(app, ["sessions"])
    assert after.exit_code == 0
    assert "cli-session-1" not in after.stdout


def test_sessions_rm_of_an_unknown_id_is_a_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_module, "_DEFAULT_DB_PATH", tmp_path / "sessions.db")
    result = runner.invoke(app, ["sessions", "rm", "no-such-thread"])
    assert result.exit_code == 2


# ── `ask` error paths ─────────────────────────────────────────────────────────


def test_missing_file_exits_4() -> None:
    result = runner.invoke(app, ["ask", "no_such_file.csv", "-q", "hi"])
    assert result.exit_code == 4


def test_unsupported_extension_exits_4(tmp_path: Path) -> None:
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    result = runner.invoke(app, ["ask", str(bad), "-q", "hi"])
    assert result.exit_code == 4


def test_unknown_provider_exits_3(tmp_path: Path) -> None:
    csv_path = _write_sales_csv(tmp_path)
    result = runner.invoke(
        app, ["ask", str(csv_path), "-q", "hi", "--model", "not-a-real-provider:some-model"]
    )
    assert result.exit_code == 3


def test_llama_without_llama_base_url_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLAMA_API_KEY", "fake-key-for-this-test-only")
    csv_path = _write_sales_csv(tmp_path)
    result = runner.invoke(app, ["ask", str(csv_path), "-q", "hi", "--model", "llama"])
    assert result.exit_code == 2


# ── `ask -q` / `--print` success paths (Engine stubbed) ──────────────────────


def test_ask_dash_q_prints_the_answer_and_exits_0(tmp_path: Path, stub_engine) -> None:
    csv_path = _write_sales_csv(tmp_path)
    result = runner.invoke(app, ["ask", str(csv_path), "-q", "total revenue?"])
    assert result.exit_code == 0
    assert "123.45" in result.stdout


def test_print_mode_produces_ansi_free_stdout(tmp_path: Path, stub_engine) -> None:
    csv_path = _write_sales_csv(tmp_path)
    result = runner.invoke(app, ["ask", str(csv_path), "-q", "total revenue?", "--print"])
    assert result.exit_code == 0
    assert "\x1b" not in result.stdout
    assert "123.45" in result.stdout
    assert "sales.csv" in result.stdout
