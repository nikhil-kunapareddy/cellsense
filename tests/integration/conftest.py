"""Fixtures shared by the integration suite.

Builds a real ``Engine`` (real graph, real permission policy, real SQLite
checkpointer) wired to a :class:`~tests.fixtures.fake_chat_model.ScriptedChatModel`
instead of a real provider -- no network, no API key, no ``langchain_*``
integration package required anywhere in this package.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from cellsense.config import Config
from cellsense.events import Event, TurnFailed, TurnFinished
from cellsense.graph import engine as engine_module
from cellsense.io.schema import Workspace
from cellsense.providers.base import ModelSettings, ModelSpec, ProviderSpec
from cellsense.providers.registry import ResolvedModel
from cellsense.tools.registry import ToolRegistry
from tests.fixtures.fake_chat_model import Script, ScriptedChatModel

__all__: list[str] = []

MakeEngine = Callable[..., engine_module.Engine]


@pytest.fixture
def make_chat_model() -> Callable[[Script], ScriptedChatModel]:
    """Factory: ``make_chat_model(script_fn) -> ScriptedChatModel``."""

    def _make(script: Script) -> ScriptedChatModel:
        return ScriptedChatModel(script=script)

    return _make


@pytest.fixture
def make_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MakeEngine:
    """Factory: ``make_engine(workspace, chat_model, config=None, db_path=None)`` -> a real :class:`~cellsense.graph.engine.Engine`
    wired to ``chat_model``.

    Bypasses real provider construction by monkeypatching
    ``cellsense.graph.engine.resolve_model`` (that module's own imported
    binding of ``providers.registry.resolve_model``) to hand back a
    ``ResolvedModel`` whose ``.chat_model`` is the fake -- ``Engine.__init__``
    cannot tell the difference from a real provider.

    Uses the real ``engine_module.STREAM_MODES`` deliberately. It was briefly
    monkeypatched here to work around a shipped bug (``STREAM_MODES`` was a
    tuple, and langgraph only tags stream chunks as ``(mode, payload)`` for a
    ``list``). That bug is fixed, and patching it here would hide a regression
    from every test in this package, so the workaround is gone -- see
    ``test_engine_bug_stream_modes.py`` for the dedicated guard.
    """

    def _make(
        workspace: Workspace,
        chat_model: ScriptedChatModel,
        *,
        config: Config | None = None,
        db_path: Path | None = None,
    ) -> engine_module.Engine:
        cfg = config if config is not None else Config()

        provider_spec = ProviderSpec(
            name="fake",
            label="Fake",
            env_key="FAKE_API_KEY",
            package="fake_pkg",
            extra="fake",
            default_model="fake-model",
            models=(),
            builder=lambda model_id, settings: chat_model,
        )
        model_spec = ModelSpec(
            id="fake-model",
            provider="fake",
            display="Fake Model",
            context_window=128_000,
            max_output_tokens=4_096,
            input_usd_per_mtok=None,
            output_usd_per_mtok=None,
        )

        def _fake_resolve_model(selector: str | None, settings: ModelSettings) -> ResolvedModel:
            return ResolvedModel(provider_spec, model_spec, settings)

        monkeypatch.setattr(engine_module, "resolve_model", _fake_resolve_model)

        path = db_path if db_path is not None else tmp_path / "sessions.db"
        registry = ToolRegistry()
        return engine_module.Engine(workspace, cfg, registry=registry, db_path=path)

    return _make


def terminal_event(events: list[Event]) -> TurnFinished | TurnFailed:
    """Assert the hard contract (ARCHITECTURE.md SS7 / ``engine.py``'s module
    docstring): exactly one ``TurnFinished`` or ``TurnFailed`` in ``events`` --
    never neither, never both -- and return it.
    """
    terminals = [e for e in events if isinstance(e, (TurnFinished, TurnFailed))]
    assert len(terminals) == 1, f"expected exactly one terminal event, got: {terminals!r}"
    return terminals[0]


@pytest.fixture
def assert_one_terminal_event() -> Callable[[list[Event]], TurnFinished | TurnFailed]:
    return terminal_event
