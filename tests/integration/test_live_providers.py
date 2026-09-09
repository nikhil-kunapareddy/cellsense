"""Live tests: real network calls against real provider APIs.

Gated behind ``@pytest.mark.live`` (deselected by the project's own
``pytest -q -m "not live"`` gate) *and* an explicit ``skipif`` per provider so
running this file with ``-m live`` but no keys exported is still a clean,
zero-network skip rather than an error.

This environment currently has **no** provider keys exported into the shell
(only Groq/Gemini/Llama keys live in the gitignored ``.env``, and nothing in
this file loads it -- see ``test_cli.py``'s isolation-fixture docstring for
why that matters), so running ``pytest -m live`` here is expected to skip
every single test below and make zero network calls. Per the task brief,
these were written and proven-to-skip but deliberately never run for real.

Kept to one cheap, single-number question per provider; expectations are
derived from the real ``data/sales_2024.csv`` via pandas rather than
hardcoded, even though ``scripts/make_sample_data.py``'s own docstring
documents a stable total for its fixed seed.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pandas as pd
import pytest

from cellsense.config import Config, ModelConfig
from cellsense.events import TurnFailed, TurnFinished
from cellsense.graph import engine as engine_module
from cellsense.graph.engine import Engine
from cellsense.io.loaders import load_workspace
from cellsense.providers.registry import PROVIDERS
from cellsense.tools.registry import ToolRegistry

pytestmark = pytest.mark.live

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SALES_CSV = DATA_DIR / "sales_2024.csv"


def _provider_ready(provider_name: str) -> tuple[bool, str]:
    spec = PROVIDERS[provider_name]
    if not os.environ.get(spec.env_key):
        return False, f"{spec.env_key} is not set"
    if provider_name == "llama" and not os.environ.get("LLAMA_BASE_URL"):
        return False, "LLAMA_BASE_URL is not set"
    return True, ""


@pytest.mark.parametrize("provider_name", sorted(PROVIDERS))
def test_total_revenue_question_matches_a_pandas_computed_total(
    provider_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready, reason = _provider_ready(provider_name)
    if not ready:
        pytest.skip(reason)
    if not SALES_CSV.exists():
        pytest.skip("data/sales_2024.csv not present -- run scripts/make_sample_data.py")

    expected_total = round(pd.read_csv(SALES_CSV)["revenue"].sum(), 2)

    # Same documented workaround as every other Engine-level test in this
    # package -- see tests/integration/conftest.py's make_engine docstring
    # and test_engine_bug_stream_modes.py for the root cause.
    monkeypatch.setattr(engine_module, "STREAM_MODES", ["custom", "updates", "messages"])

    workspace = load_workspace([SALES_CSV])
    config = Config(model=ModelConfig(default=provider_name))
    engine = Engine(workspace, config, registry=ToolRegistry(), db_path=tmp_path / "sessions.db")

    events = list(
        engine.stream_turn(
            "What is the total revenue across all rows? Answer with just the number.",
            thread_id=uuid.uuid4().hex,
            on_approval=lambda _request: "deny",
        )
    )
    terminal = next(e for e in events if isinstance(e, (TurnFinished, TurnFailed)))
    assert isinstance(terminal, TurnFinished), getattr(terminal, "message", terminal)

    answer_digits = terminal.answer.replace(",", "")
    assert str(int(expected_total)) in answer_digits, (
        f"expected the pandas-computed total {expected_total} to appear "
        f"in the model's answer: {terminal.answer!r}"
    )


@pytest.mark.parametrize("provider_name", sorted(PROVIDERS))
def test_row_count_question_matches_the_real_file(
    provider_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready, reason = _provider_ready(provider_name)
    if not ready:
        pytest.skip(reason)
    if not SALES_CSV.exists():
        pytest.skip("data/sales_2024.csv not present -- run scripts/make_sample_data.py")

    expected_rows = len(pd.read_csv(SALES_CSV))

    monkeypatch.setattr(engine_module, "STREAM_MODES", ["custom", "updates", "messages"])

    workspace = load_workspace([SALES_CSV])
    config = Config(model=ModelConfig(default=provider_name))
    engine = Engine(workspace, config, registry=ToolRegistry(), db_path=tmp_path / "sessions.db")

    events = list(
        engine.stream_turn(
            "How many rows are in this file? Answer with just the number.",
            thread_id=uuid.uuid4().hex,
            on_approval=lambda _request: "deny",
        )
    )
    terminal = next(e for e in events if isinstance(e, (TurnFinished, TurnFailed)))
    assert isinstance(terminal, TurnFinished), getattr(terminal, "message", terminal)
    assert str(expected_rows) in terminal.answer.replace(",", "")


def test_this_module_skips_cleanly_with_no_keys_exported() -> None:
    """A meta-check pinning the skip behaviour itself: with every provider
    key absent (this repo's shell has none exported -- see the module
    docstring), every parametrized case above must be a real ``skip``, never
    an error or an accidental live call.
    """
    for provider_name in PROVIDERS:
        ready, _reason = _provider_ready(provider_name)
        if os.environ.get(PROVIDERS[provider_name].env_key):
            pytest.skip(
                f"{provider_name}'s key is actually set in this environment; "
                "this meta-check only asserts the no-keys-present baseline."
            )
        assert ready is False
