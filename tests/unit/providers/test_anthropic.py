"""Pins the Anthropic provider's sampling-parameter handling.

The Claude 5 generation (Fable 5.x, Opus 5, Sonnet 5) and Opus 4.7/4.8 removed
``temperature`` / ``top_p`` / ``top_k`` and reject them with a 400.
``langchain_anthropic`` does not drop them on our behalf -- it relocates them
into ``extra_body``, which the SDK merges into the request JSON verbatim -- so
passing ``ModelSettings.temperature`` through unconditionally (as every other
provider module does, and as this one used to) breaks every call to the default
model. See ``cellsense/providers/anthropic.py``'s module docstring.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from cellsense.providers import anthropic
from cellsense.providers.base import ModelSettings


@pytest.fixture
def captured_kwargs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Capture what ``_build`` would hand the LangChain constructor.

    Patches the name as bound in the provider module, so no API key is needed
    and the ``langchain_anthropic`` package is never imported.
    """
    captured: dict[str, Any] = {}

    def _fake_build_chat_model(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(anthropic, "build_chat_model", _fake_build_chat_model)
    return captured


@pytest.mark.parametrize(
    "model_id",
    ["claude-fable-5-1", "claude-opus-5", "claude-opus-4-8", "claude-sonnet-5"],
)
def test_temperature_is_omitted_for_models_that_reject_it(
    model_id: str, captured_kwargs: dict[str, Any]
) -> None:
    anthropic._build(model_id, ModelSettings(temperature=0.0))
    assert "temperature" not in captured_kwargs
    # The rest of ModelSettings must still be passed through (ARCHITECTURE.md
    # §4 rule 5) -- this is a targeted omission, not a bypass.
    assert captured_kwargs["max_tokens"] == 4096
    assert captured_kwargs["max_retries"] == 3


@pytest.mark.parametrize("model_id", sorted(anthropic._ACCEPTS_SAMPLING_PARAMS))
def test_temperature_is_passed_for_models_that_accept_it(
    model_id: str, captured_kwargs: dict[str, Any]
) -> None:
    anthropic._build(model_id, ModelSettings(temperature=0.25))
    assert captured_kwargs["temperature"] == 0.25


def test_unknown_model_ids_omit_temperature_rather_than_guessing(
    captured_kwargs: dict[str, Any],
) -> None:
    """An unrecognised id must still build (ARCHITECTURE.md §4 rule 2), and must
    do so without ``temperature``: omitting it costs the API's default sampling,
    while sending it to a model that removed it fails the whole call.
    """
    anthropic._build("claude-some-model-released-tomorrow", ModelSettings())
    assert "temperature" not in captured_kwargs
    assert captured_kwargs["model"] == "claude-some-model-released-tomorrow"


def test_default_model_is_curated_and_rejects_sampling_params() -> None:
    assert anthropic.DEFAULT_MODEL in {spec.id for spec in anthropic.MODELS}
    assert anthropic.DEFAULT_MODEL not in anthropic._ACCEPTS_SAMPLING_PARAMS


def test_curated_ids_carry_no_date_suffix() -> None:
    """Anthropic's current model ids are complete as-is; a date-suffixed variant
    (``claude-haiku-4-5-20251001``) is a stale-training-data shape.
    """
    for spec in anthropic.MODELS:
        assert re.search(r"-\d{8}$", spec.id) is None, spec.id


def test_temperature_never_reaches_the_wire_for_the_default_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The end-to-end assertion behind all of the above, against the real
    integration: no ``temperature`` in the request body, by any route.

    ``langchain_anthropic`` hides it in ``extra_body`` rather than in the
    payload's top level, so a naive ``"temperature" not in payload`` check
    passes even when the parameter is still being sent.
    """
    pytest.importorskip("langchain_anthropic")
    monkeypatch.setenv(anthropic.ENV_KEY, "sk-ant-not-a-real-key")

    model = anthropic.SPEC.build(anthropic.DEFAULT_MODEL, ModelSettings(temperature=0.0))
    payload = model._get_request_payload([("user", "hi")])

    assert "temperature" not in payload
    assert "temperature" not in (payload.get("extra_body") or {})
