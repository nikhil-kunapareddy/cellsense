"""Pins each real provider module's MissingAPIKeyError wiring: the right
env var name is reported, and no SDK import is attempted before the key
check runs.
"""

from __future__ import annotations

import sys

import pytest

from cellsense.errors import MissingAPIKeyError
from cellsense.providers import anthropic, gemini, groq, openai
from cellsense.providers.base import ModelSettings

_REAL_PROVIDER_MODULES = [
    (anthropic, "ANTHROPIC_API_KEY", "langchain_anthropic"),
    (openai, "OPENAI_API_KEY", "langchain_openai"),
    (gemini, "GEMINI_API_KEY", "langchain_google_genai"),
    (groq, "GROQ_API_KEY", "langchain_groq"),
]


@pytest.mark.parametrize("module,env_key,sdk_package", _REAL_PROVIDER_MODULES)
def test_missing_api_key_names_the_right_env_var(module, env_key, sdk_package, monkeypatch) -> None:
    monkeypatch.delenv(env_key, raising=False)
    with pytest.raises(MissingAPIKeyError) as exc_info:
        module._build(module.DEFAULT_MODEL, ModelSettings())
    assert exc_info.value.env_key == env_key
    assert exc_info.value.provider == module.PROVIDER
    assert env_key in (exc_info.value.hint or "")


@pytest.mark.parametrize("module,env_key,sdk_package", _REAL_PROVIDER_MODULES)
def test_missing_api_key_path_never_imports_the_sdk(
    module, env_key, sdk_package, monkeypatch
) -> None:
    monkeypatch.delenv(env_key, raising=False)
    was_already_imported = sdk_package in sys.modules
    with pytest.raises(MissingAPIKeyError):
        module._build(module.DEFAULT_MODEL, ModelSettings())
    if not was_already_imported:
        assert sdk_package not in sys.modules
