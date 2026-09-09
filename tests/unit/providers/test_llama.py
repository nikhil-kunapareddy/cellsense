"""Pins cellsense.providers.llama's "no default host anymore" behaviour: a
missing LLAMA_BASE_URL raises ConfigError before any network I/O or SDK
import happens.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from cellsense.errors import ConfigError
from cellsense.providers.base import ModelSettings
from cellsense.providers.llama import DEFAULT_MODEL, _build


def test_missing_base_url_raises_config_error(monkeypatch) -> None:
    monkeypatch.delenv("LLAMA_BASE_URL", raising=False)
    monkeypatch.setenv("LLAMA_API_KEY", "some-key")  # present or not, shouldn't matter
    with pytest.raises(ConfigError, match="LLAMA_BASE_URL"):
        _build(DEFAULT_MODEL, ModelSettings())


def test_missing_base_url_raises_even_without_an_api_key(monkeypatch) -> None:
    monkeypatch.delenv("LLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("LLAMA_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        _build(DEFAULT_MODEL, ModelSettings())


def test_missing_base_url_error_never_imports_langchain_or_touches_the_network() -> None:
    """Subprocess proof: fresh interpreter, LLAMA_BASE_URL unset, call _build(),
    and confirm langchain_openai was never imported as a side effect of the
    ConfigError path.
    """
    code = (
        "import os, sys\n"
        "os.environ.pop('LLAMA_BASE_URL', None)\n"
        "from cellsense.providers.llama import _build, DEFAULT_MODEL\n"
        "from cellsense.providers.base import ModelSettings\n"
        "from cellsense.errors import ConfigError\n"
        "try:\n"
        "    _build(DEFAULT_MODEL, ModelSettings())\n"
        "except ConfigError:\n"
        "    pass\n"
        "else:\n"
        "    sys.exit(2)\n"
        "sys.exit(1 if 'langchain_openai' in sys.modules else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
