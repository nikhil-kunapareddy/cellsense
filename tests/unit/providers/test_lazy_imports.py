"""Pins the ARCHITECTURE.md §4 rule 1 "lazy import" invariant properly: each
provider module must not import its langchain_* SDK at module scope.

Run each check in a fresh subprocess (same interpreter/venv this suite runs
under) rather than inspecting sys.modules in-process -- the in-process
version would only prove "nothing *else* in this test run happened to import
it yet", which is a fact about test ordering, not about the module itself.
A subprocess is the only way to prove import cellsense.providers.x alone
does not pull in the SDK.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

_PROVIDER_MODULES = [
    ("cellsense.providers.anthropic", "langchain_anthropic"),
    ("cellsense.providers.openai", "langchain_openai"),
    ("cellsense.providers.gemini", "langchain_google_genai"),
    ("cellsense.providers.groq", "langchain_groq"),
    ("cellsense.providers.llama", "langchain_openai"),
]


@pytest.mark.parametrize("module_name,sdk_package", _PROVIDER_MODULES)
def test_importing_a_provider_module_does_not_import_its_sdk(module_name, sdk_package) -> None:
    code = (
        f"import sys, importlib\n"
        f"importlib.import_module({module_name!r})\n"
        f"sys.exit(1 if {sdk_package!r} in sys.modules else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, (
        f"importing {module_name} appears to have pulled in {sdk_package}\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


def test_importing_the_provider_registry_does_not_import_any_sdk() -> None:
    """registry.py imports all five provider modules at top level -- that's
    fine per its own docstring, as long as none of *them* eagerly import an
    SDK either.
    """
    sdk_packages = {pkg for _, pkg in _PROVIDER_MODULES}
    code = (
        "import sys, importlib\n"
        "importlib.import_module('cellsense.providers.registry')\n"
        f"leaked = {sorted(sdk_packages)!r}\n"
        "found = [p for p in leaked if p in sys.modules]\n"
        "sys.exit(1 if found else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
