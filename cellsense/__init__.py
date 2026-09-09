"""CellSense: an agent harness for spreadsheets.

This module is intentionally import-light. Importing ``cellsense`` must never
pull in pandas, langgraph, or any provider SDK -- those live behind
``cellsense.io``, ``cellsense.graph``, and ``cellsense.providers``, which are
only imported once the CLI actually needs them (see ``cellsense/cli.py``).
That keeps ``python -c "import cellsense"`` (and anything that merely wants
``__version__``) fast and dependency-free.

``__version__`` is read from the installed package's metadata so it never
drifts from what ``pyproject.toml`` declares; the hardcoded fallback only
matters for an unusual in-place/uninstalled checkout.
"""

from __future__ import annotations

from importlib import metadata

from cellsense.errors import CellSenseError

__all__ = ["CellSenseError", "__version__"]

_FALLBACK_VERSION = "0.2.0"

try:
    __version__ = metadata.version("cellsense")
except metadata.PackageNotFoundError:  # pragma: no cover - only when not installed
    __version__ = _FALLBACK_VERSION
