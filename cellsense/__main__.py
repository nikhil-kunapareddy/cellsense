"""Entry point for ``python -m cellsense``.

Mirrors the ``cellsense`` console script declared in ``pyproject.toml``
(``[project.scripts] cellsense = "cellsense.cli:main"``) so both invocation
styles behave identically.
"""

from __future__ import annotations

from cellsense.cli import main

if __name__ == "__main__":
    main()
