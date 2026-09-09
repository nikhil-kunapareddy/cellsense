#!/usr/bin/env python3
"""Deprecated entry point -- kept only so old invocations keep working.

CellSense's CLI has moved to the ``cellsense`` package (install with
``pip install -e '.[all,dev]'`` and run ``cellsense ...``; see
``pyproject.toml``'s ``[project.scripts]`` and ``docs/USAGE.md``). This shim
prints a deprecation notice to stderr, translates the one flag whose meaning
changed (``--agent {claude,llama,groq,gemini}`` -> ``--model <selector>``),
and delegates everything else to ``cellsense.cli:main`` verbatim -- so
``python main.py sales.xlsx -q "..."`` and its exit code keep working exactly
as before.
"""

from __future__ import annotations

import sys

# Old `--agent` choices mapped onto bare-provider `--model` selectors
# (`providers.registry.resolve_model` resolves a bare provider name to that
# provider's default model). "claude" is the one name that changed outright:
# the new provider registry calls that backend "anthropic".
_AGENT_TO_MODEL_SELECTOR = {
    "claude": "anthropic",
    "llama": "llama",
    "groq": "groq",
    "gemini": "gemini",
}

_DEPRECATION_NOTICE = (
    "cellsense: main.py is deprecated. Install the package (`pip install -e "
    "'.[all,dev]'`) and use the `cellsense` command instead -- this shim "
    "will be removed in a future release. See docs/USAGE.md for the new CLI."
)


def _translate_argv(argv: list[str]) -> list[str]:
    """Rewrite ``--agent <name>`` / ``--agent=<name>`` into ``--model <selector>``;
    every other token (files, ``-q``/``--query``, ...) passes through unchanged.
    """
    translated: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--agent" and i + 1 < len(argv):
            choice = argv[i + 1]
            translated.extend(["--model", _AGENT_TO_MODEL_SELECTOR.get(choice, choice)])
            i += 2
            continue
        if token.startswith("--agent="):
            choice = token.split("=", 1)[1]
            translated.extend(["--model", _AGENT_TO_MODEL_SELECTOR.get(choice, choice)])
            i += 1
            continue
        translated.append(token)
        i += 1
    return translated


def main() -> None:
    print(_DEPRECATION_NOTICE, file=sys.stderr)
    sys.argv = [sys.argv[0], *_translate_argv(sys.argv[1:])]

    from cellsense.cli import main as cellsense_main

    cellsense_main()  # exits the process itself (Click/Typer standalone mode)


if __name__ == "__main__":
    main()
