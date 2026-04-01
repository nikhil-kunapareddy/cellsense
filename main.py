#!/usr/bin/env python3
"""CellSense — AI-powered CLI for analyzing Excel and CSV files."""
import argparse
import os
import sys
from pathlib import Path

# Load .env if present — must happen before any os.environ reads
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SUPPORTED_EXTENSIONS = {".xlsx", ".xls", ".csv"}


def _parse_args():
    parser = argparse.ArgumentParser(
        prog="cellsense",
        description="Ask natural language questions about your Excel and CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py sales.xlsx\n"
            "  python main.py sales.xlsx headcount.csv --agent llama"
        ),
    )
    parser.add_argument(
        "files",
        nargs="*",
        metavar="FILE",
        help=(
            "One or more .xlsx, .xls, or .csv files to analyze. "
            "If omitted, ask the agent to find files (e.g. 'find sales files in ~/Downloads')."
        ),
    )
    parser.add_argument(
        "--agent",
        choices=["claude", "llama", "groq"],
        default="llama",
        help=(
            "LLM backend: 'claude' (Anthropic), 'llama' (Meta Llama API, default), "
            "or 'groq' (Groq)."
        ),
    )
    return parser.parse_args()


def _resolve_paths(raw_files: list[str]) -> list[Path]:
    """Turn CLI file strings into Path objects after validating each one.

    Every path must exist and end with a supported extension (.xlsx, .xls, .csv).
    Invalid entries are collected so the user sees all problems in one run; if any
    fail, messages go to stderr and the process exits with code 1.
    """
    paths: list[Path] = []
    errors: list[str] = []
    for raw in raw_files:
        p = Path(raw)
        if not p.exists():
            errors.append(f"File not found: {raw}")
        elif p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            errors.append(f"Unsupported file type '{p.suffix}': {raw}")
        else:
            paths.append(p)
    # Fail after checking every file so the user gets a full error list at once.
    if errors:
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    return paths


def _load_agent(choice: str):
    """Select the LLM backend and return its entry points for the REPL.

    ``choice`` is ``"claude"`` (Anthropic) or ``"llama"`` (Meta). Each backend
    requires its API key in the environment; if missing, prints a hint to stderr
    and exits with code 1. Otherwise imports only that agent module and returns
    ``(ask_question, AGENT_LABEL)`` for ``run_repl``.
    """
    # Import inside each branch so the unused backend's dependencies are not loaded.
    if choice == "claude":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print(
                "error: ANTHROPIC_API_KEY is not set.\n"
                "  export ANTHROPIC_API_KEY=sk-ant-...",
                file=sys.stderr,
            )
            sys.exit(1)
        from src.agents import claude
        return claude.ask_question, claude.AGENT_LABEL

    elif choice == "groq":
        if not os.environ.get("GROQ_API_KEY"):
            print(
                "error: GROQ_API_KEY is not set.\n"
                "  export GROQ_API_KEY=gsk_...",
                file=sys.stderr,
            )
            sys.exit(1)
        from src.agents import groq
        return groq.ask_question, groq.AGENT_LABEL

    else:  # llama
        if not os.environ.get("LLAMA_API_KEY"):
            print(
                "error: LLAMA_API_KEY is not set.\n"
                "  export LLAMA_API_KEY=<your-meta-llama-api-key>",
                file=sys.stderr,
            )
            sys.exit(1)
        from src.agents import llama
        return llama.ask_question, llama.AGENT_LABEL


def main() -> None:
    args = _parse_args()
    paths = _resolve_paths(args.files)
    ask_fn, agent_label = _load_agent(args.agent)

    from src.data import load_files
    from src.repl import run_repl

    file_data: dict = {}
    if paths:
        try:
            file_data = load_files(paths)
        except Exception as exc:
            print(f"error loading files: {exc}", file=sys.stderr)
            sys.exit(1)

    run_repl(file_data, ask_fn, agent_label)


if __name__ == "__main__":
    main()
