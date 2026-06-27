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
    """
    This function sets up command line arguments for the program.
    1. Defines what inputs the program expects.
    2. Parses user input.
    3. Returns the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="cellsense",
        description="Ask natural language questions about your Excel and CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py sales.xlsx\n"
            "  python main.py sales.xlsx headcount.csv --agent llama\n"
            "  python main.py sales.xlsx -q \"total revenue by region?\""
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
        choices=["claude", "llama", "groq", "gemini"],
        default="llama",
        help=(
            "LLM backend: 'claude' (Anthropic), 'llama' (Meta, default), "
            "'groq' (Groq), or 'gemini' (Google)."
        ),
    )
    parser.add_argument(
        "-q", "--query",
        metavar="QUESTION",
        default=None,
        help=(
            "Ask a single question and exit (non-interactive print mode). "
            "If omitted, CellSense starts an interactive REPL."
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
    if errors:
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    return paths


def _load_agent(choice: str):
    """Select the LLM backend and return (ask_fn, agent_label) for the REPL."""
    if choice == "claude":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print(
                "error: ANTHROPIC_API_KEY is not set.\n"
                "  export ANTHROPIC_API_KEY=sk-ant-...",
                file=sys.stderr,
            )
            sys.exit(1)
        from src.core.agents.claude import ask_question, AGENT_LABEL
        return ask_question, AGENT_LABEL

    from src.core.agents.openai_compat import GROQ, GEMINI, LLAMA, make_ask_fn, get_label

    configs = {"groq": GROQ, "gemini": GEMINI, "llama": LLAMA}
    config = configs[choice]

    if not os.environ.get(config.api_key_env):
        print(f"error: {config.api_key_env} is not set.", file=sys.stderr)
        sys.exit(1)

    return make_ask_fn(config), get_label(config)


def main() -> None:
    args = _parse_args()
    paths = _resolve_paths(args.files)
    ask_fn, agent_label = _load_agent(args.agent)

    from src.data import load_files

    file_data: dict = {}
    if paths:
        try:
            file_data = load_files(paths)
        except Exception as exc:
            print(f"error loading files: {exc}", file=sys.stderr)
            sys.exit(1)

    if args.query:
        from src.modes.print import run_print
        sys.exit(run_print(file_data, ask_fn, agent_label, args.query))

    from src.modes.interactive import run_repl
    run_repl(file_data, ask_fn, agent_label)


if __name__ == "__main__":
    main()
