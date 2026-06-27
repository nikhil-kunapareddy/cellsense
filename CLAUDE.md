# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CellSense is a terminal CLI tool that lets users ask natural language questions about Excel/CSV files. It uses LLM tool-calling to plan and answer queries, and renders output with the `rich` library. Four interchangeable backends are supported: `claude` (Anthropic), `llama` (Meta, **default**), `groq`, and `gemini`.

## Setup and running

```bash
# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# API keys are read from the environment or a local .env file.
# Set the key for whichever backend you use:
#   ANTHROPIC_API_KEY   (--agent claude)
#   LLAMA_API_KEY       (--agent llama, the default)
#   GROQ_API_KEY        (--agent groq)
#   GEMINI_API_KEY      (--agent gemini)

# Interactive REPL (one or more files)
python main.py sales.xlsx headcount.csv
python main.py data.csv --agent claude

# One-shot / print mode: answer a single question and exit
python main.py data.csv -q "total revenue by region?"

# With no files, ask the agent to find them
python main.py -q "find sales files in ./data"
```

## Architecture

Modules under `src/` are grouped by role: `core/` (engine), `tools/` (one file per tool), `modes/` (how it's run). `src/` uses implicit namespace packages — there are no `__init__.py` files.

```
main.py                         argparse entry; validates files; loads agent; dispatches to a mode
src/data.py                     load_files() → Dict[filename, FileData]; multi-sheet Excel + CSV; SUPPORTED_EXTENSIONS
src/types.py                    ToolResult + sheet-lookup helpers (_get_sheet, _sheet_label)
src/core/orchestrator.py        orchestrate() (guardrail → plan → parallel workers → synthesize) + execute_tool()
src/core/guardrails.py          check_relevance() — rejects off-topic questions before any tool calls
src/core/agents/claude.py       ask_question() — Anthropic native tool-calling loop
src/core/agents/openai_compat.py ask_question() factory for groq/gemini/llama (OpenAI-compatible; AgentConfig per provider)
src/tools/<tool>.py             one module per tool, each exposing  SCHEMA + handle(inputs, file_data)
src/tools/registry.py           derives ALL_SCHEMAS / ALL_OAI_SCHEMAS / TOOL_HANDLERS from the tool modules
src/modes/interactive.py        run_repl() — REPL loop, banner, slash commands, rich UI
src/modes/print.py              run_print() — one-shot answer for -q/--query, returns an exit code
src/mcp/server.py               standalone MCP server (stdio) exposing list_directory / find_files
src/utils/context_builder.py    build_context(file_data) → compressed string for the LLM system prompt
src/utils/citations.py          Citation dataclass + format_citations() / merge_citations()
src/utils/classifier.py         classify_file() — keyword-based business-category label
```

Tools (each its own module under `src/tools/`): `filter_rows`, `aggregate`, `join_files`, `plot`, `list_directory`, `find_files`.

**Data flow:**
1. `main.py` parses CLI args, selects a backend via `_load_agent()`, and calls `src.data.load_files()`.
2. It dispatches to a **mode**: `modes.print.run_print()` when `-q` is given, otherwise `modes.interactive.run_repl()`.
3. The mode builds context once via `context_builder.build_context()`, maintains `history: List[dict]`, and calls `ask_fn(question, file_data, context, history)`.
4. Each backend's `ask_question` delegates to `core.orchestrator.orchestrate()`: a relevance guardrail runs first, then the question is planned into sub-tasks; a single sub-task falls straight through to the backend's tool loop, multiple sub-tasks fan out across threads and are synthesized.
5. In the tool loop, when the model requests a tool, `execute_tool()` looks it up in `tools.registry.TOOL_HANDLERS` and runs it against the live DataFrames, returning a `ToolResult` (with `Citation` objects).
6. The mode renders the final answer in a `rich.Panel` with the citation block appended.

## Key types

- `FileData` (`src/data.py`): `path`, `sheets: Dict[str, pd.DataFrame]`, `file_type`; properties `filename`, `total_rows`
- `ToolResult` (`src/types.py`): `data: pd.DataFrame`, `citations: List[Citation]`, `summary: str`; `to_text()` for the model
- `Citation` (`src/utils/citations.py`): `filename`, `sheet_name` (None for CSV), `row_indices`

## LLM details

- Orchestration, the shared `SYSTEM_PROMPT`, and `MAX_TOOL_ROUNDS = 8` live in `core/orchestrator.py` (used by every backend).
- Anthropic backend: `core/agents/claude.py`, model `claude-sonnet-4-5` (set in `MODEL`); uses Anthropic's native tool use API — no LangChain.
- OpenAI-compatible backends: `core/agents/openai_compat.py`. One `AgentConfig` per provider (`GROQ`, `GEMINI`, `LLAMA`) captures base URL, default model, and Meta-specific quirks (`meta_response_format`, `normalize_bool_args`). Default models are overridable via `GROQ_MODEL` / `GEMINI_MODEL` / `LLAMA_MODEL`.
- Conversation `history` is a mutable `List[dict]` owned by the mode — the full multi-turn session is preserved.

## Adding a tool

1. Create `src/tools/<name>.py` exposing `SCHEMA` (Anthropic-style dict) and `handle(inputs, file_data) -> ToolResult`.
2. Add the module to `_MODULES` in `src/tools/registry.py`. The registry auto-generates the OpenAI schema and keys the handler off `SCHEMA["name"]` — no other wiring needed.

## Slash commands

`/help`, `/files`, `/clear`, `/exit` — all handled in `modes/interactive.run_repl()` before the question reaches the agent. (Print mode has no slash commands.)
