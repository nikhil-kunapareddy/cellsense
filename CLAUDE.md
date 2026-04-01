# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CellSense is a terminal CLI tool that lets users ask natural language questions about Excel/CSV files. It uses Claude (via Anthropic SDK tool-calling) to plan and answer queries, and renders output with the `rich` library.

## Setup and running

```bash
# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# Run with one or more files
python main.py sales.xlsx headcount.csv
python main.py data.csv
```

## Architecture

Packages under `src/` group related modules.

```
main.py                    argparse entry point; validates files; calls load_files → run_repl
src/repl/                  REPL loop, banner, slash commands (/help /files /clear /exit), rich UI
src/data/ingestion.py      load_files() → Dict[filename, FileData]; multi-sheet Excel + CSV
src/utils/context_builder.py  build_context(file_data) → compressed string for LLM system prompt
src/utils/citations.py   Citation dataclass + format_citations() / merge_citations()
src/agents/claude.py     ask_question() — Anthropic tool-calling loop (plan → tool → answer)
src/agents/llama.py      ask_question() — Meta Llama API tool-calling loop
src/tools/               Tool schemas (ALL_SCHEMAS / ALL_OAI_SCHEMAS) + handlers (filter_rows, aggregate, join_files)
```

**Data flow:**
1. `main.py` parses CLI args → calls `src.data.load_files()` → `repl.run_repl()`
2. `repl` builds context once via `context_builder.build_context()`, maintains `history: List[dict]`
3. Each question goes to `agent.ask_question(question, file_data, context, history)`
4. `agent` sends history + system prompt to Claude; when Claude requests a tool, `tools.TOOL_HANDLERS[name]` executes it against the live DataFrames and returns a `ToolResult` (which includes `Citation` objects)
5. Tool results are appended to history and the loop continues until Claude returns a final text response
6. `repl` renders the answer in a `rich.Panel` with the citation block appended

## Key types

- `FileData` (`src/data/ingestion.py`): `path`, `sheets: Dict[str, pd.DataFrame]`, `file_type`
- `ToolResult` (`src/tools/`): `data: pd.DataFrame`, `citations: List[Citation]`, `summary: str`
- `Citation` (`src/utils/citations.py`): `filename`, `sheet_name` (None for CSV), `row_indices`

## LLM details

- Model: `claude-sonnet-4-5` (set in `agents/claude.py:MODEL`)
- Tool-calling uses Anthropic's native tool use API — no LangChain
- `MAX_TOOL_ROUNDS = 8` caps the tool loop in each agent module
- The system prompt is in `agents/claude.py:_SYSTEM_PROMPT` (and Llama equivalent); file context is injected via `.format(context=context)`
- Conversation history is a mutable `List[dict]` passed through from `repl` — the full multi-turn session is preserved

## Slash commands

`/help`, `/files`, `/clear`, `/exit` — all handled in `repl.run_repl()` before the question reaches the agent.
