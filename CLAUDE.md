# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## What this is

CellSense is an **agent harness for tabular data**: a terminal CLI that answers natural
language questions about CSV/Excel files by planning, calling typed pandas tools against
live DataFrames, and citing the source rows. Orchestration is a **LangGraph** `StateGraph`.
Five model providers are supported, **API-only by design** (no local models).

The harness is the product. Models are swappable, tools are pluggable, and every turn is
streamed, checkpointed, permission-gated, and logged.

## Environment — read this first

**Never create or use a virtualenv inside this repo, and never read from one.**
`~/Documents` is iCloud-synced on this machine and the disk runs near-full, so macOS
evicts `site-packages` files into `dataless` placeholders that **block forever** on read
(uninterruptible I/O, 0% CPU, immune to SIGTERM). A previous in-repo `.venv` had 935 of
them; `import langgraph` hung permanently because `importlib.metadata.entry_points()`
scans every `entry_points.txt` in site-packages.

The working interpreter is **`~/.venvs/cellsense/bin/python`** (~9s cold import of the
full stack). `docs/DEVELOPMENT.md` has the diagnosis recipe. `cellsense doctor` checks for
this automatically.

## Setup and running

```bash
source ~/.venvs/cellsense/bin/activate
pip install -e '.[all,dev]'
cp .env.example .env                    # at least one provider key
python scripts/make_sample_data.py      # regenerates data/ from a fixed seed

cellsense data/sales_2024.csv                                  # interactive REPL
cellsense data/sales_2024.csv -q "total revenue by region?"    # one-shot, pipeable
cellsense data/*.csv --model gemini:gemini-2.5-flash
cellsense doctor                                               # diagnose the environment
```

`main.py` still works but only prints a deprecation notice and delegates to the CLI.

## Architecture

`docs/ARCHITECTURE.md` is the **binding contract** — read it before changing any module
boundary. It pins the data model, the tool and provider contracts, the event protocol,
config precedence, the turn lifecycle, and the `Engine` API.

```
cellsense/
  cli.py            Typer app: ask (default), models, sessions, init, doctor
  config.py         CLI flag > env > ./.cellsense/config.toml > ~/.cellsense/... > defaults
  errors.py         CellSenseError hierarchy; every one carries an exit_code and a hint
  events.py         the engine <-> UI event protocol. SHARED CONTRACT — changing it is a
                    breaking change for both graph/ and ui/
  permissions.py    prompt | allow | deny policy, plus session "allow always"
  io/               loaders, Workspace/FileData/ToolResult/Citation, context digest
  tools/            one module per tool, each exposing SPEC: ToolSpec
  providers/        one module per provider, each exposing SPEC: ProviderSpec
  graph/            state, prompts, nodes, build, checkpoint, engine  (LangGraph lives here)
  ui/               prompt_toolkit + rich REPL, slash commands, @-mention picker, printer
  observability/    structured JSONL logging, token/cost accounting
```

**Import direction is enforced by a test** (`tests/unit/test_architecture.py`, AST-based):
`graph/` must not import `ui/`; `ui/` must not import `graph/` at runtime (only under
`TYPE_CHECKING`); `tools/` and `io/` must not import `providers/`, `graph/`, or `ui/`;
`events.py` imports nothing from cellsense.

**Turn lifecycle:** `guardrail → planner → (1 subtask ? agent : Send fan-out to N workers)
→ agent ⇄ tools → synthesize → END`. A single subtask bypasses `synthesize` for latency.
`Engine.stream_turn()` is the one seam the CLI and UI both code against; it yields typed
events and **always terminates in exactly one `TurnFinished` or one `TurnFailed`**.

## LangGraph specifics that bite

Verified against langgraph 1.2 / langchain-core 1.6 (much has moved since 0.2):

- The umbrella `langchain` package is **not** a dependency — only `langchain_core`.
- `SqliteSaver.from_conn_string(path)` is a **context manager**, not a constructor.
- `stream_mode` must be a **`list`**, not any sequence. `Pregel.stream()` tags chunks as
  `(mode, payload)` only under `isinstance(stream_mode, list)` (`pregel/main.py:4238`);
  a tuple falls through to yielding the bare payload, and `for mode, chunk in ...` then
  dies with `too many values to unpack`. This has already regressed once, from a mypy
  cleanup that turned the list into a typed tuple constant. Do not "tidy" it back.
- **Nodes re-execute from the top on resume.** So `interrupt()` must come before any side
  effect, and the engine de-duplicates custom events by `call_id`. Completed sibling
  branches replay as `{"__metadata__": {"cached": True}}` updates — skip them or results
  double-count.

## Working on this repo

```bash
ruff check . && ruff format --check . && mypy && pytest -q -m "not live"
pytest -m live        # real API calls; auto-skips when a key is absent
```

`pytest` treats `FutureWarning` as an error (pandas 3 is installed — its default text dtype
is `str`, not `object`). Keep the gate green; it is currently fully green.

**Adding a tool:** create `cellsense/tools/<name>.py` with `SPEC: ToolSpec`, add it to
`_MODULES` in `tools/registry.py`. `side_effect=True` routes it through the approval gate
automatically. Handlers must not mutate the caller's DataFrames and must raise `ToolError`
with a message the *model* can act on (list close-matching column names, not a bare KeyError).

**Adding a provider:** create `cellsense/providers/<name>.py` with `SPEC: ProviderSpec`,
import its `langchain_*` package **inside `build()`** (never at module scope) converting
`ImportError` to `MissingDependencyError`, register it, and add a pip extra.

## Things that are true and easy to get wrong

- **Model ids rot fast.** Three of this repo's original defaults died within months: Groq's
  `llama-3.3-70b-versatile` and Gemini's `gemini-2.0-flash` are retired (404), and Meta shut
  down its hosted Llama API on 2026-07-06. Curated model lists are metadata for `/model` and
  cost display, **never a whitelist** — an unrecognised id must still pass through.
- The `llama` provider now means **bring-your-own OpenAI-compatible host**; `LLAMA_BASE_URL`
  is required and model ids are host-specific.
- Default model is `groq:openai/gpt-oss-120b`. Groq and Gemini are verified working here;
  Anthropic and OpenAI are implemented but **unverified** (no keys on this machine).
- `data/` is gitignored and regenerated by `scripts/make_sample_data.py`. Tests must not
  depend on it — build fixtures programmatically, and derive expected values with pandas
  rather than hardcoding numbers.
- Citations are rendered by `io.schema.format_citations` and **nothing may fork a second
  formatter**. Past a threshold it prints an honest count (`[240 rows]`) instead of a
  truncated index list, and the model's own `Sources:` line is stripped in favour of the
  tool-derived one.
