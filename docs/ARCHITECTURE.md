# CellSense Architecture

CellSense is an **agent harness for tabular data**. You point it at CSV/XLSX files and
ask questions in natural language; a LangGraph agent plans, calls typed pandas tools
against the live DataFrames, and answers with row-level citations.

The design goal is the same one Claude Code has: the *harness* is the product. Models
are swappable, tools are pluggable, every turn is observable, resumable, and gated.

---

## 1. Package layout

```
cellsense/
  cli.py              Typer app: argument parsing, mode dispatch, top-level error rendering
  config.py           Settings resolution: CLI flags > env > .cellsense/config.toml > defaults
  errors.py           Exception hierarchy. Every deliberate failure subclasses CellSenseError
  events.py           The engine <-> UI event protocol (dataclasses). SHARED CONTRACT
  permissions.py      Permission policy: which tools need approval, and remembered decisions

  io/                 Everything about *data*, nothing about models
    loaders.py        load_workspace() -> Workspace; CSV + multi-sheet XLSX
    schema.py         Workspace, FileData, TableRef, ToolResult, Citation
    context.py        build_context(workspace) -> compressed schema digest for the prompt
    classify.py       heuristic business-category label for a table

  tools/              One module per tool. Pure functions over DataFrames
    base.py           ToolSpec dataclass + @tool_spec decorator + side_effect flag
    registry.py       ToolRegistry: discovery, JSON schemas, LangChain StructuredTool export
    <tool>.py         aggregate, filter_rows, join, describe, plot, list_directory, find_files

  providers/          Everything about *models*, nothing about data
    base.py           ProviderSpec, ModelSpec, build_chat_model()
    registry.py       PROVIDERS map, resolve_model(), available_providers(), estimate_cost()
    anthropic.py openai.py gemini.py groq.py llama.py

  graph/              LangGraph orchestration
    state.py          TurnState TypedDict + reducers
    nodes.py          guardrail, planner, agent, tools, synthesize
    build.py          build_graph() -> CompiledStateGraph
    checkpoint.py     SQLite checkpointer, session ids, list/resume/delete
    engine.py         Engine.stream_turn() -> Iterator[Event]   PUBLIC ENTRY POINT

  ui/                 Terminal front-end (prompt_toolkit + rich)
    repl.py           interactive loop, key bindings, Ctrl-C interrupt, session banner
    slash.py          slash command table + prompt_toolkit Completer
    filepicker.py     @-mention fuzzy file attach
    render.py         Event -> rich renderable
    stream.py         live status line: spinner, elapsed, tokens, cost
    theme.py          colour tokens, single source of styling truth

  observability/
    logging.py        structured JSON logs to ~/.cellsense/logs, --debug trace
    usage.py          token + cost accounting per turn and per session
```

### Import direction (enforced by a test)

```
cli  ->  ui  ->  events  <-  graph  ->  providers
                   ^           |
                   |           v
               observability  tools  ->  io
```

* `graph/` **must not** import from `ui/`.
* `ui/` **must not** import from `graph/` except the `Engine` type.
* `tools/` and `io/` **must not** import from `providers/`, `graph/`, or `ui/`.
* `events.py` imports nothing from CellSense.

---

## 2. Data model (`io/schema.py`)

```python
@dataclass(frozen=True)
class TableRef:
    """Addresses exactly one DataFrame in the workspace."""
    filename: str
    sheet: str | None          # None for CSV; sheet name for Excel
    @property
    def label(self) -> str     # "sales.xlsx[Q1]" or "financials.csv"

@dataclass
class FileData:
    path: Path
    sheets: dict[str, pd.DataFrame]   # CSV uses the single key "default"
    file_type: Literal["csv", "excel"]
    filename: str  (property)         # path.name
    total_rows: int  (property)

class Workspace:
    """The set of loaded files. Owns all DataFrames; tools receive it read-only."""
    files: dict[str, FileData]
    def table(self, filename: str, sheet: str | None = None) -> pd.DataFrame
    def resolve(self, filename: str | None, sheet: str | None) -> tuple[TableRef, pd.DataFrame]
    def add(self, path: Path) -> FileData      # for @-mention attach at runtime
    def refs(self) -> list[TableRef]
    def is_empty(self) -> bool

@dataclass(frozen=True)
class Citation:
    filename: str
    sheet: str | None
    rows: tuple[int, ...]        # original source row indices, NOT positional

@dataclass
class ToolResult:
    data: pd.DataFrame
    summary: str                 # one line, human readable, shown in the tool trace
    citations: list[Citation]
    def to_text(self, max_rows: int = 20) -> str    # what the model sees
    def preview(self, n: int = 5) -> list[dict]     # what the UI renders
```

**Citation rule:** `rows` are indices into the *source* DataFrame. Any tool that
reshapes data (aggregate, join) must map back to the contributing source rows, not
emit indices into its own output.

**Citation rendering rule:** `format_citations` is the single renderer — nothing may
fork a second format. Consecutive indices collapse into ranges (`Rows: 3-9`), and past
a threshold the enumeration is replaced by a count (`sales.csv [240 rows]`). Printing
the first 50 of 240 contributing indices reads as a precise claim about 50 specific
rows, which is false; a count is honest. The engine's computed citations are
**authoritative**, so any `Sources:` line the model writes into its own answer is
stripped before `TurnFinished` — otherwise the user sees two disagreeing blocks.

---

## 3. Tool contract (`tools/base.py`)

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON Schema, draft-07 subset
    handler: Callable[[dict, Workspace], ToolResult]
    side_effect: bool = False  # True => routed through permissions.py before running
    reads_filesystem: bool = False
```

Each tool module exposes a module-level `SPEC: ToolSpec`. `registry.py` discovers
modules listed in `_MODULES`, validates that names are unique and schemas parse, and
exports:

* `ToolRegistry.specs()` -> `list[ToolSpec]`
* `ToolRegistry.langchain_tools()` -> `list[StructuredTool]` for `model.bind_tools(...)`
* `ToolRegistry.run(name, args, workspace)` -> `ToolResult` (raises `ToolError`)

Handlers raise `ToolError` for user-fixable problems (bad column name). The graph
converts that into a `ToolMessage` so the model can retry with different arguments —
a tool failure is never fatal to the turn.

`side_effect=True` today: `plot` (writes a PNG). `reads_filesystem=True`:
`list_directory`, `find_files`.

---

## 4. Provider contract (`providers/base.py`)

```python
@dataclass(frozen=True)
class ModelSpec:
    id: str                       # provider-native model id, sent over the wire
    provider: str
    display: str
    context_window: int
    max_output_tokens: int
    input_usd_per_mtok: float | None    # None = unknown price, cost shown as "-"
    output_usd_per_mtok: float | None
    supports_tools: bool = True
    supports_streaming: bool = True

@dataclass(frozen=True)
class ProviderSpec:
    name: str                     # "anthropic" | "openai" | "gemini" | "groq" | "llama"
    label: str                    # "Anthropic"
    env_key: str                  # "ANTHROPIC_API_KEY"
    package: str                  # "langchain_anthropic"  (imported lazily)
    extra: str                    # pip extra name, used in the install hint
    default_model: str
    models: tuple[ModelSpec, ...]
    def build(self, model_id: str, settings: ModelSettings) -> BaseChatModel
```

Rules:

1. **Lazy imports.** A provider module must not import its `langchain_*` package at
   module scope. Import inside `build()` and raise `MissingDependencyError` on
   `ImportError`. This keeps `pip install cellsense` light and lets one missing extra
   fail loudly for that provider only.
2. **Unknown model ids must still work.** `resolve_model("groq:some-new-model")`
   returns a synthesized `ModelSpec` with `None` pricing rather than raising. The
   curated `models` tuple is metadata for `/model` and cost display, not a whitelist.
3. **Selector syntax.** `--model` accepts `provider:model_id`, a bare provider name
   (uses `default_model`), or a bare model id (resolved against curated lists; ambiguous
   ids raise `ConfigError` listing the candidates).
4. **Uniform errors.** `build()` and call sites translate provider SDK exceptions into
   `AuthenticationError` / `RateLimitError` / `ProviderError`.
5. Retries/timeouts come from `ModelSettings` and are passed to the LangChain
   constructor (`max_retries`, `timeout`) — do not hand-roll a retry loop.

---

## 5. Config (`config.py`)

Precedence, highest first: **CLI flag > environment variable > `./.cellsense/config.toml`
> `~/.cellsense/config.toml` > built-in default.**

```toml
[model]
default = "groq:llama-3.3-70b-versatile"
temperature = 0.0
max_tokens = 4096
timeout_s = 120
max_retries = 3

[agent]
max_tool_rounds = 8
max_subtasks = 4
planner = true          # false => always single-shot, skips the planner call
guardrail = true

[permissions]
mode = "prompt"         # "prompt" | "allow" | "deny"
allow = ["aggregate", "filter_rows", "join", "describe"]

[ui]
theme = "auto"
show_tool_args = true

[telemetry]
log_dir = "~/.cellsense/logs"
level = "info"
```

`.env` is loaded from the working directory at startup (existing behaviour, keep it).

---

## 6. Turn lifecycle

```
                    +-------------+
  question  ---->   |  guardrail  |---- off-topic ----> TurnFinished(rejection)
                    +------+------+
                           | relevant
                    +------v------+
                    |   planner   |   1 call, JSON list of independent subtasks
                    +------+------+
                     1 task |  N tasks
             +--------------+--------------+
             |                             | Send() fan-out
      +------v------+              +-------v-------+
      |    agent    |<--+          |  worker (xN)  |  each an independent agent loop
      +------+------+   |          +-------+-------+
             |          |                  |
      +------v------+   |          +-------v-------+
      |    tools    |---+          |   synthesize  |  1 call, merges + merges citations
      +-------------+              +-------+-------+
             |                             |
             +-------------> END <---------+
```

* The `agent` <-> `tools` cycle is capped at `max_tool_rounds`; exceeding it emits a
  `Notice` and forces a final answer rather than erroring.
* `tools` calls `interrupt()` before any `side_effect=True` tool when the permission
  mode is `prompt` and the tool is not already allow-listed.
* Every node writes progress through `get_stream_writer()` so the UI sees tool traces
  from fan-out workers, not just the main thread.

---

## 7. Engine API (`graph/engine.py`) — the seam the UI codes against

```python
class Engine:
    def __init__(self, workspace: Workspace, config: Config, *, registry: ToolRegistry) -> None

    @property
    def model_label(self) -> str          # "llama-3.3-70b-versatile"
    @property
    def provider_label(self) -> str       # "Groq"

    def stream_turn(
        self,
        question: str,
        *,
        thread_id: str,
        on_approval: Callable[[ApprovalRequested], Literal["allow","allow_always","deny"]],
        cancel: threading.Event | None = None,
    ) -> Iterator[Event]: ...

    def sessions(self) -> list[SessionInfo]
    def history(self, thread_id: str) -> list[tuple[str, str]]   # [(role, text)]
```

`SessionInfo` is a **pinned structural contract** — the UI reads exactly these four
names, so `graph.checkpoint.SessionInfo` must expose them (extra fields are fine):

```python
thread_id: str  # the checkpointer key, and what --resume takes
updated_at: str  # ISO 8601 of the most recent checkpoint
preview: str  # first user question, truncated for a list column
turns: int  # number of completed user/assistant exchanges
```

Guarantees:

* The iterator always terminates with exactly one `TurnFinished` **or** one `TurnFailed`.
* Setting `cancel` mid-stream stops at the next node boundary and yields
  `TurnFailed(cancelled=True)`. It never leaves a corrupt checkpoint.
* Conversation history lives in the LangGraph checkpointer keyed by `thread_id`. The UI
  keeps no message list of its own.

---

## 8. Non-goals

* No local/offline model support — API providers only, by design.
* No SQL engine. pandas is the execution substrate.
* No write-back to source spreadsheets. CellSense reads; the only artefact it writes is
  a plot PNG, and that is permission-gated.
