# Configuration

CellSense resolves settings once per invocation, in `cellsense/config.py`'s
`load_config()`. This is the single source of truth for precedence and defaults; the CLI
(`cellsense/cli.py`) never re-implements it, only translates flags into the same shape
`load_config` already understands.

## Precedence

Highest wins, in this order:

1. **CLI flag** (only flags the user actually passed -- an absent flag never overrides
   a lower layer with `None`).
2. **Environment variable** (`CELLSENSE_<SECTION>_<KEY>`, see below).
3. **`./.cellsense/config.toml`** (current working directory).
4. **`~/.cellsense/config.toml`** (home directory).
5. **Built-in default** (the values shown below).

`--config <path>` (a CellSense-CLI-level flag, not part of `load_config` itself) loads an
*additional* TOML file and folds its contents in at CLI-flag precedence, but underneath
any individual flag also passed on the same command line -- so `--config extra.toml
--model groq:foo` uses `--model`'s value even if `extra.toml` also sets `[model].default`.

A `.env` file in the current working directory is always loaded first (via
`python-dotenv`), so provider API keys and any `CELLSENSE_*` environment override placed
there take effect without exporting them into the shell.

## Config file location

`cellsense init` writes `./.cellsense/config.toml` (refusing to overwrite one without
`--force`). `~/.cellsense/config.toml` is never created for you -- if you want a personal
default that applies everywhere, create it yourself.

## Every key

### `[model]`

| Key | Type | Default | Effect |
| --- | --- | --- | --- |
| `default` | string | `"groq:openai/gpt-oss-120b"` | The `--model` selector used when `--model` isn't passed on the command line. Same syntax as `--model` (see `docs/USAGE.md`). |
| `temperature` | float | `0.0` | Passed straight through to the LangChain chat model constructor. |
| `max_tokens` | int | `4096` | Max output tokens per model call. |
| `timeout_s` | int | `120` | Per-call timeout, in seconds. |
| `max_retries` | int | `3` | Provider-SDK-level retry count (CellSense never hand-rolls its own retry loop). |
| `streaming` | bool | `false` | Whether the chat model streams tokens. Not exposed via `--config`-free CLI flags today; set it in a TOML file or `CELLSENSE_MODEL_STREAMING=true`. |

### `[agent]`

| Key | Type | Default | Effect |
| --- | --- | --- | --- |
| `max_tool_rounds` | int | `8` | Cap on agent<->tools round trips per turn (per sub-task, for fan-out). Overridable with `--max-tool-rounds`. Exceeding it emits a notice and forces a final answer rather than erroring. |
| `max_subtasks` | int | `4` | Cap on how many independent sub-tasks the planner may decompose one question into. |
| `planner` | bool | `true` | `false` skips the planner call entirely -- every question is answered single-shot, never fanned out. |
| `guardrail` | bool | `true` | `false` skips the relevance guardrail -- every question reaches the tool-calling agent, even off-topic ones. |

### `[permissions]`

| Key | Type | Default | Effect |
| --- | --- | --- | --- |
| `mode` | `"prompt"` \| `"allow"` \| `"deny"` | `"prompt"` | How side-effecting tools (currently just `plot`) are gated. `"prompt"` asks interactively unless the tool is in `allow`; `"allow"` never asks; `"deny"` never runs them. `--yes`/`--deny` set this to `"allow"`/`"deny"` for the whole invocation. |
| `allow` | list of tool names | `["aggregate", "filter_rows", "join", "describe"]` | Tools that never need a prompt even in `"prompt"` mode. Note: none of these four are actually side-effecting today (only `plot` is), so this list only matters once a future tool sets `side_effect=True`, or if you add `"plot"` to it yourself. |

### `[ui]`

| Key | Type | Default | Effect |
| --- | --- | --- | --- |
| `theme` | string | `"auto"` | Passed as the REPL's theme mode. `--no-color` overrides this to no-color for that invocation regardless of what's configured. |
| `show_tool_args` | bool | `true` | Whether the live tool trace shows a tool's arguments (`aggregate(aggregations=...)`) or just its name. `--verbose` forces this on for the current invocation even if the config sets it to `false`. |

### `[telemetry]`

| Key | Type | Default | Effect |
| --- | --- | --- | --- |
| `log_dir` | string (path, `~` expanded) | `"~/.cellsense/logs"` | Directory for structured JSON-lines logs, one file per day (`cellsense-YYYY-MM-DD.jsonl`). |
| `level` | string | `"info"` | Log level for the file handler. `--debug` forces `DEBUG` regardless of this setting, and also mirrors debug-level logs to stderr. |

An unknown key anywhere in a TOML file or in `--config`'s file is a config error that
names the closest valid key it could find (e.g. `[modle]` -> "Did you mean 'model'?").

## Environment variable overrides

Any key above can also be set as `CELLSENSE_<SECTION>_<KEY>` (both upper-cased), e.g.:

```bash
CELLSENSE_MODEL_DEFAULT=gemini:gemini-2.5-flash
CELLSENSE_AGENT_MAX_TOOL_ROUNDS=4
CELLSENSE_PERMISSIONS_MODE=allow
CELLSENSE_PERMISSIONS_ALLOW=aggregate,filter_rows,plot   # comma-separated for list keys
CELLSENSE_UI_SHOW_TOOL_ARGS=false
```

Values are coerced to match the built-in default's type (booleans accept
`1`/`true`/`yes`/`on`, case-insensitively; lists split on commas and strip whitespace).

## Provider API keys

These are read directly from the environment (or `.env`), not from any `[section]` in
`config.toml` -- there is no way to put an API key in a TOML file, deliberately, so a
config file can be committed or shared without leaking a key.

| Env var | Provider |
| --- | --- |
| `ANTHROPIC_API_KEY` | Anthropic |
| `OPENAI_API_KEY` | OpenAI |
| `GEMINI_API_KEY` | Google Gemini |
| `GROQ_API_KEY` | Groq |
| `LLAMA_API_KEY` + `LLAMA_BASE_URL` | Llama (bring-your-own OpenAI-compatible host -- both are required; there is no default host since Meta retired its first-party hosted API) |

`cellsense doctor` reports which of these are currently set (masked) and which optional
provider packages are importable.

## The session database

`~/.cellsense/sessions.db` (SQLite) is fixed and is not currently a config key -- every
`Engine` instance opens the same path regardless of `--config`/TOML settings.

## Example: a project-local config

```toml
# ./.cellsense/config.toml
[model]
default = "gemini:gemini-2.5-flash"
temperature = 0.2

[agent]
max_tool_rounds = 5
max_subtasks = 2

[permissions]
mode = "allow"   # trusted CI environment; auto-approve plot

[telemetry]
level = "debug"
```
