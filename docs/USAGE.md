# Usage

This is the full command and flag reference for the `cellsense` CLI. For config file
keys and precedence, see [CONFIGURATION.md](CONFIGURATION.md). For module layout and the
event protocol, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Invocation

```
cellsense [FILES]... [OPTIONS]              # same as `cellsense ask [FILES]... [OPTIONS]`
cellsense ask [FILES]... [OPTIONS]
cellsense models
cellsense sessions
cellsense sessions rm <thread-id>
cellsense init [--force]
cellsense doctor
```

There is no subcommand you have to type to "ask a question" -- `ask` is the default
action. `cellsense data.csv -q "..."` and `cellsense ask data.csv -q "..."` are identical;
the bare form exists so `models`/`sessions`/`init`/`doctor` can still be typed directly.
`cellsense --help` and `cellsense -h` always show the top-level command list.

### `cellsense` / `cellsense ask`

```
Usage: cellsense ask [OPTIONS] [files]...

Arguments:
  files      CSV/XLSX files to load. Omit to let the agent find files itself.

Options:
  --query, -q <str>          Ask one question and exit.
  --model, -m <str>          Provider:model selector, e.g. groq:openai/gpt-oss-120b.
  --print, -p                 Non-interactive mode; reads the question from stdin if -q is absent.
  --verbose, -v                Show the tool trace in --print mode.
  --debug                       Debug-level logging and full tracebacks on error.
  --no-color                    Disable ANSI styling.
  --yes                         Auto-approve side-effect tools (permissions.mode=allow).
  --deny                        Auto-deny side-effect tools (permissions.mode=deny).
  --resume <str>                Resume a saved session by thread id.
  --session <str>                Use a specific thread id for this run.
  --config <path>                Load an additional TOML config file.
  --max-tool-rounds <int>        Override agent.max_tool_rounds.
  --version                      Print the version and exit.
  --help, -h                    Show this message and exit.
```

With `-q`/`--query`, CellSense answers once and exits with the resulting process code
(`0` success, the failed `CellSenseError`'s `exit_code` on a deliberate failure, `130` on
cancellation, `1` on anything unexpected). Without it, CellSense starts the interactive
REPL. **With no files at all**, both modes still start -- the agent has `find_files` and
`list_directory` tools it can call itself (e.g. `cellsense -q "find sales files in ./data"`).

File arguments are validated up front: a missing path, a directory passed where a file is
expected, or an extension outside `.csv`/`.xlsx`/`.xls` all produce one clean error message
(listing every problem found, not just the first) and exit code `4`, before any network
call is made.

`--yes` and `--deny` are mutually exclusive, as are `--resume` and `--session`; passing
both of either pair is a config error (exit code `2`). `--resume <id>` must name an id
`cellsense sessions` actually lists, or it fails the same way.

`--model`/`-m` accepts (see `docs/ARCHITECTURE.md` §4 for the underlying rule):

- `provider:model_id` -- explicit, e.g. `groq:openai/gpt-oss-120b`.
- a bare provider name, e.g. `groq` -- uses that provider's default model.
- a bare model id, if it's unambiguous across every provider's curated list.

An unrecognized `provider:model_id` still resolves (unknown pricing shows as `-`); an
unrecognized bare provider name is a config error listing the known providers.

### `cellsense models`

Prints every curated provider/model: selector, context window, per-million-token input
and output pricing (`-` where unknown), and whether that provider's API key is currently
set in the environment (`.env` in the current directory is loaded first). This table is
metadata for `/model` and cost display, not a whitelist -- any `provider:model_id` you
pass to `--model` still works even if it isn't listed here.

### `cellsense sessions`

Lists every saved session from `~/.cellsense/sessions.db`: thread id, created/updated
timestamps, message count, and the most recent question asked in that thread.

### `cellsense sessions rm <thread-id>`

Permanently deletes one saved session (all its checkpoints). Errors cleanly (exit code
`2`) if the id isn't a saved session.

### `cellsense init [--force]`

Writes a starter `.cellsense/config.toml` in the current directory with every key
commented. Refuses to clobber an existing file unless `--force` is passed.

### `cellsense doctor`

Diagnoses the local environment:

- Python version and interpreter path.
- Whether `./.cellsense/config.toml` / `~/.cellsense/config.toml` resolve cleanly.
- Which provider API keys are set (a masked prefix + length is shown -- never the key
  itself).
- Which optional provider packages (`langchain_anthropic`, `langchain_openai`, ...) are
  importable.
- Whether the session checkpoint DB's directory is writable.
- **The iCloud "dataless file" trap**: scans the active `sys.prefix` (never a repo-local
  `.venv/`) for iCloud-evicted placeholder files using `stat` metadata only (never reads
  file contents, so it can't trigger the hang it's checking for). See
  [DEVELOPMENT.md](DEVELOPMENT.md) for what this means if it finds any.

## Slash commands (interactive REPL only)

| Command | Args | Description |
| --- | --- | --- |
| `/help` | | List every slash command |
| `/files` | | List loaded files, sheets, row/column counts, and column names |
| `/model` | `[provider:model]` | List models (with key status), or switch the active model |
| `/cost` | | Token usage and cost totals for this session |
| `/permissions` | | Show the current permission mode and always-allowed tools |
| `/clear` | | Clear the screen and redraw the banner |
| `/sessions` | | List saved sessions |
| `/resume` | `[thread-id]` | List sessions, or switch this REPL to a saved thread |
| `/export` | `[path]` | Write the session transcript to a markdown file |
| `/exit` (or `/quit`) | | End the session |

`/model <selector>` builds a brand-new `Engine` for the new provider/model and swaps it
in for the rest of the session -- the model is otherwise fixed once the REPL starts,
because it's resolved at `Engine` construction. Conversation history for the current
thread id is unaffected (it lives in the checkpointer, keyed by thread id, independent of
which model answers the next turn).

## `@`-mention: attaching a file mid-conversation

Type `@` anywhere in a question to open a fuzzy completion menu over CSV/XLSX files under
the current directory (up to 6 levels deep, skipping `.git`, `venv`, `node_modules`, and
similar). Selecting one attaches it to the live workspace before the question is sent --
so `what's the total in @sales_q2.csv?` loads `sales_q2.csv` and then answers using it,
in one line. Attaching a file that fails to load (bad extension, unreadable, empty) prints
a clean error and leaves the rest of the workspace untouched.

## Approval prompts

`plot` is currently the only tool with `side_effect=True` (it writes a PNG under
`./output/`). Whenever the active permission mode is `"prompt"` (the default) and the
tool isn't already on the always-allow list, the turn pauses and prints what's being
requested and why, then asks:

```
Approve? [y]es / [a]lways / [N]o:
```

- **yes** runs it once.
- **always** runs it and adds it to this *session's* allow-list (not written back to any
  config file) so it won't ask again this session.
- **no** (or anything else, including plain Enter) denies it; the model is told the user
  declined and continues without that tool's result.

Set `--yes` to auto-approve every side-effect tool for the whole invocation, or `--deny`
to auto-decline them -- both bypass the prompt entirely (they set
`permissions.mode` to `"allow"`/`"deny"`, at which point the graph never even pauses to
ask). `--print` mode with neither flag auto-denies side-effect tools by default, since
there's no human to ask in a non-interactive pipeline.

## Sessions and resume

Every turn is checkpointed to a SQLite database at `~/.cellsense/sessions.db`, keyed by a
`thread_id`. Three ways to control which thread a run uses:

- Nothing passed: a fresh random thread id per invocation.
- `--session <id>`: use (creating if needed) a specific thread id -- handy for scripting a
  multi-step conversation across several `--print` invocations.
- `--resume <id>`: continue a thread that already has at least one saved turn; the id must
  appear in `cellsense sessions`.

Inside the REPL, `/resume [thread-id]` does the same thing without restarting the process.

## Piping and `--print` recipes

`--print`/`-p` (or just `-q`) renders only the final answer plus a citation line to
**stdout**, with ANSI color automatically disabled whenever stdout isn't a terminal --
piping through `| cat`, redirecting to a file, or capturing in a script all produce plain
text and a clean exit code. Everything else (the tool trace, notices, approval prompts)
goes to **stderr**, and only if `--verbose` is set.

```bash
# One-shot, answer to stdout, tool trace and diagnostics to stderr if -v is set
cellsense data/sales_2024.csv -q "total revenue by region?" > answer.txt

# Question from a pipe instead of -q
echo "total revenue by region?" | cellsense data/sales_2024.csv --print

# Script a multi-turn conversation against one saved session
cellsense data.csv -q "total revenue?" --session report --yes
cellsense data.csv -q "and by region?" --session report --yes

# Non-interactive, unattended run that also lets `plot` write its PNG
cellsense data.csv -q "plot revenue by month" --yes --verbose 2>trace.log
```

`cellsense --print` with no `-q` and stdin attached to a real terminal (nothing piped in)
is a config error asking you to supply one or the other -- there's nothing to read.

## Old `main.py` compatibility

`python main.py ...` still works: it prints a deprecation notice to stderr, maps the old
`--agent {claude,llama,groq,gemini}` flag onto the new `--model` selector (`claude` ->
`anthropic`; the other three keep their names), passes every other flag through unchanged,
and delegates to the real `cellsense` entry point -- including its exit code.
