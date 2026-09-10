# TODO

State as of 2026-09-09, after the rewrite from the prototype into a packaged LangGraph
harness. Ordered by what actually blocks value, not by effort.

Current baseline: 48 modules / 8.9k lines, 534 tests, 85% coverage, `ruff` + `mypy` clean.

---

## Blocked on you

- [ ] **Commit the rewrite.** 33 changed paths are sitting uncommitted in the working
      tree on `dev`. Nothing here is committed yet.
- [ ] **Decide the Llama story.** Meta retired the hosted API on 2026-07-06, so the
      provider is now bring-your-own host and `LLAMA_BASE_URL` is unset here — meaning it
      is shipped but has never once been executed. Either point it at a real host
      (Together / Bedrock / Fireworks / vLLM) and let the `live` tests cover it, or drop
      the provider. Leaving it in the middle is the worst option.
- [ ] **Add an `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY`.** Both providers are
      implemented but have **never been called**. The README says so honestly, but two of
      the five advertised providers being unverified is a real gap.
- [ ] **Re-run the cross-file join live.** The one live case that failed
      (`"which sales rep earned the most, and what department are they in?"`) failed
      because of the `filter_rows` condition bug, which is now fixed and unit-tested — but
      both providers hit their daily quota before I could re-prove it end-to-end.
      Command is in the session notes; it is a single Groq call.

---

## Known limitations (real, reproducible)

- [ ] **Context is built once per turn.** `engine.py:199` calls `build_context()` at turn
      start, so a file discovered by `find_files` mid-turn is queryable but **invisible in
      the model's schema digest** for the rest of that turn. Either rebuild context after a
      workspace mutation or make `find_files` return the new schema in its `ToolResult`.
- [ ] **~6k input tokens per simple question.** Measured breakdown: ~780 system + context,
      ~790 for the 7 tool schemas, rest transcript. On Groq's free tier (8k TPM) that means
      a two-round tool loop can rate-limit itself. Worth trimming tool descriptions and
      making `build_context` adaptive to the question.
- [ ] **No prompt caching.** Anthropic and Gemini both support it and the system prompt +
      schema block is identical across every turn of a session — this is the single
      biggest cost lever available and it is entirely unexploited.
- [ ] **Fully synchronous.** Fan-out workers use threads. Fine today, but streaming +
      concurrency would be cleaner on `astream`, and LangGraph supports it natively.
- [ ] **`UserWarning` from pandas on date inference** (`io/loaders.py:175`): `to_datetime`
      without an explicit format falls back to `dateutil` per element. Harmless and only
      hit on the sample path, but it is noise in the test output and slow on wide frames.
- [ ] **Citations are row-level, not cell-level.** `aggregate` cites every contributing
      row, which is honest but coarse — `[240 rows]` is less useful than naming the
      column(s) that produced the number.

---

## Test gaps

Coverage is 85% overall, but it is very unevenly distributed. The interactive surface is
the least-tested part of the codebase, and it is the part users touch first.

- [ ] **`ui/stream.py` — 44%** (up from 16% after the generator-lifecycle tests). Still
      uncovered: the `rich.live.Live` render loop, the spinner, and **the Esc/Ctrl-C
      cancellation path**. Cancellation is a correctness feature, not a cosmetic one, and
      the keyboard half of it is still untested.
- [ ] **`ui/repl.py` — 44%.** The REPL loop, key bindings, and banner.
- [ ] **`cli.py` — 71%.** (`ui/printer.py` is now 82%.)
- [ ] **No end-to-end test drives a real terminal.** Consider `pyte` or
      `prompt_toolkit`'s pipe input to script a full interactive session and assert on the
      rendered screen.
- [ ] **No provider-contract test.** Each provider is checked for lazy imports and key
      handling, but nothing asserts they all behave identically given the same tool call.
      A shared parametrised suite (skipped without keys) would catch provider drift — which
      bit us three times already in one day.
- [ ] Coverage is not enforced in CI. Add a floor (say 80%) so it can only go up.

---

## Features worth building

**Near term**
- [ ] **Restore the MCP server.** The prototype had `src/mcp/server.py`, a stdio MCP server
      exposing `list_directory` / `find_files` to Claude Desktop. It was deleted with the
      rest of `src/` and has **no replacement** in the new package. The tool layer is
      cleanly separable now, so re-exposing the whole registry over MCP is a small job and
      strictly more useful than the original two-tool version.
- [ ] **`export` / `write` tools** — save a filtered or aggregated result to CSV/XLSX.
      The approval-gate machinery already exists and is currently used by exactly one tool
      (`plot`), so this is nearly free and is the most obvious missing verb.
- [ ] **A `sql` tool** backed by DuckDB over the loaded frames. Models write good SQL, and
      it collapses many multi-step tool loops into one call — directly attacking the token
      and latency problem above.
- [ ] **`/undo` and `/retry`** in the REPL. The checkpointer already stores every step, so
      rewinding a turn is mostly UI work.

**Medium term**
- [ ] **Column-level lineage** so citations can say *which* columns produced a number.
- [ ] **Charts in the terminal** (sparklines / unicode plots) instead of only writing a PNG.
- [ ] **A `--json` output mode** so CellSense composes into scripts, not just pipes.
- [ ] **Multi-file globbing and lazy loading.** Everything is read eagerly into pandas at
      startup; a directory of large CSVs will be slow and memory-hungry.
- [ ] **Streaming/chunked reads for large files.** There is currently no guard at all
      against opening a file larger than RAM.

**Longer term**
- [ ] **Evals.** A fixed question set with known answers, run across providers to measure
      accuracy and cost per model. This repo is one live run away from being able to say
      "gpt-oss-120b gets 9/10 at $0.004/turn" — that is the number that should drive the
      default model choice, rather than my judgement.
- [ ] **Publish to PyPI.** `pyproject.toml` is already complete enough; needs a release
      workflow and a version-bump policy.
- [ ] **Plugin tools** — let users drop a tool module into `~/.cellsense/tools/` and have
      the registry discover it.

---

## Hygiene

- [ ] `assets/logo.png` is 7.2 MB and unused now that `banner.jpg` (81 KB) exists. It is
      already in git history, so deleting it saves the working tree but not the clone.
- [ ] `.github/workflows/ci.yml` has never actually run — no push has happened yet. Expect
      to fix something on the first green build.
- [ ] `CHANGELOG.md` does not exist. Worth starting now, at 0.2.0, rather than
      reconstructing it later.
- [ ] Consider a `pre-commit` config so `ruff`/`mypy` run before the commit rather than in
      CI.

---

## Decisions to revisit

- **Model ids rot fast.** Three defaults died within months (Groq `llama-3.3-70b-versatile`,
  Gemini `gemini-2.0-flash`, and Meta's entire hosted API). Curated lists are metadata, not
  whitelists — keep it that way, and re-check the pricing table periodically. Every price
  in `providers/*.py` carries a source URL and the date it was checked.
- **The planner is on by default.** It costs an extra LLM call on every turn to decide
  whether to decompose, and most real questions are single-task. Measure whether it earns
  its keep, or make it conditional on question complexity.
- **The guardrail is also an extra call per turn** and fails open by design. Same question.
- **Keep virtualenvs outside `~/Documents`.** iCloud eviction makes an in-repo `.venv`
  permanently unreadable; see `docs/DEVELOPMENT.md`. `cellsense doctor` checks for it.
