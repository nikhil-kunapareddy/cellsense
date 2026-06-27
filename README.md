# CellSense

A terminal AI agent for asking natural-language questions about Excel and CSV files.
It plans queries, runs tools (filter, aggregate, join, plot, file discovery) over your
data, and answers with row-level citations.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add an API key for the backend you want (env var or a `.env` file in the project root):

| `--agent`           | Default model             | Key                 |
| ------------------- | ------------------------- | ------------------- |
| `llama` *(default)* | `Llama-3.3-70B-Instruct`  | `LLAMA_API_KEY`     |
| `groq`              | `llama-3.3-70b-versatile` | `GROQ_API_KEY`      |
| `gemini`            | `gemini-2.0-flash`        | `GEMINI_API_KEY`    |
| `claude`            | `claude-sonnet-4-5`       | `ANTHROPIC_API_KEY` |


Override a model with `LLAMA_MODEL` / `GROQ_MODEL` / `GEMINI_MODEL`.

## Usage

```bash
# Interactive session (one or more files)
python main.py sales.xlsx headcount.csv
python main.py data.csv --agent claude

# One-shot: answer a single question and exit
python main.py data.csv -q "total revenue by region?"

# No files: let the agent find them
python main.py -q "find sales files in ./data"
```

In the interactive session, ask questions in plain English. Slash commands:
`/help`, `/files`, `/clear`, `/exit`.

## Supported files

- `.xlsx` / `.xls` — Excel (multi-sheet)
- `.csv` — UTF-8 or latin-1

Requires Python 3.11+.

