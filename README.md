# CellSense

A terminal AI agent for asking natural language questions about Excel and CSV files.

## Setup

```bash
# Clone and enter the directory
cd cellsense

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Choosing an agent

CellSense supports two LLM backends, selected with the `--agent` flag.

### Claude (default)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python main.py sales.xlsx
```

### Meta Llama API

```bash
export LLAMA_API_KEY=<your-meta-llama-api-key>
python main.py sales.xlsx --agent llama
```

The default model is `Llama-4-Maverick-17B-128E-Instruct-FP8`. Override with:

```bash
export LLAMA_MODEL=Llama-3.3-70B-Instruct
```

## Usage

Pass one or more Excel or CSV files as arguments:

```bash
python main.py sales.xlsx
python main.py sales.xlsx headcount.csv --agent llama
python main.py data.csv
```

CellSense will load the files and drop into an interactive session:

```
cellsense > What is total revenue by region?
cellsense > Which products have the highest return rate?
cellsense > Join sales and headcount by department and show average revenue per employee
```

### Example: Financial report

```bash
python main.py Financial_Report.xlsx --agent llama
```

```
cellsense > What is the total revenue?
# The total revenue for the 6 months ended March 29, 2025 is $219,659 million.
# Sources: Financial_Report.xlsx [Sheet: CONDENSED CONSOLIDATED STATEMEN, Rows: 1, 20, 23]

cellsense > What are the main segments and their revenue breakdown?
# Segment 1: $92,963 million
# Segment 2: $58,315 million
# Segment 3: $34,515 million
# Segment 4: $16,285 million
# Segment 5: $17,581 million
# Sources: Financial_Report.xlsx [Sheet: Segment Information and Geogr_3, Rows: 2, 6, 10, 14, 18, 22]
```

### Slash commands

| Command | Description |
|---|---|
| `/help` | Show available commands |
| `/files` | List loaded files, sheets, and columns |
| `/clear` | Clear the screen |
| `/exit` | End the session |

## Supported file types

- `.xlsx` / `.xls` — Excel (multi-sheet supported)
- `.csv` — comma-separated values (UTF-8 and latin-1 encoding)

## Requirements

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/) and/or a [Meta Llama API key](https://llama.meta.com/)
