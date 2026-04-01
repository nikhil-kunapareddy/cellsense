"""Visualization tool for CellSense.

Generates charts from loaded file data and saves them as PNG files under
the output/ directory. The agent calls this tool the same way it calls
filter_rows or aggregate — by specifying the file, columns, and chart type.
"""
from __future__ import annotations

import matplotlib
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from src.data import FileData
from src.tools import ToolResult, _get_sheet
from src.utils.citations import Citation

matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt  # noqa: E402

OUTPUT_DIR = Path("output")

# ── Schema ─────────────────────────────────────────────────────────────────────

PLOT_SCHEMA: Dict[str, Any] = {
    "name": "plot",
    "description": (
        "Generate a chart from file data and save it as a PNG. "
        "Optionally filter rows before plotting, or group and aggregate "
        "the data first (e.g. sum revenue by region). "
        "Supported chart types: bar, line, scatter, histogram, pie."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Exact filename as provided at startup.",
            },
            "sheet_name": {
                "type": "string",
                "description": "Sheet name for Excel; use 'default' for CSV.",
            },
            "chart_type": {
                "type": "string",
                "enum": ["bar", "line", "scatter", "histogram", "pie"],
            },
            "x_column": {
                "type": "string",
                "description": "Column for the x-axis (or category labels for bar/pie).",
            },
            "y_column": {
                "type": "string",
                "description": (
                    "Column for the y-axis (or values for bar/pie). "
                    "Not required for histogram."
                ),
            },
            "query": {
                "type": "string",
                "description": "Optional pandas query string to filter rows before plotting.",
            },
            "group_by": {
                "type": "string",
                "description": "Optional column to group by before plotting (e.g. 'Region').",
            },
            "agg_func": {
                "type": "string",
                "enum": ["sum", "mean", "count", "max", "min"],
                "description": "Aggregation to apply when group_by is set. Defaults to 'sum'.",
            },
            "title": {
                "type": "string",
                "description": "Chart title. Auto-generated if omitted.",
            },
        },
        "required": ["filename", "sheet_name", "chart_type", "x_column"],
    },
}


# ── Handler ────────────────────────────────────────────────────────────────────

def handle_plot(inputs: Dict[str, Any], file_data: Dict[str, FileData]) -> ToolResult:
    filename = inputs["filename"]
    sheet_name = inputs["sheet_name"]
    chart_type = inputs["chart_type"]
    x_col = inputs["x_column"]
    y_col = inputs.get("y_column")
    query = inputs.get("query")
    group_by = inputs.get("group_by")
    agg_func = inputs.get("agg_func", "sum")
    title = inputs.get("title")

    df = _get_sheet(file_data, filename, sheet_name)

    # Optional filter
    if query:
        df = df.query(query)

    # Optional groupby aggregation
    if group_by and y_col:
        df = df.groupby(group_by)[y_col].agg(agg_func).reset_index()
        df.columns = [group_by, f"{agg_func}_{y_col}"]
        x_col = group_by
        y_col = f"{agg_func}_{y_col}"

    _validate_columns(df, x_col, y_col, chart_type)

    # Build chart
    fig, ax = plt.subplots(figsize=(8, 5))
    _draw(ax, df, chart_type, x_col, y_col)

    chart_title = title or _auto_title(chart_type, x_col, y_col, filename)
    ax.set_title(chart_title, fontsize=13, pad=12)
    fig.tight_layout()

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"plot_{timestamp}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    citations = [Citation(
        filename=filename,
        sheet_name=None if sheet_name == "default" else sheet_name,
        row_indices=list(df.index),
    )]
    summary = f"Chart saved to {out_path}  ({chart_type}, {len(df)} data points)"
    return ToolResult(data=df, citations=citations, summary=summary)


# ── Drawing helpers ────────────────────────────────────────────────────────────

def _draw(
    ax: plt.Axes,
    df: pd.DataFrame,
    chart_type: str,
    x_col: str,
    y_col: str | None,
) -> None:
    if chart_type == "bar":
        ax.bar(df[x_col].astype(str), df[y_col])
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        plt.xticks(rotation=30, ha="right")

    elif chart_type == "line":
        ax.plot(df[x_col], df[y_col], marker="o")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

    elif chart_type == "scatter":
        ax.scatter(df[x_col], df[y_col], alpha=0.7)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

    elif chart_type == "histogram":
        ax.hist(df[x_col], bins="auto", edgecolor="white")
        ax.set_xlabel(x_col)
        ax.set_ylabel("Count")

    elif chart_type == "pie":
        ax.pie(
            df[y_col],
            labels=df[x_col].astype(str),
            autopct="%1.1f%%",
            startangle=140,
        )
        ax.axis("equal")


def _validate_columns(
    df: pd.DataFrame,
    x_col: str,
    y_col: str | None,
    chart_type: str,
) -> None:
    if x_col not in df.columns:
        raise ValueError(f"Column '{x_col}' not found. Available: {list(df.columns)}")
    if chart_type != "histogram" and y_col and y_col not in df.columns:
        raise ValueError(f"Column '{y_col}' not found. Available: {list(df.columns)}")


def _auto_title(chart_type: str, x_col: str, y_col: str | None, filename: str) -> str:
    base = filename.rsplit(".", 1)[0]
    if y_col:
        return f"{y_col} by {x_col} ({base})"
    return f"{x_col} distribution ({base})"
