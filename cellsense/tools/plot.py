"""plot: render a chart to a PNG under ./output/.

``side_effect=True`` -- this is the one tool that writes to disk, so the
graph routes it through the permission gate before running it. matplotlib is
forced onto the Agg backend before ``pyplot`` is ever imported: this process
has no display, and importing pyplot with an interactive backend selected
would either crash or silently try to open a GUI window.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")  # must precede the pyplot import; there is no display here
import matplotlib.axes
import matplotlib.pyplot as plt
import pandas as pd

from cellsense.errors import ToolError
from cellsense.io.schema import Citation, ToolResult, Workspace
from cellsense.tools.base import apply_conditions, check_columns, tool_spec

OUTPUT_DIR = Path("output")

_CHART_TYPES = ("bar", "line", "scatter", "hist", "pie")
_AGG_FUNCS = ("count", "sum", "mean", "median", "min", "max", "nunique", "std")

_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "filename": {
            "type": "string",
            "description": "File to read. Omit if only one file is loaded.",
        },
        "sheet": {
            "type": "string",
            "description": "Sheet name for Excel files. Omit for CSV or single-sheet Excel.",
        },
        "chart_type": {"type": "string", "enum": list(_CHART_TYPES)},
        "x_column": {
            "type": "string",
            "description": "Column for the x-axis (or category labels for bar/pie).",
        },
        "y_column": {
            "type": "string",
            "description": "Column for the y-axis (or values for bar/pie). Not required for hist.",
        },
        "conditions": {
            "type": "array",
            "description": "Optional row filters applied before plotting (see filter_rows).",
            "items": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "operator": {"type": "string"},
                    "value": {},
                },
                "required": ["column", "operator"],
            },
        },
        "combine": {"type": "string", "enum": ["and", "or"]},
        "group_by": {
            "type": "string",
            "description": "Optional column to group by before plotting, e.g. 'Region'.",
        },
        "agg_func": {
            "type": "string",
            "enum": list(_AGG_FUNCS),
            "description": "Aggregation applied when group_by is set. Defaults to 'sum'.",
        },
        "title": {"type": "string", "description": "Chart title. Auto-generated if omitted."},
    },
    "required": ["chart_type", "x_column"],
}


def _handle(args: dict[str, Any], workspace: Workspace) -> ToolResult:
    ref, df = workspace.resolve(args.get("filename"), args.get("sheet"))
    chart_type = args["chart_type"]
    if chart_type not in _CHART_TYPES:
        raise ToolError(
            f"Unknown chart_type {chart_type!r}.", hint=f"Supported: {', '.join(_CHART_TYPES)}"
        )

    x_col: str = args["x_column"]
    y_col: str | None = args.get("y_column")
    if chart_type != "hist" and not y_col:
        raise ToolError(f"chart_type={chart_type!r} requires 'y_column'.")
    if chart_type == "hist":
        y_col = None  # hist never uses y_col, even if the caller passed one

    conditions = args.get("conditions") or []
    combine = args.get("combine", "and")
    group_by = args.get("group_by")
    agg_func = args.get("agg_func", "sum")
    title = args.get("title")

    working = apply_conditions(df, conditions, combine) if conditions else df

    if group_by and y_col:
        check_columns(working, [group_by, y_col])
        grouped = working.groupby(group_by, dropna=False)
        plot_df = grouped[y_col].agg(agg_func).reset_index()
        plot_df.columns = [group_by, f"{agg_func}_{y_col}"]
        x_col, y_col = group_by, f"{agg_func}_{y_col}"
        citation_rows = sorted({int(i) for idx in grouped.groups.values() for i in idx})
    else:
        plot_df = working
        citation_rows = [int(i) for i in working.index]

    if chart_type == "hist":
        check_columns(plot_df, [x_col])
    else:
        assert y_col is not None  # guaranteed by the chart_type/y_column check above
        check_columns(plot_df, [x_col, y_col])

    fig, ax = plt.subplots(figsize=(8, 5))
    _draw(ax, plot_df, chart_type, x_col, y_col)
    ax.set_title(title or _auto_title(x_col, y_col, ref.label), fontsize=13, pad=12)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = OUTPUT_DIR / f"plot_{timestamp}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    citations = [Citation(filename=ref.filename, sheet=ref.sheet, rows=tuple(citation_rows))]
    summary = f"Chart saved to {out_path} ({chart_type}, {len(plot_df)} data point(s))"
    return ToolResult(data=plot_df, citations=citations, summary=summary)


def _draw(
    ax: matplotlib.axes.Axes,
    df: pd.DataFrame,
    chart_type: str,
    x_col: str,
    y_col: str | None,
) -> None:
    if chart_type == "hist":
        _require_any_finite(df[x_col], x_col)
        ax.hist(df[x_col], bins="auto", edgecolor="white")
        ax.set_xlabel(x_col)
        ax.set_ylabel("Count")
        return

    # Every other chart type requires y_col; _handle already enforces this
    # before calling _draw, so this just gives the type checker the same fact.
    assert y_col is not None
    if chart_type == "bar":
        ax.bar(df[x_col].astype(str), df[y_col])
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    elif chart_type == "line":
        ax.plot(df[x_col], df[y_col], marker="o")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
    elif chart_type == "scatter":
        ax.scatter(df[x_col], df[y_col], alpha=0.7)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
    elif chart_type == "pie":
        _require_all_finite(df[y_col], y_col)
        labels = df[x_col].astype(str).tolist()
        ax.pie(df[y_col], labels=labels, autopct="%1.1f%%", startangle=140)
        ax.axis("equal")


def _require_all_finite(series: pd.Series, column: str) -> None:
    """A pie chart's wedge sizes must *all* be finite -- matplotlib raises a
    raw ``ValueError`` for a single NaN/inf, which would otherwise escape as
    an unhandled crash instead of the retryable ``ToolError`` every other
    failure path in this tool produces (ARCHITECTURE.md SS3: a tool failure
    is never fatal to the turn).

    Only applies to numeric columns: a categorical/text column is a legitimate
    (if unusual) thing to hand to a chart, is not what this bug is about, and
    ``pd.to_numeric(..., errors="coerce")`` would otherwise flag every one of
    its values as "non-finite" simply for not being a number.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return
    bad = int((~np.isfinite(series.astype(float))).sum())
    if bad:
        raise ToolError(
            f"Column {column!r} has {bad} non-finite value(s) (NaN/inf); "
            "a pie chart requires every value to be finite.",
            hint="Filter out missing rows first (see filter_rows), or plot a different column.",
        )


def _require_any_finite(series: pd.Series, column: str) -> None:
    """A histogram only breaks when *every* value is non-finite -- matplotlib
    can't autodetect a bin range from an all-NaN column and raises a raw
    ``ValueError``. A handful of NaNs mixed with real data is fine; matplotlib
    already drops them silently, so this only guards the all-non-finite case.

    Only applies to numeric columns: matplotlib's hist() also happily buckets
    a categorical/text column by value, and coercing it to numeric first would
    make every row look "non-finite" and reject a perfectly plottable column.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return
    if not np.isfinite(series.astype(float)).any():
        raise ToolError(
            f"Column {column!r} has no finite values to plot.",
            hint="Check for an all-null column, or filter to rows that have data first.",
        )


def _auto_title(x_col: str, y_col: str | None, label: str) -> str:
    return f"{y_col} by {x_col} ({label})" if y_col else f"{x_col} distribution ({label})"


SPEC = tool_spec(
    name="plot",
    description=(
        "Render a chart (bar, line, scatter, hist, pie) from a table and save "
        "it as a PNG. Optionally filter rows first, or group and aggregate "
        "before plotting (e.g. sum revenue by region)."
    ),
    parameters=_PARAMETERS,
    side_effect=True,
)(_handle)
