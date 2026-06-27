"""Builds a compressed, LLM-ready context string from loaded FileData objects."""
from __future__ import annotations

from typing import Dict

import pandas as pd

from src.data import FileData
from src.utils.classifier import classify_file

# Caps to keep context token-friendly
_MAX_SAMPLE_ROWS = 5
_MAX_SAMPLE_COL_WIDTH = 40
_MAX_NUMERIC_COLS = 10  # only show stats for first N numeric columns


def build_context(file_data: Dict[str, FileData]) -> str:
    """Return a single string summarising all loaded files for injection into the LLM."""
    sections: list[str] = []
    for filename, fd in file_data.items():
        category = classify_file(fd)
        header = f"=== File: {filename} (type: {fd.file_type}, total rows: {fd.total_rows:,}, category: {category}) ==="
        sheet_sections = [_summarise_sheet(df, sheet_name, fd.file_type) for sheet_name, df in fd.sheets.items()]
        sections.append(header + "\n" + "\n\n".join(sheet_sections))
    return "\n\n" + "\n\n".join(sections) + "\n"


# ── private ────────────────────────────────────────────────────────────────────

def _summarise_sheet(df: pd.DataFrame, sheet_name: str, file_type: str) -> str:
    lines: list[str] = []

    if file_type == "excel":
        lines.append(f"--- Sheet: {sheet_name} ---")

    lines.append(f"Shape: {len(df):,} rows × {len(df.columns)} columns")

    col_lines = ["Columns:"]
    for col in df.columns:
        dtype_label = _human_dtype(df[col])
        null_count = int(df[col].isna().sum())
        entry = f"  - {col!r}: {dtype_label}"
        if null_count:
            entry += f"  ({null_count} nulls)"
        col_lines.append(entry)
    lines.append("\n".join(col_lines))

    sample_n = min(_MAX_SAMPLE_ROWS, len(df))
    if sample_n > 0:
        sample = df.head(sample_n).to_string(index=True, max_colwidth=_MAX_SAMPLE_COL_WIDTH)
        lines.append(f"Sample rows (index 0–{sample_n - 1}):\n{sample}")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        cols_to_show = numeric_cols[:_MAX_NUMERIC_COLS]
        stats = df[cols_to_show].describe().loc[["min", "mean", "max"]].to_string()
        omitted = len(numeric_cols) - len(cols_to_show)
        note = f" (first {len(cols_to_show)} of {len(numeric_cols)})" if omitted else ""
        lines.append(f"Numeric stats{note}:\n{stats}")

    return "\n".join(lines)


def _human_dtype(series: pd.Series) -> str:
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_integer_dtype(dtype):
        return "integer"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_string_dtype(dtype) or dtype == object:
        n_unique = series.nunique()
        return f"categorical ({n_unique} unique)" if n_unique <= 20 else "text"
    return str(dtype)
