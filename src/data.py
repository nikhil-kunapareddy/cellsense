"""File loading: Excel (multi-sheet) and CSV with basic cleanup."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

SUPPORTED_EXTENSIONS = {".xlsx", ".xls", ".csv"}


@dataclass
class FileData:
    """Data class for a file. One file normalized into a queryable object."""
    path: Path
    sheets: Dict[str, pd.DataFrame]  # sheet_name → DataFrame
    file_type: str                    # "excel" | "csv"

    @property
    def filename(self) -> str:
        return self.path.name

    @property
    def total_rows(self) -> int:
        return sum(len(df) for df in self.sheets.values())


def load_files(paths: List[Path]) -> Dict[str, FileData]:
    """Load all files; returns {filename: FileData}. Raises on unrecognised types."""
    result: Dict[str, FileData] = {}
    for path in paths:
        suffix = path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            result[path.name] = _load_excel(path)
        elif suffix == ".csv":
            result[path.name] = _load_csv(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix!r} ({path.name})")
    return result


# ── private helpers ────────────────────────────────────────────────────────────

def _load_excel(path: Path) -> FileData:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress openpyxl style warnings
        xl = pd.ExcelFile(path, engine="openpyxl")

    sheets: Dict[str, pd.DataFrame] = {}
    for name in xl.sheet_names:
        df = xl.parse(name)
        if df.empty:
            continue
        sheets[name] = _clean(df)

    if not sheets:
        raise ValueError(f"{path.name} contains no readable sheets.")

    return FileData(path=path, sheets=sheets, file_type="excel")


def _load_csv(path: Path) -> FileData:
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="warn")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", on_bad_lines="warn")

    if df.empty:
        raise ValueError(f"{path.name} is empty.")

    return FileData(path=path, sheets={"default": _clean(df)}, file_type="csv")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from names/string values; infer better dtypes."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    for col in df.select_dtypes(include="object").columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
            continue
        except (ValueError, TypeError):
            pass
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df[col] = pd.to_datetime(df[col])
        except (ValueError, TypeError):
            pass

    return df.infer_objects()
