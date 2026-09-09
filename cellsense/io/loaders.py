"""File loading: CSV and multi-sheet Excel, normalized into a ``Workspace``.

This is a from-scratch port of the original prototype's loader, hardened in
three ways the prototype was not:

1. Every deliberate failure raises a :class:`~cellsense.errors.DataError` (or
   the more specific :class:`~cellsense.errors.UnsupportedFileTypeError`)
   instead of a bare ``ValueError``, so the CLI can render it cleanly.
2. Dtype inference is pandas-3-safe. Pandas 3 made ``"str"`` the default
   dtype for text columns (not ``object``), which means the prototype's
   ``select_dtypes(include="object")`` silently stopped matching text columns
   and, worse, emits a deprecation warning when it is coerced into still
   matching them. See ``_string_like_columns`` below.
3. Datetime inference is name/shape-gated. ``pd.to_datetime`` will happily
   reinterpret a column of small integers-as-strings (an ID column, a
   4-digit product code) as a year, e.g. ``"1000"`` becomes
   ``1000-01-01``. The prototype attempted ``to_datetime`` on *every* column
   that failed numeric conversion; here we only attempt it when the column
   name reads as date-like or a sample of its values actually looks like a
   date (contains ``-`` or ``/`` separators), never on bare digit strings.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd

from cellsense.errors import DataError, UnsupportedFileTypeError
from cellsense.io.schema import FileData, Workspace

__all__ = ["SUPPORTED_EXTENSIONS", "load_workspace"]

SUPPORTED_EXTENSIONS = {".xlsx", ".xls", ".csv"}

_DATE_NAME_HINTS = (
    "date",
    "time",
    "timestamp",
    "day",
    "month",
    "year",
    "dob",
    "created",
    "updated",
    "modified",
)
_DATE_SEP_PATTERN = re.compile(r"^\s*\d{1,4}[-/]\d{1,2}[-/]\d{1,4}")


def load_workspace(paths: list[Path]) -> Workspace:
    """Load every path into one ``Workspace``.

    Files are keyed by bare filename unless two loaded paths share a name
    (e.g. ``data/2023/report.csv`` and ``data/2024/report.csv``), in which
    case *both* are disambiguated with a ``<parent-dir>/<name>`` prefix so
    neither silently shadows the other.
    """
    groups: dict[str, list[Path]] = {}
    for path in paths:
        groups.setdefault(path.name, []).append(path)

    files: dict[str, FileData] = {}
    for name, group in groups.items():
        collision = len(group) > 1
        for path in group:
            key = f"{path.resolve().parent.name}/{name}" if collision else name
            files[key] = _load_one(path)
    return Workspace(files=files)


# ── private helpers ──────────────────────────────────────────────────────────


def _load_one(path: Path) -> FileData:
    """Dispatch a single file to the right loader. Not part of the public
    API -- used by ``load_workspace`` and by ``Workspace.add()`` for runtime
    attach, which is why this lives at module scope instead of nested inside
    ``load_workspace``.
    """
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return _load_excel(path)
    if suffix == ".csv":
        return _load_csv(path)
    raise UnsupportedFileTypeError(str(path), path.suffix, SUPPORTED_EXTENSIONS)


def _load_excel(path: Path) -> FileData:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # openpyxl style/theme warnings, not ours to fix
        xl = pd.ExcelFile(path, engine="openpyxl")

    sheets: dict[str, pd.DataFrame] = {}
    for name in xl.sheet_names:
        df = xl.parse(name)
        if df.empty:
            continue
        sheets[str(name)] = _clean(df)

    if not sheets:
        raise DataError(
            f"{path.name} contains no readable sheets.",
            hint="Every sheet was empty after loading.",
        )
    return FileData(path=path, sheets=sheets, file_type="excel")


def _load_csv(path: Path) -> FileData:
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="warn")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", on_bad_lines="warn")

    if df.empty:
        raise DataError(f"{path.name} is empty.", hint="The file has no data rows.")

    return FileData(path=path, sheets={"default": _clean(df)}, file_type="csv")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names and string values, then try to
    tighten each text column's dtype (numeric, then datetime) without
    guessing wrong on IDs. Never mutates the caller's frame.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    text_cols = _string_like_columns(df)
    for col in text_cols:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    for col in _string_like_columns(df):
        series = df[col]
        numeric = pd.to_numeric(series, errors="coerce")
        # Only commit the numeric cast if it accounted for every non-null
        # value -- a partially-numeric column (e.g. "N/A" mixed with
        # numbers) should stay text rather than silently losing rows to NaN.
        if series.notna().any() and numeric.notna().sum() == series.notna().sum():
            df[col] = numeric
            continue
        if _looks_date_like(col, series):
            df[col] = pd.to_datetime(series, errors="coerce")

    return df


def _string_like_columns(df: pd.DataFrame) -> list[str]:
    """Columns holding text, under either pandas 3's default ``"str"`` dtype
    or the legacy ``object`` dtype -- without the deprecation warning that
    ``select_dtypes(include="object")`` now raises on pandas 3 when it falls
    back to also matching ``"str"`` columns for backward compatibility.
    """
    return [
        c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c].dtype)
    ]


def _looks_date_like(column: str, series: pd.Series) -> bool:
    """Gate for attempting ``to_datetime``: the column name reads as a date
    field, or a sample of its values actually contains date separators. Pure
    digit strings (IDs, zip codes, product codes) never reach ``to_datetime``
    through this path, so they can't be silently reinterpreted as years.
    """
    name_hint = any(hint in column.lower() for hint in _DATE_NAME_HINTS)
    sample = series.dropna().astype(str).head(20)
    if sample.empty:
        return False
    content_hint = sample.str.match(_DATE_SEP_PATTERN).mean() > 0.8
    if not (name_hint or content_hint):
        return False
    try:
        pd.to_datetime(sample, format=None, errors="raise")
    except (ValueError, TypeError):
        return False
    return True
