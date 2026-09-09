"""The data model every other layer of CellSense builds on.

``Workspace`` is the single source of truth for loaded tables: tools read from
it, the graph passes it around untouched, and the UI never sees a raw
DataFrame without a ``TableRef`` attached to it. Keeping that model in one
small module (rather than scattering ad-hoc dicts of DataFrames, as the
original prototype did) is what lets citations, multi-sheet Excel, and
CSV-vs-Excel differences stay consistent everywhere they are used.

Citations are the other half of the contract: any tool that reshapes data
(aggregate, join) must be able to point back at the exact source rows that
produced a value. ``Citation.rows`` is a tuple specifically so citations are
hashable -- ``merge_citations`` puts them in a set-like dict keyed on
``(filename, sheet)`` to dedupe and union row indices from multiple tool
calls in the same turn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from cellsense.errors import DataError

__all__ = [
    "Citation",
    "FileData",
    "TableRef",
    "ToolResult",
    "Workspace",
    "format_citations",
    "merge_citations",
]

# format_citations() thresholds: past this many collapsed range/singleton
# tokens, or this many total row indices, an enumerated list stops being
# readable (and, worse, inviting truncation makes it a false-precision claim
# -- see the "Citation rendering rule" in docs/ARCHITECTURE.md SS2). Beyond
# either threshold the citation collapses to an honest count instead.
_MAX_RANGE_TOKENS = 10
_MAX_ROW_INDICES = 20


@dataclass(frozen=True)
class TableRef:
    """Addresses exactly one DataFrame in a :class:`Workspace`.

    ``sheet`` is ``None`` for CSV files -- there is nothing to disambiguate --
    and the sheet name for Excel files, even when that workbook only has one
    sheet. This asymmetry (rather than always defaulting to a synthetic
    ``"default"`` label) is what lets :func:`format_citations` omit the
    ``[Sheet: ...]`` clause for CSVs without a special case at every call site.
    """

    filename: str
    sheet: str | None

    @property
    def label(self) -> str:
        """Human-readable address, e.g. ``"sales.xlsx[Q1]"`` or ``"data.csv"``."""
        return f"{self.filename}[{self.sheet}]" if self.sheet is not None else self.filename


@dataclass
class FileData:
    """One loaded file, normalized into a dict of sheet name -> DataFrame.

    CSV files always use the single key ``"default"`` internally (see
    ``Workspace.resolve`` for how that maps back to a ``TableRef`` with
    ``sheet=None``); Excel files use their real sheet names.
    """

    path: Path
    sheets: dict[str, pd.DataFrame]
    file_type: Literal["csv", "excel"]

    @property
    def filename(self) -> str:
        return self.path.name

    @property
    def total_rows(self) -> int:
        return sum(len(df) for df in self.sheets.values())


class Workspace:
    """The set of loaded files for one session.

    Tools are handed a ``Workspace`` and are expected to treat the DataFrames
    it holds as immutable -- read a table out, transform a copy, never write
    back into ``workspace.files``. The one sanctioned mutation is ``add()``,
    used to bring a brand-new file into scope at runtime (an ``@``-mention in
    the UI, or a file the ``find_files`` tool just discovered) without
    disturbing anything already loaded.
    """

    def __init__(self, files: dict[str, FileData] | None = None) -> None:
        self.files: dict[str, FileData] = files if files is not None else {}

    def is_empty(self) -> bool:
        return not self.files

    def refs(self) -> list[TableRef]:
        """Every addressable table currently in scope, in load order."""
        result: list[TableRef] = []
        for fd in self.files.values():
            if fd.file_type == "csv":
                result.append(TableRef(filename=fd.filename, sheet=None))
            else:
                result.extend(TableRef(filename=fd.filename, sheet=s) for s in fd.sheets)
        return result

    def table(self, filename: str, sheet: str | None = None) -> pd.DataFrame:
        """Direct lookup: ``filename`` is required, unlike ``resolve()``."""
        _, df = self.resolve(filename, sheet)
        return df

    def resolve(self, filename: str | None, sheet: str | None) -> tuple[TableRef, pd.DataFrame]:
        """The one tool-facing lookup: turns loose (filename, sheet) arguments
        from a model's tool call into a concrete ``TableRef`` and DataFrame.

        ``filename=None`` defaults to the only loaded file when there is
        exactly one; with several files loaded the model must say which one.
        For a CSV, any ``sheet`` other than ``None``/``"default"`` is a user
        error, and the resolved ``TableRef.sheet`` is always ``None`` -- CSVs
        have no sheets to cite.
        """
        if filename is None:
            if len(self.files) == 1:
                filename = next(iter(self.files))
            elif not self.files:
                raise DataError("No files are loaded.", hint="Load a file before querying it.")
            else:
                raise DataError(
                    "Multiple files are loaded; specify which one.",
                    hint=f"Loaded files: {', '.join(self.files)}",
                )

        fd = self._lookup_file(filename)
        return self._resolve_sheet(fd, filename, sheet)

    def add(self, path: Path) -> FileData:
        """Load one new file into the workspace without touching anything
        already loaded. Used for runtime attach (an ``@``-mention, or a file
        the ``find_files`` tool just found) rather than the bulk startup load.

        Imports ``cellsense.io.loaders`` lazily: that module imports this one
        at module scope to build ``Workspace`` objects, so importing it back
        here at module scope would be a cycle.
        """
        from cellsense.io.loaders import _load_one

        fd = _load_one(path)
        key = fd.filename
        if key in self.files:
            key = f"{path.resolve().parent.name}/{fd.filename}"
        self.files[key] = fd
        return fd

    # ── private ──────────────────────────────────────────────────────────────

    def _lookup_file(self, filename: str) -> FileData:
        if filename in self.files:
            return self.files[filename]
        close = get_close_matches(filename, self.files.keys(), n=3)
        hint = f"Loaded files: {', '.join(self.files)}" if self.files else "No files are loaded."
        if close:
            hint = f"Did you mean: {', '.join(close)}? {hint}"
        raise DataError(f"No loaded file named {filename!r}.", hint=hint)

    def _resolve_sheet(
        self, fd: FileData, filename: str, sheet: str | None
    ) -> tuple[TableRef, pd.DataFrame]:
        if fd.file_type == "csv":
            if sheet not in (None, "default"):
                raise DataError(
                    f"{filename!r} is a CSV; it has no sheet named {sheet!r}.",
                    hint="Omit 'sheet' or pass 'default' for CSV files.",
                )
            return TableRef(filename=filename, sheet=None), fd.sheets["default"]

        sheets = fd.sheets
        if sheet is None:
            if len(sheets) == 1:
                only = next(iter(sheets))
                return TableRef(filename=filename, sheet=only), sheets[only]
            raise DataError(
                f"{filename!r} has multiple sheets; specify which one.",
                hint=f"Available sheets: {', '.join(sheets)}",
            )

        if sheet not in sheets:
            close = get_close_matches(sheet, sheets.keys(), n=3)
            hint = f"Available sheets: {', '.join(sheets)}"
            if close:
                hint = f"Did you mean: {', '.join(close)}? {hint}"
            raise DataError(f"Sheet {sheet!r} not found in {filename!r}.", hint=hint)

        return TableRef(filename=filename, sheet=sheet), sheets[sheet]


@dataclass(frozen=True)
class Citation:
    """A pointer back at the source rows behind a piece of the answer.

    ``rows`` holds original DataFrame index values, never positional offsets
    into a reshaped result -- an aggregate over 10,000 rows still cites the
    real row numbers a user could go look up in the spreadsheet.
    """

    filename: str
    sheet: str | None
    rows: tuple[int, ...] = ()

    def is_csv(self) -> bool:
        return self.sheet is None


def merge_citations(citations: list[Citation]) -> list[Citation]:
    """Group citations by (filename, sheet), unioning and sorting their row
    indices. Does **not** truncate: row-level citations are this product's
    headline feature, and a capped row set silently understates how many
    rows actually contributed -- a false-precision claim, not a convenience.
    ``format_citations`` is where a large row set becomes readable (by
    collapsing into ranges, or an honest count past a threshold); that is a
    rendering concern, not a reason to lose data here.

    Multiple tool calls in one turn often touch the same table; without this
    the citation block at the end of an answer would repeat
    ``sales.xlsx [Rows: ...]`` several times instead of once with the full
    set of rows actually used.
    """
    grouped: dict[tuple[str, str | None], set[int]] = {}
    for c in citations:
        grouped.setdefault((c.filename, c.sheet), set()).update(c.rows)

    merged: list[Citation] = []
    for (filename, sheet), rows in grouped.items():
        merged.append(Citation(filename=filename, sheet=sheet, rows=tuple(sorted(rows))))
    return merged


def format_citations(citations: list[Citation]) -> str:
    """Render citations as the standard source block shown to the user, e.g.

    ``Sources: sales.xlsx [Sheet: Q1, Rows: 3-9, 14, 20-22] | data.csv [Rows: 0-1]``

    This is the single renderer -- nothing else should format a citation.
    Consecutive row indices collapse into ``a-b`` ranges. Past
    ``_MAX_RANGE_TOKENS`` collapsed tokens, or past ``_MAX_ROW_INDICES``
    total rows (whichever trips first), the whole row list is replaced by an
    honest count instead, e.g. ``sales_2024.csv [240 rows]``: printing the
    first N of many contributing indices reads as a precise claim about N
    specific rows, which would be false, whereas a count is always true.
    """
    if not citations:
        return ""
    parts = [
        f"{c.filename} [{_describe_rows(c.rows)}]"
        if c.is_csv()
        else f"{c.filename} [Sheet: {c.sheet}, {_describe_rows(c.rows)}]"
        for c in citations
    ]
    return "Sources: " + " | ".join(parts)


def _describe_rows(rows: tuple[int, ...]) -> str:
    """``"Rows: 3-9, 14, 20-22"`` for a small/compact row set, or ``"N rows"``
    once that would be unreadable or too long to be an honest listing.
    """
    if not rows:
        return "Rows: (none)"
    ranges = _collapse_ranges(sorted(rows))
    if len(ranges) > _MAX_RANGE_TOKENS or len(rows) > _MAX_ROW_INDICES:
        return f"{len(rows)} rows"
    rendered = ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)
    return f"Rows: {rendered}"


def _collapse_ranges(ordered: list[int]) -> list[tuple[int, int]]:
    """Collapse sorted, deduplicated row indices into ``(start, end)`` runs,
    e.g. ``[3,4,5,6,7,8,9,14,20,21,22]`` -> ``[(3,9), (14,14), (20,22)]``.
    """
    ranges: list[tuple[int, int]] = []
    start = prev = ordered[0]
    for row in ordered[1:]:
        if row == prev + 1:
            prev = row
            continue
        ranges.append((start, prev))
        start = prev = row
    ranges.append((start, prev))
    return ranges


@dataclass
class ToolResult:
    """What a tool handler returns: the reshaped data, its provenance, and a
    one-line human summary for the tool trace shown in the UI.
    """

    data: pd.DataFrame
    summary: str
    citations: list[Citation] = field(default_factory=list)

    def to_text(self, max_rows: int = 20) -> str:
        """What the model sees in the tool result message: the summary, then
        a plain-text table capped at ``max_rows`` so a 50,000-row filter_rows
        call doesn't blow the context window.
        """
        rows_shown = min(max_rows, len(self.data))
        table = self.data.head(rows_shown).to_string(index=True, max_colwidth=50)
        note = (
            f"\n[showing {rows_shown} of {len(self.data)} rows]"
            if len(self.data) > rows_shown
            else ""
        )
        return f"{self.summary}\n\n{table}{note}"

    def preview(self, n: int = 5) -> list[dict[str, Any]]:
        """What the UI renders in a tool-trace row: a small list of row dicts
        with every value coerced to a JSON-safe scalar (numpy/pandas scalars
        to native Python, NaN/NaT to ``None``, timestamps to ISO strings) so
        it can go straight into an event payload without a custom encoder.
        """
        head = self.data.head(n)
        return [
            {str(col): _json_safe(value) for col, value in row.items()}
            for row in head.to_dict(orient="records")
        ]


def _json_safe(value: Any) -> Any:
    """Coerce one cell value to something ``json.dumps`` can handle unaided."""
    try:
        is_na = pd.isna(value)
    except (TypeError, ValueError):
        is_na = False
    if is_na is True:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        f = float(value)
        return None if math.isnan(f) else f
    if isinstance(value, np.generic):
        return value.item()
    return value
