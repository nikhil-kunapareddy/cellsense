"""Shared pytest fixtures for the CellSense unit test suite.

Nothing here depends on ``data/`` (gitignored). Fixtures build small,
deterministic in-memory DataFrames (via ``tests/fixtures/builders.py``) and,
where a test genuinely needs a file on disk (loader tests), write them under
``tmp_path`` so nothing touches the real filesystem outside pytest's own
sandbox.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cellsense.io.schema import FileData, Workspace
from tests.fixtures import builders as fb

# ── DataFrame fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def sales_df() -> pd.DataFrame:
    return fb.sales_frame()


@pytest.fixture
def headcount_df() -> pd.DataFrame:
    return fb.headcount_frame()


@pytest.fixture
def big_sales_frame() -> pd.DataFrame:
    """A 240-row sales table -- see ``fixtures.builders.big_sales_frame`` for
    why the integration citation-rendering tests need something bigger than
    ``sales_df``.
    """
    return fb.big_sales_frame()


# ── Workspace builders (no disk I/O) ────────────────────────────────────────


@pytest.fixture
def make_workspace():
    """Factory: ``make_workspace(sales=df, headcount=df2)`` -> a ``Workspace``
    with one CSV-shaped file per keyword argument, keyed as ``"<name>.csv"``.
    """

    def _make(**named_frames: pd.DataFrame) -> Workspace:
        files: dict[str, FileData] = {}
        for name, df in named_frames.items():
            filename = f"{name}.csv"
            files[filename] = FileData(
                path=Path(f"/virtual/{filename}"),
                sheets={"default": df.copy()},
                file_type="csv",
            )
        return Workspace(files=files)

    return _make


@pytest.fixture
def make_excel_workspace():
    """Factory: ``make_excel_workspace("book.xlsx", {"Sheet1": df})`` -> a
    ``Workspace`` with one Excel-shaped file holding the given sheets.
    """

    def _make(filename: str, sheets: dict[str, pd.DataFrame]) -> Workspace:
        files = {
            filename: FileData(
                path=Path(f"/virtual/{filename}"),
                sheets={name: df.copy() for name, df in sheets.items()},
                file_type="excel",
            )
        }
        return Workspace(files=files)

    return _make


# ── On-disk file writers (loader tests only) ────────────────────────────────


@pytest.fixture
def write_csv(tmp_path: Path):
    """Factory: ``write_csv("sales.csv", df)`` -> ``Path`` to a real CSV file
    under a fresh ``tmp_path``. Pass ``subdir`` to nest it (for duplicate-
    filename disambiguation tests).
    """

    def _write(name: str, df: pd.DataFrame, *, subdir: str | None = None) -> Path:
        directory = (tmp_path / subdir) if subdir else tmp_path
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        df.to_csv(path, index=False)
        return path

    return _write


@pytest.fixture
def write_xlsx(tmp_path: Path):
    """Factory: ``write_xlsx("book.xlsx", {"Sheet1": df})`` -> ``Path`` to a
    real multi-sheet workbook under a fresh ``tmp_path``.
    """

    def _write(name: str, sheets: dict[str, pd.DataFrame], *, subdir: str | None = None) -> Path:
        directory = (tmp_path / subdir) if subdir else tmp_path
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        return path

    return _write


# ── Misc ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def assert_no_mutation():
    """Factory: snapshot a workspace's tables, run a callable, then assert
    none of the original DataFrames changed -- the tool contract requires
    every handler to treat ``workspace.files`` as read-only.
    """

    def _check(workspace: Workspace) -> callable:
        snapshots = {
            key: {sheet: df.copy(deep=True) for sheet, df in fd.sheets.items()}
            for key, fd in workspace.files.items()
        }

        def _verify() -> None:
            for key, fd in workspace.files.items():
                for sheet, df in fd.sheets.items():
                    pd.testing.assert_frame_equal(df, snapshots[key][sheet])

        return _verify

    return _check
