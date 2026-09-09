"""Pins cellsense.tools.find_files: discovery + immediate load into the
workspace, pattern matching, already-loaded skip, non-directory/empty
results, and path-traversal refusal.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cellsense.errors import ToolError
from cellsense.io.schema import Workspace
from cellsense.tools.find_files import SPEC


def _run(args, workspace=None):
    return SPEC.handler(args, workspace if workspace is not None else Workspace())


@pytest.fixture(autouse=True)
def _in_tmp_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write_csv(path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_finds_and_loads_matching_files(_in_tmp_root) -> None:
    _write_csv(_in_tmp_root / "sales_2024.csv", pd.DataFrame({"a": [1]}))
    _write_csv(_in_tmp_root / "headcount.csv", pd.DataFrame({"b": [2]}))
    ws = Workspace()
    result = _run({}, ws)
    assert set(ws.files) == {"sales_2024.csv", "headcount.csv"}
    assert "loaded 2 new" in result.summary


def test_pattern_restricts_matches(_in_tmp_root) -> None:
    _write_csv(_in_tmp_root / "sales_2024.csv", pd.DataFrame({"a": [1]}))
    _write_csv(_in_tmp_root / "headcount.csv", pd.DataFrame({"b": [2]}))
    ws = Workspace()
    _run({"pattern": "sales*"}, ws)
    assert set(ws.files) == {"sales_2024.csv"}


def test_recursive_finds_nested_files(_in_tmp_root) -> None:
    _write_csv(_in_tmp_root / "sub" / "deep.csv", pd.DataFrame({"a": [1]}))
    ws = Workspace()
    result = _run({"recursive": True}, ws)
    assert "deep.csv" in ws.files
    assert "found 1 file" in result.summary


def test_non_recursive_ignores_nested_files(_in_tmp_root) -> None:
    _write_csv(_in_tmp_root / "sub" / "deep.csv", pd.DataFrame({"a": [1]}))
    ws = Workspace()
    _run({"recursive": False}, ws)
    assert "deep.csv" not in ws.files


def test_already_loaded_file_is_reported_and_not_reloaded(_in_tmp_root) -> None:
    path = _in_tmp_root / "sales_2024.csv"
    _write_csv(path, pd.DataFrame({"a": [1]}))
    ws = Workspace()
    ws.add(path)
    original_df = ws.files["sales_2024.csv"].sheets["default"]

    result = _run({}, ws)

    assert ws.files["sales_2024.csv"].sheets["default"] is original_df
    row = result.data[result.data["file"] == "sales_2024.csv"].iloc[0]
    assert row["status"] == "already loaded"


def test_citations_only_cover_newly_loaded_files(_in_tmp_root) -> None:
    already = _in_tmp_root / "already.csv"
    _write_csv(already, pd.DataFrame({"a": [1]}))
    ws = Workspace()
    ws.add(already)
    _write_csv(_in_tmp_root / "new_one.csv", pd.DataFrame({"a": [1]}))

    result = _run({}, ws)

    cited_files = {c.filename for c in result.citations}
    assert cited_files == {"new_one.csv"}


def test_no_matches_returns_empty_result(_in_tmp_root) -> None:
    ws = Workspace()
    result = _run({}, ws)
    assert len(result.data) == 0
    assert "No spreadsheet files found" in result.summary


def test_non_directory_path_returns_status_row(_in_tmp_root) -> None:
    path = _in_tmp_root / "a.csv"
    _write_csv(path, pd.DataFrame({"a": [1]}))
    ws = Workspace()
    result = _run({"path": "a.csv"}, ws)
    assert "Not a directory" in result.summary
    assert len(result.data) == 0


def test_path_escaping_root_raises(_in_tmp_root) -> None:
    with pytest.raises(ToolError, match="outside the working directory"):
        _run({"path": "../"})


def test_empty_csv_load_failure_is_reported_gracefully(_in_tmp_root) -> None:
    """An empty CSV raises DataError from the loader -- find_files' own
    except DataError clause must turn that into a per-file 'error: ...' row
    rather than letting it escape.
    """
    (_in_tmp_root / "empty.csv").write_text("a,b\n", encoding="utf-8")  # header only
    ws = Workspace()
    result = _run({}, ws)
    row = result.data[result.data["file"] == "empty.csv"].iloc[0]
    assert row["status"].startswith("error:")
    assert "empty.csv" not in ws.files


def test_schema_summary_lists_loaded_columns(_in_tmp_root) -> None:
    _write_csv(_in_tmp_root / "sales.csv", pd.DataFrame({"region": ["a"], "revenue": [1]}))
    ws = Workspace()
    result = _run({}, ws)
    assert "region" in result.summary
    assert "revenue" in result.summary


def test_corrupt_xlsx_is_reported_gracefully_not_raised(_in_tmp_root) -> None:
    (_in_tmp_root / "bad.xlsx").write_bytes(b"not a real workbook, just garbage bytes")
    ws = Workspace()
    result = _run({}, ws)
    row = result.data[result.data["file"] == "bad.xlsx"].iloc[0]
    assert row["status"].startswith("error:")


def test_spec_metadata() -> None:
    assert SPEC.name == "find_files"
    assert SPEC.reads_filesystem is True
    assert SPEC.side_effect is False
