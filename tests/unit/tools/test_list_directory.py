"""Pins cellsense.tools.list_directory: browsing, spreadsheet flagging,
symlink exclusion, and path-traversal refusal (chdir into a tmp root so the
sandbox has a controlled cwd to test against).
"""

from __future__ import annotations

import pytest

from cellsense.errors import ToolError
from cellsense.io.schema import Workspace
from cellsense.tools.list_directory import SPEC


def _run(args, workspace=None):
    return SPEC.handler(args, workspace or Workspace())


@pytest.fixture(autouse=True)
def _in_tmp_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_lists_files_and_directories(_in_tmp_root) -> None:
    (_in_tmp_root / "data.csv").write_text("a,b\n1,2\n")
    (_in_tmp_root / "sub").mkdir()
    result = _run({})
    names = set(result.data["name"])
    assert names == {"data.csv", "sub"}


def test_flags_supported_spreadsheet_extensions(_in_tmp_root) -> None:
    (_in_tmp_root / "a.csv").write_text("x\n1\n")
    (_in_tmp_root / "b.xlsx").write_bytes(b"not a real workbook")
    (_in_tmp_root / "c.txt").write_text("notes")
    result = _run({})
    flags = {
        name: bool(flag)
        for name, flag in zip(result.data["name"], result.data["spreadsheet"], strict=True)
    }
    assert flags["a.csv"] is True
    assert flags["b.xlsx"] is True
    assert flags["c.txt"] is False


def test_directories_are_not_flagged_as_spreadsheets(_in_tmp_root) -> None:
    (_in_tmp_root / "folder").mkdir()
    result = _run({})
    row = result.data[result.data["name"] == "folder"].iloc[0]
    assert row["type"] == "dir"
    assert bool(row["spreadsheet"]) is False


def test_sorts_directories_before_files_then_alphabetically(_in_tmp_root) -> None:
    (_in_tmp_root / "zzz_file.csv").write_text("a\n1\n")
    (_in_tmp_root / "aaa_dir").mkdir()
    result = _run({})
    assert result.data["name"].tolist() == ["aaa_dir", "zzz_file.csv"]


def test_symlinked_entries_are_omitted(_in_tmp_root) -> None:
    real = _in_tmp_root / "real.csv"
    real.write_text("a\n1\n")
    link = _in_tmp_root / "link.csv"
    link.symlink_to(real)
    result = _run({})
    assert "link.csv" not in set(result.data["name"])
    assert "real.csv" in set(result.data["name"])


def test_subdirectory_can_be_listed_explicitly(_in_tmp_root) -> None:
    sub = _in_tmp_root / "sub"
    sub.mkdir()
    (sub / "nested.csv").write_text("a\n1\n")
    result = _run({"path": "sub"})
    assert result.data["name"].tolist() == ["nested.csv"]


def test_nonexistent_path_raises(_in_tmp_root) -> None:
    with pytest.raises(ToolError, match="does not exist"):
        _run({"path": "nowhere"})


def test_path_pointing_at_a_file_raises(_in_tmp_root) -> None:
    (_in_tmp_root / "a.csv").write_text("a\n1\n")
    with pytest.raises(ToolError, match="Not a directory"):
        _run({"path": "a.csv"})


def test_path_escaping_root_raises(_in_tmp_root) -> None:
    with pytest.raises(ToolError, match="outside the working directory"):
        _run({"path": "../"})


def test_empty_directory_returns_zero_rows(_in_tmp_root) -> None:
    result = _run({})
    assert len(result.data) == 0
    assert list(result.data.columns) == ["name", "type", "spreadsheet"]


def test_no_citations_are_produced() -> None:
    result = _run({})
    assert result.citations == []


def test_spec_metadata() -> None:
    assert SPEC.name == "list_directory"
    assert SPEC.reads_filesystem is True
    assert SPEC.side_effect is False
