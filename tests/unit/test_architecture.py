"""Enforces docs/ARCHITECTURE.md §1's import-direction rules via static AST
analysis (no imports of the target modules, so this is fast and can never
have a side effect):

* `events.py` imports nothing from cellsense.
* `graph/` must not import from `ui/`.
* `ui/` must not import from `graph/` at runtime (a TYPE_CHECKING-guarded
  import is fine -- it never executes).
* `tools/` and `io/` must not import from `providers/`, `graph/`, or `ui/`.

Skips gracefully if cellsense/graph does not exist yet.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

CELLSENSE_ROOT = Path(__file__).resolve().parents[2] / "cellsense"


class _ImportCollector(ast.NodeVisitor):
    """Collects dotted module names from `import x.y` / `from x.y import z`,
    skipping anything nested under `if TYPE_CHECKING:` (those imports never
    run).
    """

    def __init__(self) -> None:
        self.imports: list[str] = []

    def visit_If(self, node: ast.If) -> None:
        if _is_type_checking(node.test):
            for stmt in node.orelse:
                self.visit(stmt)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.level == 0:  # ignore relative imports (none used here anyway)
            self.imports.append(node.module)


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _cellsense_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    collector = _ImportCollector()
    collector.visit(tree)
    return [
        name for name in collector.imports if name == "cellsense" or name.startswith("cellsense.")
    ]


def _iter_py_files(pkg_dir: Path) -> list[Path]:
    if not pkg_dir.is_dir():
        return []
    return sorted(pkg_dir.rglob("*.py"))


def _violations(pkg_dir: Path, forbidden_prefixes: tuple[str, ...]) -> dict[str, list[str]]:
    violations: dict[str, list[str]] = {}
    for path in _iter_py_files(pkg_dir):
        bad = [
            imp
            for imp in _cellsense_imports(path)
            if any(imp == prefix or imp.startswith(prefix + ".") for prefix in forbidden_prefixes)
        ]
        if bad:
            violations[str(path.relative_to(CELLSENSE_ROOT.parent))] = bad
    return violations


def test_events_module_imports_nothing_from_cellsense() -> None:
    events_path = CELLSENSE_ROOT / "events.py"
    assert events_path.is_file(), "cellsense/events.py should always exist"
    assert _cellsense_imports(events_path) == []


def test_graph_package_does_not_import_ui() -> None:
    graph_dir = CELLSENSE_ROOT / "graph"
    if not graph_dir.is_dir():
        pytest.skip("cellsense/graph does not exist yet")
    violations = _violations(graph_dir, ("cellsense.ui",))
    assert not violations, f"graph/ importing from ui/: {violations}"


def test_ui_package_does_not_import_graph_at_runtime() -> None:
    ui_dir = CELLSENSE_ROOT / "ui"
    assert ui_dir.is_dir()
    violations = _violations(ui_dir, ("cellsense.graph",))
    assert not violations, f"ui/ importing from graph/ at runtime: {violations}"


def test_tools_package_does_not_import_providers_graph_or_ui() -> None:
    tools_dir = CELLSENSE_ROOT / "tools"
    assert tools_dir.is_dir()
    violations = _violations(tools_dir, ("cellsense.providers", "cellsense.graph", "cellsense.ui"))
    assert not violations, f"tools/ importing forbidden package: {violations}"


def test_io_package_does_not_import_providers_graph_or_ui() -> None:
    io_dir = CELLSENSE_ROOT / "io"
    assert io_dir.is_dir()
    violations = _violations(io_dir, ("cellsense.providers", "cellsense.graph", "cellsense.ui"))
    assert not violations, f"io/ importing forbidden package: {violations}"


def test_type_checking_guarded_import_is_correctly_ignored_by_the_scanner(tmp_path) -> None:
    """A meta-test for the AST scanner itself: a TYPE_CHECKING-guarded import
    of a forbidden package must not be flagged (it never runs), while the
    same import outside the guard must be.
    """
    guarded = tmp_path / "guarded.py"
    guarded.write_text(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from cellsense.graph.engine import Engine\n"
    )
    assert _cellsense_imports(guarded) == []

    unguarded = tmp_path / "unguarded.py"
    unguarded.write_text("from cellsense.graph.engine import Engine\n")
    assert _cellsense_imports(unguarded) == ["cellsense.graph.engine"]
