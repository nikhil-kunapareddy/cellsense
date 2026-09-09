"""The ``@``-mention fuzzy file picker.

Typing ``@`` mid-question opens a completion menu over CSV/XLSX files under
the current directory. Deliberately cheap: it reports relative path and file
size only -- never opens a file to count rows, since that would make every
keystroke in a large workspace pay for a pandas read.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

#: Directories never worth descending into, whether or not they're gitignored.
EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".cellsense",
    }
)
SUPPORTED_SUFFIXES = frozenset({".csv", ".xlsx", ".xls"})
MAX_ENTRIES = 5000
MAX_DEPTH = 6

_MENTION_TOKEN_RE = re.compile(r"(?<!\S)@(\S+)")


@dataclass(frozen=True)
class FileHit:
    """One attachable file: enough to display and to open later, nothing more."""

    relpath: str
    size_bytes: int

    def size_hint(self) -> str:
        size = float(self.size_bytes)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024 or unit == "GB":
                return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}GB"


def _fuzzy_score(query: str, text: str) -> int | None:
    """Subsequence match: every character of ``query`` must appear, in order, in ``text``.

    Returns ``None`` on no match; otherwise a higher score for tighter and
    earlier matches so e.g. "sales" ranks ``sales_q1.csv`` above
    ``data/archive/old_sales_report.csv``.
    """
    if not query:
        return 0
    cursor = 0
    first = -1
    last = 0
    for char in query:
        idx = text.find(char, cursor)
        if idx == -1:
            return None
        if first == -1:
            first = idx
        last = idx
        cursor = idx + 1
    span = last - first + 1
    return 1000 - span - first


class FilePicker:
    """Caches one directory walk per session; call :meth:`refresh` to rescan."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._cache: list[FileHit] | None = None

    def refresh(self) -> None:
        self._cache = None

    def files(self) -> list[FileHit]:
        if self._cache is None:
            self._cache = self._walk()
        return self._cache

    def search(self, query: str, limit: int = 15) -> list[FileHit]:
        if not query:
            return self.files()[:limit]
        needle = query.lower()
        scored: list[tuple[int, FileHit]] = []
        for hit in self.files():
            score = _fuzzy_score(needle, hit.relpath.lower())
            if score is not None:
                scored.append((score, hit))
        scored.sort(key=lambda pair: (-pair[0], len(pair[1].relpath)))
        return [hit for _, hit in scored[:limit]]

    def _walk(self) -> list[FileHit]:
        hits: list[FileHit] = []
        stack: list[tuple[Path, int]] = [(self._root, 0)]
        seen = 0
        while stack and seen < MAX_ENTRIES:
            current, depth = stack.pop()
            try:
                entries = list(os.scandir(current))
            except OSError:
                continue
            for entry in entries:
                if seen >= MAX_ENTRIES:
                    break
                seen += 1
                if entry.is_dir(follow_symlinks=False):
                    if entry.name in EXCLUDED_DIRS or entry.name.startswith("."):
                        continue
                    if depth < MAX_DEPTH:
                        stack.append((Path(entry.path), depth + 1))
                    continue
                if entry.name.startswith("."):
                    continue
                suffix = Path(entry.name).suffix.lower()
                if suffix not in SUPPORTED_SUFFIXES:
                    continue
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = 0
                relpath = os.path.relpath(entry.path, self._root)
                hits.append(FileHit(relpath=relpath, size_bytes=size))
        hits.sort(key=lambda hit: hit.relpath)
        return hits


class AtMentionCompleter(Completer):
    """Offers :class:`FileHit` completions while typing an ``@token``."""

    def __init__(self, picker: FilePicker) -> None:
        self._picker = picker

    def get_completions(self, document: Document, complete_event: object) -> Iterable[Completion]:
        text = document.text_before_cursor
        at = text.rfind("@")
        if at == -1:
            return
        token = text[at + 1 :]
        if any(ch.isspace() for ch in token):
            return
        if at > 0 and not text[at - 1].isspace():
            return
        for hit in self._picker.search(token, limit=15):
            yield Completion(
                f"@{hit.relpath}",
                start_position=-(len(token) + 1),
                display=f"{hit.relpath}  ({hit.size_hint()})",
                display_meta="attach",
            )


def extract_attachments(line: str) -> tuple[str, list[str]]:
    """Strip ``@path`` tokens from a submitted line, returning (text, paths).

    The REPL calls this on the submitted question so ``on_attach`` sees plain
    paths and the model sees a clean question without ``@`` noise.
    """
    paths = _MENTION_TOKEN_RE.findall(line)
    stripped = _MENTION_TOKEN_RE.sub("", line)
    stripped = re.sub(r"[ \t]+", " ", stripped).strip()
    return stripped, paths
