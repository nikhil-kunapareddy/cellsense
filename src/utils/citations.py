from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class Citation:
    filename: str
    sheet_name: str | None  # None for CSV "default" sheet
    row_indices: List[int] = field(default_factory=list)

    def is_csv(self) -> bool:
        return self.sheet_name is None or self.sheet_name == "default"


def format_citations(citations: List[Citation]) -> str:
    """Format a list of citations into the standard source block.

    Output: Sources: sales.xlsx [Sheet: Q3, Rows: 12, 45, 67] | headcount.csv [Rows: 3, 8]
    """
    if not citations:
        return ""

    parts = []
    for c in citations:
        rows_str = ", ".join(str(i) for i in sorted(set(c.row_indices)))
        if c.is_csv():
            parts.append(f"{c.filename} [Rows: {rows_str}]")
        else:
            parts.append(f"{c.filename} [Sheet: {c.sheet_name}, Rows: {rows_str}]")

    return "Sources: " + " | ".join(parts)


def merge_citations(citations: List[Citation]) -> List[Citation]:
    """Merge citations from the same file+sheet, deduplicating row indices."""
    index: dict[tuple[str, str | None], Citation] = {}
    for c in citations:
        key = (c.filename, c.sheet_name)
        if key in index:
            index[key].row_indices.extend(c.row_indices)
        else:
            index[key] = Citation(
                filename=c.filename,
                sheet_name=c.sheet_name,
                row_indices=list(c.row_indices),
            )
    return list(index.values())
