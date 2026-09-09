"""filter_rows: select matching rows via structured conditions.

Conditions are a list of ``{column, operator, value}`` dicts rather than a
free-form pandas query string (the prototype's approach): a model-controlled
``DataFrame.query()`` expression is an arbitrary-expression evaluation risk,
and structured conditions are no less expressive for the operators tools
actually need. See ``cellsense.tools.base.condition_mask`` for the operator
implementations shared with ``plot``.
"""

from __future__ import annotations

from typing import Any

from cellsense.errors import ToolError
from cellsense.io.schema import Citation, ToolResult, Workspace
from cellsense.tools.base import OPERATORS, apply_conditions, tool_spec

_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "filename": {
            "type": "string",
            "description": "File to read. Omit if only one file is loaded.",
        },
        "sheet": {
            "type": "string",
            "description": "Sheet name for Excel files. Omit for CSV or single-sheet Excel.",
        },
        "conditions": {
            "type": "array",
            "description": "One or more row conditions, combined per 'combine'.",
            "items": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "operator": {"type": "string", "enum": list(OPERATORS)},
                    "value": {
                        "description": (
                            "Comparison value. A list for 'in'/'notin'; "
                            "omit for 'isnull'/'notnull'."
                        ),
                    },
                },
                "required": ["column", "operator"],
            },
        },
        "combine": {
            "type": "string",
            "enum": ["and", "or"],
            "description": "How to combine multiple conditions. Defaults to 'and'.",
        },
    },
    "required": ["conditions"],
}


def _handle(args: dict[str, Any], workspace: Workspace) -> ToolResult:
    ref, df = workspace.resolve(args.get("filename"), args.get("sheet"))
    conditions: list[dict[str, Any]] = args["conditions"]
    combine: str = args.get("combine", "and")

    if not conditions:
        raise ToolError("filter_rows requires at least one condition.")

    result = apply_conditions(df, conditions, combine)

    rows = tuple(int(i) for i in result.index)
    citations = [Citation(filename=ref.filename, sheet=ref.sheet, rows=rows)]
    summary = (
        f"filter_rows({ref.label}): {len(conditions)} condition(s) [{combine}] "
        f"-> {len(result)} row(s) matched"
    )
    return ToolResult(data=result, citations=citations, summary=summary)


SPEC = tool_spec(
    name="filter_rows",
    description=(
        "Filter rows in a file/sheet by one or more column conditions "
        f"({', '.join(OPERATORS)}), combined with AND or OR. String "
        "comparisons are case-insensitive."
    ),
    parameters=_PARAMETERS,
)(_handle)
