"""Heuristic business-category label for a table.

No LLM call: this exists purely so ``build_context`` can give the model a
one-word steer ("Financial", "People / HR", ...) for free, before it has
spent a single token deciding what a table is about. A keyword scoreboard is
crude but cheap and deterministic, which matters more here than precision --
worst case the model ignores a wrong guess once it has seen the columns.
"""

from __future__ import annotations

import pandas as pd

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Financial": [
        "budget",
        "revenue",
        "expense",
        "profit",
        "loss",
        "p&l",
        "forecast",
        "capex",
        "opex",
        "cashflow",
        "cash_flow",
        "cash flow",
        "balance",
        "invoice",
        "cost",
        "margin",
        "fiscal",
        "quarterly",
        "annual",
        "income",
        "ebitda",
        "ledger",
        "accounting",
        "finance",
        "spend",
        "billing",
    ],
    "Operational": [
        "inventory",
        "supply",
        "chain",
        "project",
        "task",
        "milestone",
        "status",
        "pipeline",
        "crm",
        "lead",
        "order",
        "shipment",
        "vendor",
        "stock",
        "warehouse",
        "logistics",
        "delivery",
        "ticket",
        "sprint",
        "backlog",
        "procurement",
        "operations",
        "ops",
        "fulfillment",
    ],
    "People / HR": [
        "employee",
        "headcount",
        "salary",
        "compensation",
        "hire",
        "hiring",
        "department",
        "org",
        "role",
        "position",
        "payroll",
        "skills",
        "attrition",
        "performance",
        "review",
        "onboarding",
        "offboarding",
        "benefits",
        "workforce",
        "staff",
        "hr",
        "people",
        "talent",
        "recruiter",
        "tenure",
    ],
    "Strategic / Reporting": [
        "kpi",
        "okr",
        "metric",
        "scorecard",
        "board",
        "qbr",
        "market",
        "competitor",
        "growth",
        "target",
        "objective",
        "strategy",
        "reporting",
        "dashboard",
        "exec",
        "quarterly review",
        "north star",
        "vision",
        "roadmap",
        "benchmark",
        "analyst",
    ],
}


def classify_table(df: pd.DataFrame, name: str) -> str:
    """Return one of five category labels based on a table's columns and its
    display name (e.g. ``"sales.xlsx[Q1]"``). Falls back to ``"Others"`` when
    no keyword scores above zero.
    """
    tokens = [str(c).lower() for c in df.columns]
    tokens.append(name.lower())
    combined = " ".join(tokens)

    scores = {cat: sum(kw in combined for kw in kws) for cat, kws in _CATEGORY_KEYWORDS.items()}
    best_cat, best_score = max(scores.items(), key=lambda item: item[1])
    return best_cat if best_score > 0 else "Others"
