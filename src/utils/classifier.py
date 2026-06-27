"""Classifies loaded files into one of five business categories."""
from __future__ import annotations

from src.data import FileData

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Financial": [
        "budget", "revenue", "expense", "profit", "loss", "p&l", "forecast",
        "capex", "opex", "cashflow", "cash_flow", "cash flow", "balance",
        "invoice", "cost", "margin", "fiscal", "quarterly", "annual", "income",
        "ebitda", "ledger", "accounting", "finance", "spend", "billing",
    ],
    "Operational": [
        "inventory", "supply", "chain", "project", "task", "milestone", "status",
        "pipeline", "crm", "lead", "order", "shipment", "vendor", "stock",
        "warehouse", "logistics", "delivery", "ticket", "sprint", "backlog",
        "procurement", "operations", "ops", "fulfillment",
    ],
    "People / HR": [
        "employee", "headcount", "salary", "compensation", "hire", "hiring",
        "department", "org", "role", "position", "payroll", "skills", "attrition",
        "performance", "review", "onboarding", "offboarding", "benefits",
        "workforce", "staff", "hr", "people", "talent", "recruiter", "tenure",
    ],
    "Strategic / Reporting": [
        "kpi", "okr", "metric", "scorecard", "board", "qbr", "market",
        "competitor", "growth", "target", "objective", "strategy", "reporting",
        "dashboard", "exec", "quarterly review", "north star", "vision",
        "roadmap", "benchmark", "analyst",
    ],
}


def classify_file(fd: FileData) -> str:
    """Return one of five category labels based on column/sheet/filename keywords."""
    tokens: list[str] = []
    for df in fd.sheets.values():
        tokens.extend(str(c).lower() for c in df.columns)
    tokens.extend(s.lower() for s in fd.sheets.keys())
    tokens.append(fd.filename.lower())

    combined = " ".join(tokens)
    scores = {cat: sum(kw in combined for kw in kws) for cat, kws in _CATEGORY_KEYWORDS.items()}
    best_cat, best_score = max(scores.items(), key=lambda x: x[1])
    return best_cat if best_score > 0 else "Others"
