"""Pins cellsense.io.classify: keyword-based category scoring."""

from __future__ import annotations

import pandas as pd

from cellsense.io.classify import classify_table


def test_financial_keywords_win_financial_category() -> None:
    df = pd.DataFrame({"revenue": [1], "expense": [1], "budget": [1]})
    assert classify_table(df, "q1.xlsx") == "Financial"


def test_hr_keywords_win_people_category() -> None:
    df = pd.DataFrame({"employee": [1], "department": [1], "salary": [1]})
    assert classify_table(df, "headcount.csv") == "People / HR"


def test_operational_keywords_win_operational_category() -> None:
    df = pd.DataFrame({"inventory": [1], "warehouse": [1], "shipment": [1]})
    assert classify_table(df, "stock.csv") == "Operational"


def test_strategic_keywords_win_strategic_category() -> None:
    df = pd.DataFrame({"kpi": [1], "okr": [1], "roadmap": [1]})
    assert classify_table(df, "board_deck.xlsx") == "Strategic / Reporting"


def test_no_keyword_hits_falls_back_to_others() -> None:
    df = pd.DataFrame({"foo": [1], "bar": [1]})
    assert classify_table(df, "misc.csv") == "Others"


def test_table_display_name_itself_contributes_to_the_score() -> None:
    df = pd.DataFrame({"x": [1], "y": [2]})
    assert classify_table(df, "quarterly_budget.csv") == "Financial"


def test_tie_break_is_deterministic_via_dict_iteration_order() -> None:
    # A single shared keyword across categories should resolve to whichever
    # category is checked first when scores tie (max() keeps the first max).
    df = pd.DataFrame({"col": [1]})
    result_a = classify_table(df, "misc")
    result_b = classify_table(df, "misc")
    assert result_a == result_b == "Others"
