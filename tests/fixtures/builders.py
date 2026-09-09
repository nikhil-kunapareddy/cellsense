"""Programmatic fixture builders for the CellSense unit test suite.

Every function here is a pure, deterministic builder of a small pandas
DataFrame (or a dict of them, for a workbook) covering one tricky shape the
loader/tools/context builder must handle correctly. Nothing here touches
``data/`` (gitignored, regenerated from a different seed by
``scripts/make_sample_data.py``) -- unit tests must stand on their own.

Callers that need real files on disk (loader tests) write these out with the
``write_csv``/``write_xlsx`` fixtures in ``tests/conftest.py``; callers that
just need an in-memory ``Workspace`` (tool tests) use ``make_workspace``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 20260909


def rng() -> np.random.Generator:
    """A fresh, deterministically-seeded generator -- call once per builder
    invocation so repeated calls in one test module don't share state.
    """
    return np.random.default_rng(SEED)


def sales_frame(n: int = 24) -> pd.DataFrame:
    """A small sales table: real NaNs in a numeric column, a genuine date
    column, a small-cardinality categorical column, and an alnum order-id
    column that must never be reinterpreted as a date.
    """
    r = rng()
    regions = ["North", "South", "East", "West"]
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    revenue = r.uniform(100, 10_000, size=n).round(2)
    revenue[::7] = np.nan
    return pd.DataFrame(
        {
            "order_id": [f"E{2000 + i}" for i in range(n)],
            "order_date": dates.strftime("%Y-%m-%d"),
            "region": [regions[i % len(regions)] for i in range(n)],
            "revenue": revenue,
            "units": r.integers(1, 50, size=n),
        }
    )


def big_sales_frame(n: int = 240) -> pd.DataFrame:
    """A larger sales table, shaped like :func:`sales_frame` but big enough
    (``n`` > ``io.schema._MAX_ROW_INDICES``) that a tool touching every row
    must render as an honest row *count* citation (e.g. ``"240 rows"``)
    rather than an enumerated list -- see ARCHITECTURE.md SS2's citation
    rendering rule. Used by integration tests that need this threshold
    without depending on the real, gitignored ``data/sales_2024.csv``.
    """
    r = rng()
    regions = ["North", "South", "East", "West"]
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    revenue = r.uniform(100, 10_000, size=n).round(2)
    return pd.DataFrame(
        {
            "order_id": [f"E{2000 + i}" for i in range(n)],
            "order_date": dates.strftime("%Y-%m-%d"),
            "region": [regions[i % len(regions)] for i in range(n)],
            "revenue": revenue,
            "units": r.integers(1, 50, size=n),
        }
    )


def headcount_frame(n: int = 10) -> pd.DataFrame:
    """A small headcount table, joinable to :func:`sales_frame` on ``region``."""
    r = rng()
    regions = ["North", "South", "East", "West"]
    departments = ["Sales", "Engineering", "Support"]
    return pd.DataFrame(
        {
            "region": [regions[i % len(regions)] for i in range(n)],
            "employee_id": [f"E{3000 + i}" for i in range(n)],
            "department": [departments[i % len(departments)] for i in range(n)],
            "salary": r.uniform(50_000, 150_000, size=n).round(2),
        }
    )


def id_like_bare_digit_frame() -> pd.DataFrame:
    """Product codes that are pure digit strings.

    ``pd.to_datetime`` would happily reinterpret ``"1000"`` as the year 1000
    if handed to it blindly -- the loader must keep a column shaped like this
    numeric or textual, never datetime, based purely on the column's own
    name/content (no date hint either way here).
    """
    return pd.DataFrame(
        {
            "product_code": ["1000", "1001", "1002", "1003"],
            "label": ["Widget A", "Widget B", "Widget C", "Widget D"],
        }
    )


def alnum_id_frame() -> pd.DataFrame:
    """IDs like ``E2001`` that fail numeric coercion and must stay text."""
    return pd.DataFrame(
        {
            "employee_id": ["E2001", "E2002", "E2003"],
            "name": ["Ana", "Ben", "Cy"],
        }
    )


def real_date_frame() -> pd.DataFrame:
    """A genuine, separator-bearing date column that *should* parse as a date."""
    return pd.DataFrame(
        {
            "order_date": ["2024-01-15", "2024-02-20", "2024-03-05"],
            "amount": [100.0, 200.0, 300.0],
        }
    )


def partially_numeric_text_frame() -> pd.DataFrame:
    """A column that is mostly numeric-looking but has a few non-numeric
    sentinel values -- must stay text, not be silently coerced to numeric
    with the sentinels turned into NaN.

    Deliberately avoids pandas' own default CSV ``na_values`` sentinels
    (``"N/A"``, ``"NA"``, ``"null"``, ...) -- those get turned into a real
    NaN by ``pd.read_csv`` itself before ``cellsense``'s own dtype-inference
    ever runs, which would test pandas' CSV parser rather than the loader's
    "don't silently coerce a partially-numeric column" guard.
    """
    return pd.DataFrame(
        {
            "code": ["100", "200", "pending", "300", "unknown"],
            "value": [1, 2, 3, 4, 5],
        }
    )


def unicode_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["Renée", "Müller", "田中", "Øyvind", "😀 Chris"],
            "score": [1, 2, 3, 4, 5],
        }
    )


def nan_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0],
            "b": ["x", None, "z"],
        }
    )


def mixed_types_frame() -> pd.DataFrame:
    """A column pandas loads as ``object`` holding genuinely mixed Python types."""
    return pd.DataFrame(
        {
            "mixed": pd.Series([1, "two", 3.0, None, True], dtype=object),
            "n": [1, 2, 3, 4, 5],
        }
    )


def empty_frame() -> pd.DataFrame:
    """A zero-row frame -- the loader must skip a sheet shaped like this."""
    return pd.DataFrame({"a": pd.Series(dtype="float64"), "b": pd.Series(dtype="object")})


def multi_sheet_workbook() -> dict[str, pd.DataFrame]:
    """Catalog + Inventory (joinable on ``sku``) + a truly empty sheet that
    the loader must skip on load.
    """
    return {
        "Catalog": pd.DataFrame(
            {
                "sku": ["SKU-001", "SKU-002", "SKU-003"],
                "name": ["Atlas Router", "Bolt Switch", "Cirrus AP"],
                "price": [49.99, 19.99, 89.50],
            }
        ),
        "Inventory": pd.DataFrame(
            {
                "sku": ["SKU-001", "SKU-002", "SKU-003"],
                "quantity": [120, 80, 45],
            }
        ),
        "Empty": empty_frame(),
    }
