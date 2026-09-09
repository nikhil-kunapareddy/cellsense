#!/usr/bin/env python3
"""Regenerate the deterministic sample datasets in ``data/``.

``data/`` is gitignored (it doubles as your own scratch area), so this script
exists to recreate a known-good corpus for demos and manual testing:

    python scripts/make_sample_data.py

The seed is fixed, so the numbers below are stable and the integration tests
assert against them:

    sales_2024.csv  total revenue      2242849.81
                    top rep by revenue Femi Adeyemi
                    null channel rows  6

The corpus is deliberately shaped to exercise the hard paths: a joinable key
across three files, a multi-sheet workbook, ID-like strings that must NOT be
parsed as dates, real NaNs, booleans, and a datetime column.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260909
OUT = Path(__file__).resolve().parent.parent / "data"

REGIONS = ["North America", "EMEA", "APAC", "LATAM"]
CHANNELS = ["Direct", "Partner", "Online"]
SKUS = [f"SKU-{i:03d}" for i in range(1, 13)]
REPS = [
    "Ana Reyes",
    "Ben Cho",
    "Carla Diaz",
    "Dmitri Volkov",
    "Elena Petrova",
    "Femi Adeyemi",
    "Grace Lin",
    "Hugo Martins",
    "Ines Duval",
    "Jonas Weber",
]
DEPARTMENTS = ["Sales", "Engineering", "Marketing", "Support", "Finance"]

PRODUCT_NAMES = [
    "Atlas Router",
    "Bolt Switch",
    "Cirrus AP",
    "Delta Modem",
    "Echo Gateway",
    "Flux NIC",
    "Grid Firewall",
    "Helix Cable",
    "Ion Repeater",
    "Juno Hub",
    "Kilo Bridge",
    "Lumen Optic",
]
CATEGORIES = [
    "Networking",
    "Networking",
    "Wireless",
    "Access",
    "Access",
    "Components",
    "Security",
    "Components",
    "Wireless",
    "Networking",
    "Networking",
    "Optics",
]


def build_sales(rng: np.random.Generator, n: int = 240) -> pd.DataFrame:
    """Order-level sales facts. Joins to products on ``sku``, headcount on ``sales_rep``."""
    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    df = pd.DataFrame(
        {
            "order_id": [f"ORD-{100000 + i}" for i in range(n)],
            "order_date": dates,
            "region": rng.choice(REGIONS, n, p=[0.40, 0.28, 0.22, 0.10]),
            "sales_rep": rng.choice(REPS, n),
            "sku": rng.choice(SKUS, n),
            "channel": rng.choice(CHANNELS, n, p=[0.45, 0.30, 0.25]),
            "units": rng.integers(1, 40, n),
            "unit_price": np.round(rng.uniform(18, 940, n), 2),
        }
    )
    df["revenue"] = np.round(df["units"] * df["unit_price"], 2)
    # A partially-populated numeric column, so NaN handling is exercised.
    df["discount_pct"] = np.where(
        rng.random(n) < 0.3, np.round(rng.uniform(0.05, 0.25, n), 3), np.nan
    )
    # A few missing categoricals, so groupby-with-nulls is exercised.
    df.loc[rng.choice(n, 6, replace=False), "channel"] = None
    return df


def build_headcount(rng: np.random.Generator, m: int = 42) -> pd.DataFrame:
    """Employee dimension. ``employee_id`` is intentionally ID-like, not a date."""
    hire = pd.to_datetime("2016-01-01") + pd.to_timedelta(rng.integers(0, 3200, m), unit="D")
    return pd.DataFrame(
        {
            "employee_id": [f"E{2000 + i}" for i in range(m)],
            # The 10 sales reps come first and appear exactly once, so joining
            # sales_2024.sales_rep -> headcount.name is unambiguously one-to-one.
            "name": [REPS[i] if i < len(REPS) else f"Employee {i}" for i in range(m)],
            "department": rng.choice(DEPARTMENTS, m),
            "region": rng.choice(REGIONS, m),
            "hire_date": hire.date,
            "salary_usd": rng.integers(62_000, 240_000, m),
            "manager": [None if i % 9 == 0 else REPS[(i * 3) % len(REPS)] for i in range(m)],
            "is_remote": rng.choice([True, False], m, p=[0.55, 0.45]),
        }
    )


def build_products(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two sheets of one workbook, so multi-sheet addressing is exercised."""
    catalog = pd.DataFrame(
        {
            "sku": SKUS,
            "product_name": PRODUCT_NAMES,
            "category": CATEGORIES,
            "unit_cost": np.round(rng.uniform(10, 520, len(SKUS)), 2),
            "list_price": np.round(rng.uniform(30, 980, len(SKUS)), 2),
        }
    )
    inventory = pd.DataFrame(
        {
            "sku": np.repeat(SKUS, 2),
            "warehouse": ["WH-EAST", "WH-WEST"] * len(SKUS),
            "on_hand": rng.integers(0, 900, len(SKUS) * 2),
            "reorder_point": rng.integers(50, 200, len(SKUS) * 2),
        }
    )
    return catalog, inventory


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    sales = build_sales(rng)
    sales.to_csv(OUT / "sales_2024.csv", index=False)

    headcount = build_headcount(rng)
    headcount.to_csv(OUT / "headcount.csv", index=False)

    catalog, inventory = build_products(rng)
    with pd.ExcelWriter(OUT / "products.xlsx", engine="openpyxl") as writer:
        catalog.to_excel(writer, sheet_name="Catalog", index=False)
        inventory.to_excel(writer, sheet_name="Inventory", index=False)

    for name in ("sales_2024.csv", "headcount.csv", "products.xlsx"):
        size_kb = (OUT / name).stat().st_size / 1024
        print(f"  {name:20} {size_kb:7.1f} KB")

    print("\n  ground truth")
    print(f"    total revenue      {sales['revenue'].sum():.2f}")
    print(f"    top rep by revenue {sales.groupby('sales_rep')['revenue'].sum().idxmax()}")
    print(f"    null channel rows  {int(sales['channel'].isna().sum())}")


if __name__ == "__main__":
    main()
