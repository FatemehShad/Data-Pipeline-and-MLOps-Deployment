"""CSV + JSON ingestion with validation and transactional SQLite publication."""

import json
import sqlite3
from pathlib import Path

import pandas as pd


def ingest(orders_path: Path, products_path: Path, database: Path, sql_path: Path) -> pd.DataFrame:
    orders = pd.read_csv(orders_path)
    products = pd.DataFrame(json.loads(products_path.read_text()))
    required = {"order_id", "order_date", "product_id", "units", "unit_price"}
    if not required.issubset(orders) or not {"product_id", "category"}.issubset(products):
        raise ValueError("Missing required source columns")
    if orders.empty or products.empty or orders.isna().any().any() or products.isna().any().any():
        raise ValueError("Empty sources or null values")
    if orders.order_id.duplicated().any() or products.product_id.duplicated().any():
        raise ValueError("Duplicate primary keys")
    orders["order_date"] = pd.to_datetime(
        orders.order_date, format="%Y-%m-%d", errors="raise"
    ).dt.strftime("%Y-%m-%d")
    for column in ("units", "unit_price"):
        orders[column] = pd.to_numeric(orders[column], errors="raise")
        if not ((orders[column] > 0) & (orders[column] < float("inf"))).all():
            raise ValueError(f"{column} must contain finite positive values")
    if not orders.product_id.isin(products.product_id).all():
        raise ValueError("Orders reference unknown products")
    database.parent.mkdir(parents=True, exist_ok=True)
    # Stage in memory: failed validation/transformation never replaces existing warehouse tables.
    with sqlite3.connect(":memory:") as stage:
        orders.to_sql("orders", stage, index=False)
        products.to_sql("products", stage, index=False)
        sales = pd.read_sql_query(sql_path.read_text(), stage)
        if len(sales) != len(orders):
            raise ValueError("Transformation changed the order count")
        sales.to_sql("sales", stage, index=False)
        with sqlite3.connect(database) as destination:
            stage.backup(destination)
    return sales
