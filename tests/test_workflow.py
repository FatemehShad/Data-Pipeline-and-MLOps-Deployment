import json
import sqlite3
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient

from sales_mlops.api import app
from sales_mlops.pipeline import ingest
from sales_mlops.training import MODEL_NAME, train

ROOT = Path(__file__).resolve().parents[1]


def pipeline(tmp_path, orders=None):
    return ingest(
        orders or ROOT / "data/orders.csv",
        ROOT / "data/products.json",
        tmp_path / "warehouse.db",
        ROOT / "sql/transform.sql",
    )


def test_ingestion_integrates_sources_and_is_repeatable(tmp_path):
    first = pipeline(tmp_path)
    second = pipeline(tmp_path)
    assert len(first) == 300
    pd.testing.assert_frame_equal(first, second)
    assert (first.revenue == first.units * first.unit_price).all()
    assert set(first.category) == {"books", "home", "electronics"}
    with sqlite3.connect(tmp_path / "warehouse.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0] == 300


@pytest.mark.parametrize("defect", ["duplicate", "unknown_product", "negative", "null", "date"])
def test_invalid_source_preserves_warehouse(tmp_path, defect):
    pipeline(tmp_path)
    data = pd.read_csv(ROOT / "data/orders.csv")
    if defect == "duplicate":
        data.loc[1, "order_id"] = data.loc[0, "order_id"]
    elif defect == "unknown_product":
        data.loc[0, "product_id"] = 999
    elif defect == "negative":
        data.loc[0, "units"] = -1
    elif defect == "null":
        data.loc[0, "unit_price"] = float("nan")
    else:
        data.loc[0, "order_date"] = "invalid-date"
    bad = tmp_path / "bad.csv"
    data.to_csv(bad, index=False)
    with pytest.raises(ValueError):
        pipeline(tmp_path, bad)
    with sqlite3.connect(tmp_path / "warehouse.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0] == 300


def test_training_registry_api_and_promotion_gate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/tracking.db")
    monkeypatch.delenv("MODEL_URI", raising=False)
    pipeline(tmp_path)
    report = train(tmp_path / "warehouse.db", tmp_path / "reports")
    assert report["accepted"] and report["mae"] < 15
    assert json.loads((tmp_path / "reports/evaluation.json").read_text()) == report
    client = MlflowClient()
    champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    assert champion.version == report["version"]
    with TestClient(app) as api:
        assert api.get("/ready").status_code == 200
        result = api.post("/predict", json={"units": 3, "unit_price": 20, "category": "home"})
        assert result.status_code == 200
        assert abs(result.json()["predicted_revenue"] - 60) < 15
        assert (
            api.post(
                "/predict", json={"units": -1, "unit_price": 20, "category": "home"}
            ).status_code
            == 422
        )
        assert "sales_prediction_requests_total" in api.get("/metrics").text
    with pytest.raises(ValueError, match="promotion threshold"):
        train(tmp_path / "warehouse.db", tmp_path / "rejected", max_mae=0)
    assert client.get_model_version_by_alias(MODEL_NAME, "champion").version == champion.version
    assert len(mlflow.search_runs(experiment_names=["sales-revenue"])) == 2
