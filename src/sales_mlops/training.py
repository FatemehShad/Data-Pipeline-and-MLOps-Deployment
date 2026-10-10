"""Time-ordered evaluation, experiment tracking, and versioned model promotion."""

import json
import os
import sqlite3
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = ["units", "unit_price", "category"]
MODEL_NAME = "sales-revenue"


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///.runtime/mlflow.db")


def train(database: Path, output: Path, max_mae: float = 15.0) -> dict:
    if max_mae < 0:
        raise ValueError("max_mae must be nonnegative")
    with sqlite3.connect(database) as connection:
        data = pd.read_sql_query("SELECT * FROM sales ORDER BY order_date, order_id", connection)
    if len(data) < 30:
        raise ValueError("At least 30 sales rows are required")
    data["unit_price"] = data["unit_price"].astype(float)
    split = int(len(data) * 0.8)
    train_data, test_data = data.iloc[:split], data.iloc[split:]
    model = Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        ("numeric", "passthrough", ["units", "unit_price"]),
                        ("category", OneHotEncoder(handle_unknown="ignore"), ["category"]),
                    ]
                ),
            ),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=2)),
        ]
    )
    model.fit(train_data[FEATURES], train_data.revenue)
    predictions = model.predict(test_data[FEATURES])
    mae = float(mean_absolute_error(test_data.revenue, predictions))
    rmse = float(mean_squared_error(test_data.revenue, predictions) ** 0.5)
    output.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri())
    if mlflow.get_experiment_by_name("sales-revenue") is None:
        artifacts = Path(".runtime/artifacts").resolve()
        artifacts.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment("sales-revenue", artifact_location=artifacts.as_uri())
    mlflow.set_experiment("sales-revenue")
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "random_state": 42,
                "n_estimators": 100,
                "split": "chronological",
                "train_rows": len(train_data),
                "test_rows": len(test_data),
            }
        )
        mlflow.log_metrics({"mae": mae, "rmse": rmse})
        mlflow.set_tags(
            {
                "git_commit": os.environ.get("GITHUB_SHA", "local"),
                "purpose": "synthetic demonstration",
            }
        )
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=train_data[FEATURES].head(3),
            signature=infer_signature(train_data[FEATURES], model.predict(train_data[FEATURES])),
        )
        version = mlflow.register_model(info.model_uri, MODEL_NAME)
        accepted = mae <= max_mae
        if accepted:
            MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", version.version)
        report = {
            "run_id": run.info.run_id,
            "version": version.version,
            "model_uri": info.model_uri,
            "mae": mae,
            "rmse": rmse,
            "accepted": accepted,
        }
        mlflow.log_dict(report, "evaluation.json")
        (output / "evaluation.json").write_text(json.dumps(report, indent=2))
    if not accepted:
        raise ValueError(f"Model MAE {mae:.3f} exceeds promotion threshold {max_mae}")
    return report
