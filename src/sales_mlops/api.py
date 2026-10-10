"""Prediction API with readiness checks and Prometheus telemetry."""

import os
import time
from contextlib import asynccontextmanager

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from sales_mlops.training import MODEL_NAME, tracking_uri

REQUESTS = Counter("sales_prediction_requests_total", "Prediction calls", ["status"])
LATENCY = Histogram("sales_prediction_seconds", "Prediction latency")
REVENUE = Histogram(
    "sales_predicted_revenue", "Predicted revenue", buckets=(10, 50, 100, 250, 500, 1000)
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(tracking_uri())
    app.state.model = mlflow.pyfunc.load_model(
        os.environ.get("MODEL_URI", f"models:/{MODEL_NAME}@champion")
    )
    yield


app = FastAPI(title="Sales revenue model", lifespan=lifespan)


class Sale(BaseModel):
    units: int = Field(gt=0, le=10000)
    unit_price: float = Field(gt=0, le=1000000, allow_inf_nan=False)
    category: str = Field(min_length=1, max_length=100)


@app.get("/health")
def health():
    return {"status": "alive"}


@app.get("/ready")
def ready():
    if not hasattr(app.state, "model"):
        raise HTTPException(503, "Model unavailable")
    return {"status": "ready"}


@app.post("/predict")
def predict(sale: Sale):
    start = time.monotonic()
    try:
        revenue = float(app.state.model.predict(pd.DataFrame([sale.model_dump()]))[0])
    except Exception as error:
        REQUESTS.labels("error").inc()
        raise HTTPException(503, "Prediction unavailable") from error
    finally:
        LATENCY.observe(time.monotonic() - start)
    REQUESTS.labels("success").inc()
    REVENUE.observe(revenue)
    return {"predicted_revenue": revenue}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
