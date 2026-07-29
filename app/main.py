"""FastAPI application exposing the Rossmann sales forecasting model.

Endpoints:
    GET /health: Liveness probe.
    POST /predict: Predict daily sales for a single store/day record.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import MODEL_FILE
from src.features import clean_feature_names, encode_categoricals
from src.models import predict_ensemble

app = FastAPI(
    title="Rossmann Sales Forecast",
    description="API for predicting daily sales for Rossmann stores (Kaggle).",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    """Input payload describing a single store/day to forecast.

    Attributes:
        store: Store identifier.
        day_of_week: ISO day of week (1=Monday .. 7=Sunday).
        date: Calendar date of the forecast.
        open: Whether the store is open (0 or 1).
        promo: Whether a promo is running that day (0 or 1).
        state_holiday: State holiday code ("0", "a", "b", "c").
        school_holiday: Whether schools are closed (0 or 1).
        store_type: Store model code ("a", "b", "c", "d").
        assortment: Assortment level code ("a", "b", "c").
        competition_distance: Distance to nearest competitor in meters.
    """

    store: int = Field(..., gt=0)
    day_of_week: int = Field(..., ge=1, le=7)
    date: date
    open: int = Field(..., ge=0, le=1)
    promo: int = Field(..., ge=0, le=1)
    state_holiday: str = "0"
    school_holiday: int = Field(0, ge=0, le=1)
    store_type: str = "a"
    assortment: str = "a"
    competition_distance: float | None = None


class PredictionResponse(BaseModel):
    """Prediction output payload."""

    store: int
    predicted_sales: float


@lru_cache(maxsize=1)
def _load_model_bundle() -> dict:
    """Load and cache the persisted model ensemble."""
    if not MODEL_FILE.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model artefact not found at {MODEL_FILE}. Run training first.",
        )
    return joblib.load(MODEL_FILE)


@app.get("/health")
def health() -> dict[str, str]:
    """Report service liveness."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict daily sales for a single store/day record."""
    bundle = _load_model_bundle()
    models = bundle["models"]
    feature_columns = bundle["feature_columns"]

    row = pd.DataFrame(
        [
            {
                "Store": request.store,
                "DayOfWeek": request.day_of_week,
                "Date": pd.Timestamp(request.date),
                "Open": request.open,
                "Promo": request.promo,
                "StateHoliday": request.state_holiday,
                "SchoolHoliday": request.school_holiday,
                "StoreType": request.store_type,
                "Assortment": request.assortment,
                "CompetitionDistance": request.competition_distance,
            }
        ]
    )

    missing_columns = [c for c in feature_columns if c not in row.columns]
    for col in missing_columns:
        row[col] = np.nan

    row = encode_categoricals(row)

    x = clean_feature_names(row[feature_columns])
    prediction = float(predict_ensemble(models, x)[0])

    return PredictionResponse(store=request.store, predicted_sales=prediction)
