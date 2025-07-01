# app/api/v1/routes/health_data.py
from fastapi import APIRouter, Depends, HTTPException, Query
from datetime import datetime, timedelta
# from app.api.v1.auth import decode_token
from app.api.auth.auth import decode_token
from app.db.health_data_model import get_health_data_by_range,save_health_data,get_metric_summary, extract_metric_value
from fastapi.security import OAuth2PasswordBearer
from typing import Optional, Union, List, Dict
from pydantic import BaseModel
from bson import ObjectId
from pytz import UTC
from enum import Enum

# Configuration settings
METRIC_SETTINGS = {
    "steps": {"aggregation": "sum", "decimals": 0},
    "heartRate": {"aggregation": "avg", "decimals": 0},
    "spo2": {"aggregation": "avg", "decimals": 1},
    "sleep": {"aggregation": "sum", "decimals": 0},
    "calories": {"aggregation": "sum", "decimals": 0}
}

TIME_RANGES = {
    "daily": {"days": 1, "format": "%H:%M"},
    "weekly": {"days": 7, "format": "%Y-%m-%d"},
    "monthly": {"days": 30, "format": "%Y-%m-%d"},
    "yearly": {"days": 365, "format": "%Y-%m"}
}

class MetricType(str, Enum):
    steps = "steps"
    heartRate = "heartRate"
    spo2 = "spo2"
    sleep = "sleep"
    calories = "calories"

class TimeRange(str, Enum):
    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"
    yearly = "yearly"

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class HealthPayload(BaseModel):
    steps: Dict[str, int] = {}
    heartRate: Dict[str, int] = {}
    spo2: Dict[str, float] = {}
    sleep: Dict[str, int] = {}

@router.post("/health/save")
def save(payload: HealthPayload, token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)
    save_health_data(username, payload.dict())
    return {"message": "Saved"}



@router.get("/health/summary")
def get_metric_summary_api(
    metric: str = Query(..., enum=["heartRate", "spo2", "steps", "sleep","calories"]),
    mode: str = Query(..., enum=["daily", "weekly", "monthly","yearly"]),
    token: str = Depends(oauth2_scheme)
):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)

    now = datetime.utcnow()
    if mode == "daily":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif mode == "weekly":
        start = now - timedelta(days=7)
    elif mode == "monthly":
        start = now - timedelta(days=30)
    elif mode == "yearly":
        start = now - timedelta(days=365)
    else:
        raise HTTPException(status_code=400, detail="Invalid mode")

    end = now
    summary = get_metric_summary(username, metric, start, end)
    if not summary:
        return {"message": "No data found"}
    return {"metric": metric, "mode": mode, "summary": summary}


@router.get("/health/graph-data")
def get_graph_data_api(
    metric: MetricType,
    mode: TimeRange,
    token: str = Depends(oauth2_scheme)
):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)

    now = datetime.now(UTC)

    # 1️⃣ Define start date based on mode
    if mode == TimeRange.daily:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif mode == TimeRange.weekly:
        start = now - timedelta(days=7)
    elif mode == TimeRange.monthly:
        start = now - timedelta(days=30)
    elif mode == TimeRange.yearly:
        start = now - timedelta(days=365)
    else:
        raise HTTPException(status_code=400, detail="Invalid mode")

    try:
        # 2️⃣ Get data from DB
        records = get_health_data_by_range(username, start, now)
        if not records:
            return {"graph": [], "message": "No data found for the specified period"}

        grouped = {}

        for entry in records:
            ts = entry["timestamp"].astimezone(UTC)

            # 3️⃣ Custom time grouping key
            if mode == TimeRange.daily:
                key = ts.strftime("%H:%M")  # Hourly
            elif mode == TimeRange.weekly:
                key = ts.strftime("%A")  # Weekday name: Sunday, Monday, ...
            elif mode == TimeRange.monthly:
                key = ts.strftime("%B")  # Month name: January, February, ...
            elif mode == TimeRange.yearly:
                key = ts.strftime("%Y")  # Year: 2024, 2025, ...
            else:
                key = ts.strftime("%Y-%m-%d")

            if key not in grouped:
                grouped[key] = []

            value = extract_metric_value(entry, metric)
            if value is not None:
                grouped[key].append(value)

        metric_settings = METRIC_SETTINGS[metric]
        graph_data = []

        for label, values in sorted(grouped.items()):
            if not values:
                continue

            if metric_settings["aggregation"] == "sum":
                y = sum(values)
            else:  # avg
                y = round(sum(values) / len(values), metric_settings["decimals"])

            graph_data.append({"x": label, "y": y})

        return {
            "graph": graph_data,
            "metric": metric,
            "mode": mode,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing health data: {str(e)}"
        )