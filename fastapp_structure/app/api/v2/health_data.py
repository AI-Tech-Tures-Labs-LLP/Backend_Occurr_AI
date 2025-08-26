# app/api/v1/routes/health_data.py
from fastapi import APIRouter, Depends, HTTPException, Query
from datetime import datetime, timedelta,timezone
# from app.api.v1.auth import decode_token
from app.api.auth.auth import decode_token
from app.db.health_data_model import get_health_data_by_range,save_health_data,get_metric_summary, extract_metric_value
from fastapi.security import OAuth2PasswordBearer
from typing import Optional, Union, List, Dict
from pydantic import BaseModel
from bson import ObjectId
from pytz import UTC
from enum import Enum

from typing import List, Dict, Any, Tuple
from zoneinfo import ZoneInfo


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







  # if you prefer local tz, else stick to UTC

UTC = timezone.utc

def _month_start(dt: datetime) -> datetime:
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

def _next_month(dt: datetime) -> datetime:
    # move to 1st of next month
    year, month = dt.year + (dt.month // 12), (dt.month % 12) + 1
    return dt.replace(year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0)

def _week_start(dt: datetime) -> datetime:
    # ISO week start (Monday)
    d0 = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return d0 - timedelta(days=d0.weekday())

def _bucket_result(values: List[float], agg: str, decimals: int) -> float | None:
    if not values:
        return 0.0 if agg == "sum" else None
    if agg == "sum":
        return float(sum(values))
    # avg
    return round(sum(values) / len(values), decimals)

def _empty_value(agg: str):
    return 0.0 if agg == "sum" else None

def _label_hour(dt: datetime) -> str:
    return dt.strftime("%H:00")  # 00:00 .. 23:00

def _label_day(dt: datetime) -> str:
    # Example: "Mon 25"
    return dt.strftime("%a %d")

def _label_week(start: datetime, end: datetime) -> str:
    # Example: "Wk 1 (Aug 04–Aug 10)"
    # Keep short: "Wk 1"…"Wk 5"
    return f"Wk {start.isocalendar().week}"

def _label_month(dt: datetime) -> str:
    return dt.strftime("%b %Y")  # "Aug 2025"

def _collect_values_by_bucket(
    records: List[Dict[str, Any]],
    metric: MetricType,
    ranges: List[Tuple[datetime, datetime, str]],
) -> List[Dict[str, Any]]:
    """
    ranges: list of (start, end, label) for each bucket in chronological order
    """
    buckets: List[List[float]] = [[] for _ in ranges]

    for entry in records:
        ts = entry["timestamp"].astimezone(UTC)
        val = extract_metric_value(entry, metric)
        if val is None:
            continue

        # place into the first matching bucket (ranges are disjoint & ordered)
        # (Binary search would be more efficient but not necessary here)
        for i, (b_start, b_end, _) in enumerate(ranges):
            if b_start <= ts < b_end:
                buckets[i].append(val)
                break

    agg = METRIC_SETTINGS[metric]["aggregation"]  # "sum" | "avg"
    decimals = METRIC_SETTINGS[metric]["decimals"]

    result = []
    for (__, __, label), values in zip(ranges, buckets):
        y = _bucket_result(values, agg, decimals)
        result.append({"x": label, "y": y})
    return result

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
    agg = METRIC_SETTINGS[metric]["aggregation"]

    # ---- Build bucket ranges & labels
    ranges: List[Tuple[datetime, datetime, str]] = []

    if mode == TimeRange.daily:
        # Today 00:00 → 24 hourly buckets up to 24:00
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        for h in range(24):
            b_start = day_start + timedelta(hours=h)
            b_end = b_start + timedelta(hours=1)
            label = _label_hour(b_start)
            ranges.append((b_start, b_end, label))

        # fetch within the whole day
        start, end = ranges[0][0], ranges[-1][1]

    elif mode == TimeRange.weekly:
        # Last 7 *full* days including today as the last bucket
        # Buckets are per-day
        end_of_today = now.replace(hour=23, minute=59, second=59, microsecond=999999)
        start_day = (end_of_today - timedelta(days=6)).replace(hour=0, minute=0, second=0, microsecond=0)
        for i in range(7):
            b_start = start_day + timedelta(days=i)
            b_end = b_start + timedelta(days=1)
            label = _label_day(b_start)
            ranges.append((b_start, b_end, label))
        start, end = ranges[0][0], ranges[-1][1]

    elif mode == TimeRange.monthly:
        # Last 5 weeks → 5 buckets, each a 7-day window (not necessarily calendar-aligned)
        # Choose the start so the last bucket ends at "now"
        last_bucket_end = now.replace(minute=0, second=0, microsecond=0)
        first_bucket_start = (last_bucket_end - timedelta(weeks=5))
        # Create 5 consecutive week windows
        cur = first_bucket_start
        for i in range(5):
            b_start = cur
            b_end = b_start + timedelta(days=7)
            # If you prefer ISO week alignment, replace the above two lines with:
            # b_start = _week_start(cur)
            # b_end = b_start + timedelta(days=7)
            label = _label_week(b_start, b_end)
            ranges.append((b_start, b_end, label))
            cur = b_end
        start, end = ranges[0][0], ranges[-1][1]

    elif mode == TimeRange.yearly:
        # Last 12 months → 12 buckets (month-aligned)
        # Include current month as the last bucket
        this_month_start = _month_start(now)
        first = this_month_start
        # go back 11 more months
        months: List[datetime] = []
        cur = first
        for _ in range(12):
            months.append(cur)
            # move backward by 1 month
            prev_month_end = cur - timedelta(days=1)
            cur = _month_start(prev_month_end)
        months = list(reversed(months))  # chronological

        for m_start in months:
            m_end = _next_month(m_start)
            label = _label_month(m_start)
            ranges.append((m_start, m_end, label))
        start, end = ranges[0][0], ranges[-1][1]

    else:
        raise HTTPException(status_code=400, detail="Invalid mode")

    try:
        # Pull only once for the overall window
        records = get_health_data_by_range(username, start, end)
        if not records:
            # still return empty buckets so the chart renders fixed axes
            empty_y = _empty_value(agg)
            return {
                "graph": [{"x": label, "y": empty_y} for (_, _, label) in ranges],
                "metric": metric,
                "mode": mode,
                "message": "No data found for the specified period"
            }

        graph = _collect_values_by_bucket(records, metric, ranges)

        return {
            "graph": graph,
            "metric": metric,
            "mode": mode,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing health data: {str(e)}"
        )



# @router.get("/health/graph-data")
# def get_graph_data_api(
#     metric: MetricType,
#     mode: TimeRange,
#     token: str = Depends(oauth2_scheme)
# ):
#     valid, username = decode_token(token)
#     if not valid:
#         raise HTTPException(status_code=401, detail=username)

#     now = datetime.now(UTC)

#     # 1️⃣ Define start date based on mode
#     if mode == TimeRange.daily:
#         start = now.replace(hour=0, minute=0, second=0, microsecond=0)
#     elif mode == TimeRange.weekly:
#         start = now - timedelta(days=7)
#     elif mode == TimeRange.monthly:
#         start = now - timedelta(days=30)
#     elif mode == TimeRange.yearly:
#         start = now - timedelta(days=365)
#     else:
#         raise HTTPException(status_code=400, detail="Invalid mode")

#     try:
#         # 2️⃣ Get data from DB
#         records = get_health_data_by_range(username, start, now)
#         if not records:
#             return {"graph": [], "message": "No data found for the specified period"}

#         grouped = {}

#         for entry in records:
#             ts = entry["timestamp"].astimezone(UTC)

#             # 3️⃣ Custom time grouping key
#             if mode == TimeRange.daily:
#                 key = ts.strftime("%H:%M")  # Hourly
#             elif mode == TimeRange.weekly:
#                 key = ts.strftime("%A")  # Weekday name: Sunday, Monday, ...
#             elif mode == TimeRange.monthly:
#                 key = ts.strftime("%B")  # Month name: January, February, ...
#             elif mode == TimeRange.yearly:
#                 key = ts.strftime("%Y")  # Year: 2024, 2025, ...
#             else:
#                 key = ts.strftime("%Y-%m-%d")

#             if key not in grouped:
#                 grouped[key] = []

#             value = extract_metric_value(entry, metric)
#             if value is not None:
#                 grouped[key].append(value)

#         metric_settings = METRIC_SETTINGS[metric]
#         graph_data = []

#         for label, values in sorted(grouped.items()):
#             if not values:
#                 continue

#             if metric_settings["aggregation"] == "sum":
#                 y = sum(values)
#             else:  # avg
#                 y = round(sum(values) / len(values), metric_settings["decimals"])

#             graph_data.append({"x": label, "y": y})

#         return {
#             "graph": graph_data,
#             "metric": metric,
#             "mode": mode,
#         }

#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error processing health data: {str(e)}"
#         )