# app/api/v1/routes/health_data.py
# from fastapi import APIRouter, Depends, HTTPException, Query
# from datetime import datetime, timedelta,timezone
# # from app.api.v1.auth import decode_token
# from app.api.auth.auth import decode_token
# from app.db.health_data_model import get_health_data_by_range,save_health_data,get_metric_summary, extract_metric_value
# from fastapi.security import OAuth2PasswordBearer
# from typing import Optional, Union, List, Dict
# from pydantic import BaseModel
# from bson import ObjectId
# from pytz import UTC
# from enum import Enum

# from typing import List, Dict, Any, Tuple
# from zoneinfo import ZoneInfo


# # Configuration settings
# METRIC_SETTINGS = {
#     "steps": {"aggregation": "sum", "decimals": 0},
#     "heartRate": {"aggregation": "avg", "decimals": 0},
#     "spo2": {"aggregation": "avg", "decimals": 1},
#     "sleep": {"aggregation": "sum", "decimals": 0},
#     "calories": {"aggregation": "sum", "decimals": 0}
# }

# TIME_RANGES = {
#     "daily": {"days": 1, "format": "%H:%M"},
#     "weekly": {"days": 7, "format": "%Y-%m-%d"},
#     "monthly": {"days": 30, "format": "%Y-%m-%d"},
#     "yearly": {"days": 365, "format": "%Y-%m"}
# }

# class MetricType(str, Enum):
#     steps = "steps"
#     heartRate = "heartRate"
#     spo2 = "spo2"
#     sleep = "sleep"
#     calories = "calories"

# class TimeRange(str, Enum):
#     daily = "daily"
#     weekly = "weekly"
#     monthly = "monthly"
#     yearly = "yearly"

# router = APIRouter()
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# class HealthPayload(BaseModel):
#     steps: Dict[str, int] = {}
#     heartRate: Dict[str, int] = {}
#     spo2: Dict[str, float] = {}
#     sleep: Dict[str, int] = {}

# @router.post("/health/save")
# def save(payload: HealthPayload, token: str = Depends(oauth2_scheme)):
#     valid, username = decode_token(token)
#     if not valid:
#         raise HTTPException(status_code=401, detail=username)
#     save_health_data(username, payload.dict())
#     return {"message": "Saved"}



# @router.get("/health/summary")
# def get_metric_summary_api(
#     metric: str = Query(..., enum=["heartRate", "spo2", "steps", "sleep","calories"]),
#     mode: str = Query(..., enum=["daily", "weekly", "monthly","yearly"]),
#     token: str = Depends(oauth2_scheme)
# ):
#     valid, username = decode_token(token)
#     if not valid:
#         raise HTTPException(status_code=401, detail=username)

#     now = datetime.utcnow()
#     if mode == "daily":
#         start = now.replace(hour=0, minute=0, second=0, microsecond=0)
#     elif mode == "weekly":
#         start = now - timedelta(days=7)
#     elif mode == "monthly":
#         start = now - timedelta(days=30)
#     elif mode == "yearly":
#         start = now - timedelta(days=365)
#     else:
#         raise HTTPException(status_code=400, detail="Invalid mode")

#     end = now
#     summary = get_metric_summary(username, metric, start, end)
#     if not summary:
#         return {"message": "No data found"}
#     return {"metric": metric, "mode": mode, "summary": summary}







#   # if you prefer local tz, else stick to UTC

# UTC = timezone.utc

# def _month_start(dt: datetime) -> datetime:
#     return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

# def _next_month(dt: datetime) -> datetime:
#     # move to 1st of next month
#     year, month = dt.year + (dt.month // 12), (dt.month % 12) + 1
#     return dt.replace(year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0)

# def _week_start(dt: datetime) -> datetime:
#     # ISO week start (Monday)
#     d0 = dt.replace(hour=0, minute=0, second=0, microsecond=0)
#     return d0 - timedelta(days=d0.weekday())

# def _bucket_result(values: List[float], agg: str, decimals: int) -> float | None:
#     if not values:
#         return 0.0 if agg == "sum" else None
#     if agg == "sum":
#         return float(sum(values))
#     # avg
#     return round(sum(values) / len(values), decimals)

# def _empty_value(agg: str):
#     return 0.0 if agg == "sum" else None

# def _label_hour(dt: datetime) -> str:
#     return dt.strftime("%H:00")  # 00:00 .. 23:00

# def _label_day(dt: datetime) -> str:
#     # Example: "Mon 25"
#     return dt.strftime("%a %d")

# def _label_week(start: datetime, end: datetime) -> str:
#     # Example: "Wk 1 (Aug 04–Aug 10)"
#     # Keep short: "Wk 1"…"Wk 5"
#     return f"Wk {start.isocalendar().week}"

# def _label_month(dt: datetime) -> str:
#     return dt.strftime("%b %Y")  # "Aug 2025"

# def _collect_values_by_bucket(
#     records: List[Dict[str, Any]],
#     metric: MetricType,
#     ranges: List[Tuple[datetime, datetime, str]],
# ) -> List[Dict[str, Any]]:
#     """
#     ranges: list of (start, end, label) for each bucket in chronological order
#     """
#     buckets: List[List[float]] = [[] for _ in ranges]

#     for entry in records:
#         ts = entry["timestamp"].astimezone(UTC)
#         val = extract_metric_value(entry, metric)
#         if val is None:
#             continue

#         # place into the first matching bucket (ranges are disjoint & ordered)
#         # (Binary search would be more efficient but not necessary here)
#         for i, (b_start, b_end, _) in enumerate(ranges):
#             if b_start <= ts < b_end:
#                 buckets[i].append(val)
#                 break

#     agg = METRIC_SETTINGS[metric]["aggregation"]  # "sum" | "avg"
#     decimals = METRIC_SETTINGS[metric]["decimals"]

#     result = []
#     for (__, __, label), values in zip(ranges, buckets):
#         y = _bucket_result(values, agg, decimals)
#         result.append({"x": label, "y": y})
#     return result

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
#     agg = METRIC_SETTINGS[metric]["aggregation"]

#     # ---- Build bucket ranges & labels
#     ranges: List[Tuple[datetime, datetime, str]] = []

#     if mode == TimeRange.daily:
#         # Today 00:00 → 24 hourly buckets up to 24:00
#         day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
#         for h in range(24):
#             b_start = day_start + timedelta(hours=h)
#             b_end = b_start + timedelta(hours=1)
#             label = _label_hour(b_start)
#             ranges.append((b_start, b_end, label))

#         # fetch within the whole day
#         start, end = ranges[0][0], ranges[-1][1]

#     elif mode == TimeRange.weekly:
#         # Last 7 *full* days including today as the last bucket
#         # Buckets are per-day
#         end_of_today = now.replace(hour=23, minute=59, second=59, microsecond=999999)
#         start_day = (end_of_today - timedelta(days=6)).replace(hour=0, minute=0, second=0, microsecond=0)
#         for i in range(7):
#             b_start = start_day + timedelta(days=i)
#             b_end = b_start + timedelta(days=1)
#             label = _label_day(b_start)
#             ranges.append((b_start, b_end, label))
#         start, end = ranges[0][0], ranges[-1][1]

#     elif mode == TimeRange.monthly:
#         # Last 5 weeks → 5 buckets, each a 7-day window (not necessarily calendar-aligned)
#         # Choose the start so the last bucket ends at "now"
#         last_bucket_end = now.replace(minute=0, second=0, microsecond=0)
#         first_bucket_start = (last_bucket_end - timedelta(weeks=5))
#         # Create 5 consecutive week windows
#         cur = first_bucket_start
#         for i in range(5):
#             b_start = cur
#             b_end = b_start + timedelta(days=7)
#             # If you prefer ISO week alignment, replace the above two lines with:
#             # b_start = _week_start(cur)
#             # b_end = b_start + timedelta(days=7)
#             label = _label_week(b_start, b_end)
#             ranges.append((b_start, b_end, label))
#             cur = b_end
#         start, end = ranges[0][0], ranges[-1][1]

#     elif mode == TimeRange.yearly:
#         # Last 12 months → 12 buckets (month-aligned)
#         # Include current month as the last bucket
#         this_month_start = _month_start(now)
#         first = this_month_start
#         # go back 11 more months
#         months: List[datetime] = []
#         cur = first
#         for _ in range(12):
#             months.append(cur)
#             # move backward by 1 month
#             prev_month_end = cur - timedelta(days=1)
#             cur = _month_start(prev_month_end)
#         months = list(reversed(months))  # chronological

#         for m_start in months:
#             m_end = _next_month(m_start)
#             label = _label_month(m_start)
#             ranges.append((m_start, m_end, label))
#         start, end = ranges[0][0], ranges[-1][1]

#     else:
#         raise HTTPException(status_code=400, detail="Invalid mode")

#     try:
#         # Pull only once for the overall window
#         records = get_health_data_by_range(username, start, end)
#         if not records:
#             # still return empty buckets so the chart renders fixed axes
#             empty_y = _empty_value(agg)
#             return {
#                 "graph": [{"x": label, "y": empty_y} for (_, _, label) in ranges],
#                 "metric": metric,
#                 "mode": mode,
#                 "message": "No data found for the specified period"
#             }

#         graph = _collect_values_by_bucket(records, metric, ranges)

#         return {
#             "graph": graph,
#             "metric": metric,
#             "mode": mode,
#         }

#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error processing health data: {str(e)}"
#         )










# app/api/v1/health.py  (example path)

# from fastapi import APIRouter, Depends, HTTPException, Query
# from fastapi.security import OAuth2PasswordBearer
# from pydantic import BaseModel
# from typing import Optional, Union, List, Dict, Any, Tuple
# from enum import Enum

# from datetime import datetime, timedelta, timezone
# from zoneinfo import ZoneInfo
# from bson import ObjectId
# from pymongo import MongoClient
# import os

# # ---- Your existing auth & DB helpers ----
# from app.api.auth.auth import decode_token
# from app.db.health_data_model import (
#     get_health_data_by_range,
#     save_health_data,
#     get_metric_summary,
#     extract_metric_value,
# )

# # ===================== Config =====================

# router = APIRouter()
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# UTC = timezone.utc
# MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
# _mclient = MongoClient(MONGO_URL)
# _mdb = _mclient["techjewel"]
# user_dates_collection = _mdb["user_dates"]  # NEW collection

# # Metric config (how to aggregate + decimals)
# METRIC_SETTINGS: Dict[str, Dict[str, Union[str, int]]] = {
#     "steps":     {"aggregation": "sum", "decimals": 0},
#     "heartrate": {"aggregation": "avg", "decimals": 0},
#     "spo2":      {"aggregation": "avg", "decimals": 1},
#     "sleep":     {"aggregation": "sum", "decimals": 0},
#     "calories":  {"aggregation": "sum", "decimals": 0},
# }

# class MetricType(str, Enum):
#     steps = "steps"
#     heartrate = "heartrate"
#     spo2 = "spo2"
#     sleep = "sleep"
#     calories = "calories"

# class TimeRange(str, Enum):
#     daily = "daily"
#     weekly = "weekly"
#     monthly = "monthly"
#     yearly = "yearly"

# # ===================== Models =====================

# class HealthPayload(BaseModel):
#     steps: Dict[str, int] = {}
#     heartrate: Dict[str, int] = {}
#     spo2: Dict[str, float] = {}
#     sleep: Dict[str, int] = {}
#     calories: Dict[str, int] = {}

# class UserDateEntry(BaseModel):
#     username: str
#     date: Optional[datetime] = None  # if None, we'll default to now (UTC)

# # ===================== User Anchor Date =====================

# def get_user_anchor_date(username: str) -> datetime:
#     """
#     Returns the latest saved date for this user from user_dates collection.
#     Falls back to current UTC time if nothing found.
#     Ensures tz-aware UTC.
#     """
#     doc = user_dates_collection.find_one(
#         {"username": username},
#         sort=[("date", -1), ("created_at", -1)]
#     )
#     if doc and "date" in doc:
#         dt = doc["date"]
#         if isinstance(dt, datetime):
#             return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
#         if isinstance(dt, str) and dt.strip():
#             try:
#                 # parse simple 'YYYY-MM-DD' or ISO; treat as UTC if naive
#                 parsed = datetime.fromisoformat(dt.replace("Z", "+00:00"))
#                 return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
#             except Exception:
#                 pass
#     return datetime.now(UTC)

# # ===================== Label / Bucket Helpers =====================

# def _month_start(dt: datetime) -> datetime:
#     return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC)

# def _next_month(dt: datetime) -> datetime:
#     year, month = dt.year + (dt.month // 12), (dt.month % 12) + 1
#     return dt.replace(year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC)

# def _week_start(dt: datetime) -> datetime:
#     d0 = dt.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC)
#     return d0 - timedelta(days=d0.weekday())

# def _bucket_result(values: List[float], agg: str, decimals: int) -> Optional[float]:
#     if not values:
#         return 0.0 if agg == "sum" else None
#     if agg == "sum":
#         return float(sum(values))
#     # avg
#     return round(sum(values) / len(values), decimals)

# def _empty_value(agg: str):
#     return 0.0 if agg == "sum" else None

# def _label_hour(dt: datetime) -> str:
#     return dt.strftime("%H:00")

# def _label_day(dt: datetime) -> str:
#     return dt.strftime("%a %d")

# def _label_week(start: datetime, end: datetime) -> str:
#     return f"Wk {start.isocalendar().week}"

# def _label_month(dt: datetime) -> str:
#     return dt.strftime("%b %Y")

# def _collect_values_by_bucket(
#     records: List[Dict[str, Any]],
#     metric: MetricType,
#     ranges: List[Tuple[datetime, datetime, str]],
# ) -> List[Dict[str, Any]]:
#     buckets: List[List[float]] = [[] for _ in ranges]
#     for entry in records:
#         ts = entry["timestamp"]
#         if ts.tzinfo is None:
#             ts = ts.replace(tzinfo=UTC)
#         else:
#             ts = ts.astimezone(UTC)
#         val = extract_metric_value(entry, metric)
#         if val is None:
#             continue
#         for i, (b_start, b_end, _) in enumerate(ranges):
#             if b_start <= ts < b_end:
#                 buckets[i].append(val)
#                 break

#     agg = METRIC_SETTINGS[str(metric)]["aggregation"]  # type: ignore
#     decimals = METRIC_SETTINGS[str(metric)]["decimals"]  # type: ignore

#     result = []
#     for (__, __, label), values in zip(ranges, buckets):
#         y = _bucket_result(values, agg, decimals)  # type: ignore
#         result.append({"x": label, "y": y})
#     return result



# # ===================== Health APIs =====================

# @router.post("/health/save")
# def save(payload: HealthPayload, token: str = Depends(oauth2_scheme)):
#     """
#     Save health payload for the authenticated user.
#     """
#     valid, username = decode_token(token)
#     if not valid:
#         raise HTTPException(status_code=401, detail=username)
#     save_health_data(username, payload.dict())
#     return {"message": "Saved"}

# @router.get("/health/summary")
# def get_metric_summary_api(
#     metric: str = Query(..., enum=["heartrate", "spo2", "steps", "sleep", "calories"]),
#     mode: str = Query(..., enum=["daily", "weekly", "monthly", "yearly"]),
#     token: str = Depends(oauth2_scheme),
# ):
#     """
#     Summary for a metric over a window anchored to the user's latest saved date.
#     """
#     valid, username = decode_token(token)
#     if not valid:
#         raise HTTPException(status_code=401, detail=username)

#     base = get_user_anchor_date(username)  # <-- anchor from user_dates

#     if mode == "daily":
#         start = base.replace(hour=0, minute=0, second=0, microsecond=0)
#     elif mode == "weekly":
#         start = (base - timedelta(days=7)).replace(microsecond=0)
#     elif mode == "monthly":
#         start = (base - timedelta(days=30)).replace(microsecond=0)
#     elif mode == "yearly":
#         start = (base - timedelta(days=365)).replace(microsecond=0)
#     else:
#         raise HTTPException(status_code=400, detail="Invalid mode")

#     end = base
#     summary = get_metric_summary(username, metric, start, end)
#     if not summary:
#         return {"metric": metric, "mode": mode, "anchor_date": base.isoformat(), "message": "No data found"}

#     return {"metric": metric, "mode": mode, "anchor_date": base.isoformat(), "summary": summary}

# @router.get("/health/graph-data")
# def get_graph_data_api(
#     metric: MetricType,
#     mode: TimeRange,
#     token: str = Depends(oauth2_scheme),
# ):
#     """
#     Graph data buckets for a metric, anchored to the user's latest saved date.
#     """
#     valid, username = decode_token(token)
#     if not valid:
#         raise HTTPException(status_code=401, detail=username)

#     base = get_user_anchor_date(username)  # <-- anchor from user_dates
#     agg = METRIC_SETTINGS[str(metric)]["aggregation"]  # type: ignore

#     ranges: List[Tuple[datetime, datetime, str]] = []

#     if mode == TimeRange.daily:
#         day_start = base.replace(hour=0, minute=0, second=0, microsecond=0)
#         for h in range(24):
#             b_start = day_start + timedelta(hours=h)
#             b_end = b_start + timedelta(hours=1)
#             ranges.append((b_start, b_end, _label_hour(b_start)))
#         start, end = ranges[0][0], ranges[-1][1]

#     elif mode == TimeRange.weekly:
#         end_of_anchor = base.replace(hour=23, minute=59, second=59, microsecond=999999)
#         start_day = (end_of_anchor - timedelta(days=6)).replace(hour=0, minute=0, second=0, microsecond=0)
#         for i in range(7):
#             b_start = start_day + timedelta(days=i)
#             b_end = b_start + timedelta(days=1)
#             ranges.append((b_start, b_end, _label_day(b_start)))
#         start, end = ranges[0][0], ranges[-1][1]

#     elif mode == TimeRange.monthly:
#         last_bucket_end = base.replace(minute=0, second=0, microsecond=0)
#         first_bucket_start = last_bucket_end - timedelta(weeks=5)
#         cur = first_bucket_start
#         for _ in range(5):
#             b_start = cur
#             b_end = b_start + timedelta(days=7)
#             ranges.append((b_start, b_end, _label_week(b_start, b_end)))
#             cur = b_end
#         start, end = ranges[0][0], ranges[-1][1]

#     elif mode == TimeRange.yearly:
#         this_month_start = _month_start(base)
#         months: List[datetime] = []
#         cur = this_month_start
#         for _ in range(12):
#             months.append(cur)
#             cur = _month_start(cur - timedelta(days=1))
#         months = list(reversed(months))
#         for m_start in months:
#             m_end = _next_month(m_start)
#             ranges.append((m_start, m_end, _label_month(m_start)))
#         start, end = ranges[0][0], ranges[-1][1]

#     else:
#         raise HTTPException(status_code=400, detail="Invalid mode")

#     try:
#         records = get_health_data_by_range(username, start, end)
#         if not records:
#             empty_y = _empty_value(agg)  # type: ignore
#             return {
#                 "graph": [{"x": label, "y": empty_y} for (_, _, label) in ranges],
#                 "metric": metric,
#                 "mode": mode,
#                 "anchor_date": base.isoformat(),
#                 "message": "No data found for the specified period",
#             }

#         graph = _collect_values_by_bucket(records, metric, ranges)
#         return {"graph": graph, "metric": metric, "mode": mode, "anchor_date": base.isoformat()}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing health data: {str(e)}")






# app/api/v2/health_data.py

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Optional, Union, List, Dict, Any, Tuple
from enum import Enum
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pymongo import MongoClient
import os

# ---- Your existing auth & DB helpers ----
from app.api.auth.auth import decode_token
from app.db.health_data_model import (
    get_health_data_by_range,  # (username: str, start: dt, end: dt) -> List[dict]
    save_health_data,          # (username: str, payload: dict) -> None
    get_metric_summary,        # (username: str, metric: str, start: dt, end: dt) -> Any
    extract_metric_value,      # Optional helper you already have; we fallback to it
)

# ===================== Config =====================

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

UTC = timezone.utc
LOCAL_TZ = os.getenv("LOCAL_TZ", "UTC")  # used only for naive timestamps

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
_mclient = MongoClient(MONGO_URL)
_mdb = _mclient["techjewel"]
user_dates_collection = _mdb["user_dates"]      # anchor-date store

# Metric config (how to aggregate + decimals)
# Canonical keys match your DB docs: "heartrate", "spo2", ...
METRIC_SETTINGS: Dict[str, Dict[str, Union[str, int]]] = {
    "steps":     {"aggregation": "sum", "decimals": 0},
    "heartrate": {"aggregation": "avg", "decimals": 0},
    "spo2":      {"aggregation": "avg", "decimals": 1},
    "sleep":     {"aggregation": "sum", "decimals": 0},
    "calories":  {"aggregation": "sum", "decimals": 0},
}

class MetricType(str, Enum):
    steps = "steps"
    heartrate = "heartrate"
    spo2 = "spo2"
    sleep = "sleep"
    calories = "calories"

class TimeRange(str, Enum):
    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"
    yearly = "yearly"

# ===================== Models =====================

class HealthPayload(BaseModel):
    # If you use /health/save, keep these aligned to what save_health_data expects
    steps: Dict[str, int] = {}
    heartrate: Dict[str, int] = {}
    spo2: Dict[str, float] = {}
    sleep: Dict[str, int] = {}
    calories: Dict[str, int] = {}

# ===================== Anchor date (user_dates) =====================

def get_user_anchor_date(username: str) -> datetime:
    """
    Returns the latest saved date for this user from user_dates.
    Falls back to current UTC time if nothing found. Ensures tz-aware UTC.
    """
    doc = user_dates_collection.find_one(
        {"username": username},
        sort=[("date", -1), ("created_at", -1)]
    )
    if doc and "date" in doc:
        dt = doc["date"]
        if isinstance(dt, datetime):
            return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
        if isinstance(dt, str) and dt.strip():
            try:
                parsed = datetime.fromisoformat(dt.replace("Z", "+00:00"))
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
            except Exception:
                pass
    return datetime.now(UTC)

# ===================== Metric + timestamp helpers =====================

# Normalize aliases to canonical settings key
_ALIAS_TO_KEY = {
    "heartrate": "heartrate",
    "heart_rate": "heartrate",
    "hr": "heartrate",
    "steps": "steps",
    "spo2": "spo2",
    "sleep": "sleep",
    "calories": "calories",
}

def _metric_key(m: Union[MetricType, str]) -> str:
    """Enum or string -> canonical key used in METRIC_SETTINGS/DB."""
    key = m.value if isinstance(m, MetricType) else str(m)
    low = key.replace("-", "").replace(" ", "").lower()
    return _ALIAS_TO_KEY.get(low, key)

def _coerce_ts(value: Any) -> datetime:
    """
    Accept datetime/string/epoch -> tz-aware UTC datetime.
    If naive, assume LOCAL_TZ (default 'UTC'), then convert to UTC.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=ZoneInfo(LOCAL_TZ)).astimezone(UTC)
        return value.astimezone(UTC)

    if isinstance(value, (int, float)):
        sec = value / 1000.0 if value > 10_000_000_000 else value
        return datetime.fromtimestamp(sec, tz=UTC)

    if isinstance(value, str):
        # Use fromisoformat best-effort (can be extended with dateutil if needed)
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            from dateutil import parser as dtparser  # fallback if not strict ISO
            dt = dtparser.parse(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo(LOCAL_TZ))
        return dt.astimezone(UTC)

    raise ValueError(f"Unrecognized timestamp type: {type(value)}")

def _get_ts(entry: Dict[str, Any]) -> datetime:
    """Find a timestamp field in common keys and return tz-aware UTC."""
    for k in ("timestamp", "ts", "time", "datetime", "date"):
        if k in entry:
            return _coerce_ts(entry[k])
    raise KeyError("No timestamp field found in entry (timestamp/ts/time/datetime/date).")

def _safe_extract_value(entry: Dict[str, Any], metric: Union[MetricType, str]) -> Optional[float]:
    """
    Extract a numeric value for the metric from the record.
    Supports your doc shape:
      { metric: "heartrate", value: 138, timestamp: ... }
    Falls back to existing helper and common shapes.
    """
    mkey = _metric_key(metric)

    # Shape A: per-doc metric + numeric value
    if str(entry.get("metric", "")).lower() == mkey.lower():
        v = entry.get("value")
        if isinstance(v, (int, float)):
            return float(v)

    # Shape B: fallback to your helper if it supports other schemas
    try:
        v = extract_metric_value(entry, mkey)  # if your helper expects Enum, pass `metric`
        if isinstance(v, (int, float)):
            return float(v)
    except Exception:
        pass

    # Shape C: direct numeric under property (rare, but safe)
    for k in (mkey, mkey.lower(), "heartRate"):
        if k in entry and isinstance(entry[k], (int, float)):
            return float(entry[k])

    return None

# ===================== Bucketing helpers =====================

def _month_start(dt: datetime) -> datetime:
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC)

def _next_month(dt: datetime) -> datetime:
    year, month = dt.year + (dt.month // 12), (dt.month % 12) + 1
    return dt.replace(year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC)

def _bucket_result(values: List[float], agg: str, decimals: int) -> Optional[float]:
    if not values:
        return 0.0 if agg == "sum" else None  # keep None for avg to show gaps
    if agg == "sum":
        return float(sum(values))
    return round(sum(values) / len(values), decimals)

def _empty_value(agg: str):
    return 0.0 if agg == "sum" else None

def _label_hour(dt: datetime) -> str:
    return dt.strftime("%H:00")

def _label_day(dt: datetime) -> str:
    return dt.strftime("%a %d")

def _label_week(start: datetime, end: datetime) -> str:
    return f"Wk {start.isocalendar().week}"

def _label_month(dt: datetime) -> str:
    return dt.strftime("%b %Y")

def _collect_values_by_bucket(
    records: List[Dict[str, Any]],
    metric: MetricType,
    ranges: List[Tuple[datetime, datetime, str]],
) -> List[Dict[str, Any]]:
    buckets: List[List[float]] = [[] for _ in ranges]

    for entry in records:
        # robust timestamp
        try:
            ts = _get_ts(entry)
        except Exception:
            continue

        # robust numeric extraction
        val = _safe_extract_value(entry, metric)
        if val is None:
            continue

        # place into range
        for i, (b_start, b_end, _) in enumerate(ranges):
            if b_start <= ts < b_end:
                buckets[i].append(val)
                break

    mkey = _metric_key(metric)
    agg = METRIC_SETTINGS[mkey]["aggregation"]  # type: ignore
    decimals = METRIC_SETTINGS[mkey]["decimals"]  # type: ignore

    out = []
    for (__, __, label), values in zip(ranges, buckets):
        y = _bucket_result(values, agg, decimals)  # type: ignore
        out.append({"x": label, "y": y})
    return out

# ===================== APIs =====================

@router.post("/health/save")
def save(payload: HealthPayload, token: str = Depends(oauth2_scheme)):
    """
    Save health payload for the authenticated user.
    """
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)
    save_health_data(username, payload.dict())
    return {"message": "Saved"}

@router.get("/health/summary")
def get_metric_summary_api(
    metric: str = Query(..., enum=["heartrate", "spo2", "steps", "sleep", "calories"]),
    mode: str = Query(..., enum=["daily", "weekly", "monthly", "yearly"]),
    token: str = Depends(oauth2_scheme),
):
    """
    Summary for a metric over a window anchored to the user's latest saved date.
    """
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)

    base = get_user_anchor_date(username)

    if mode == "daily":
        start = base.replace(hour=0, minute=0, second=0, microsecond=0)
    elif mode == "weekly":
        start = (base - timedelta(days=7)).replace(microsecond=0)
    elif mode == "monthly":
        start = (base - timedelta(days=30)).replace(microsecond=0)
    elif mode == "yearly":
        start = (base - timedelta(days=365)).replace(microsecond=0)
    else:
        raise HTTPException(status_code=400, detail="Invalid mode")

    end = base
    metric_key = _metric_key(metric)
    summary = get_metric_summary(username, metric_key, start, end)
    if not summary:
        return {"metric": metric_key, "mode": mode, "anchor_date": base.isoformat(), "message": "No data found"}

    return {"metric": metric_key, "mode": mode, "anchor_date": base.isoformat(), "summary": summary}

@router.get("/health/graph-data")
def get_graph_data_api(
    metric: MetricType,
    mode: TimeRange,
    token: str = Depends(oauth2_scheme),
):
    """
    Graph data buckets for a metric, anchored to the user's latest saved date.
    """
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)

    base = get_user_anchor_date(username)
    metric_key = _metric_key(metric)
    agg = METRIC_SETTINGS[metric_key]["aggregation"]  # type: ignore

    ranges: List[Tuple[datetime, datetime, str]] = []

    if mode == TimeRange.daily:
        day_start = base.replace(hour=0, minute=0, second=0, microsecond=0)
        for h in range(24):
            b_start = day_start + timedelta(hours=h)
            b_end = b_start + timedelta(hours=1)
            ranges.append((b_start, b_end, _label_hour(b_start)))
        start, end = ranges[0][0], ranges[-1][1]

    elif mode == TimeRange.weekly:
        end_of_anchor = base.replace(hour=23, minute=59, second=59, microsecond=999999)
        start_day = (end_of_anchor - timedelta(days=6)).replace(hour=0, minute=0, second=0, microsecond=0)
        for i in range(7):
            b_start = start_day + timedelta(days=i)
            b_end = b_start + timedelta(days=1)
            ranges.append((b_start, b_end, _label_day(b_start)))
        start, end = ranges[0][0], ranges[-1][1]

    elif mode == TimeRange.monthly:
        last_bucket_end = base.replace(minute=0, second=0, microsecond=0)
        first_bucket_start = last_bucket_end - timedelta(weeks=5)
        cur = first_bucket_start
        for _ in range(5):
            b_start = cur
            b_end = b_start + timedelta(days=7)
            ranges.append((b_start, b_end, _label_week(b_start, b_end)))
            cur = b_end
        start, end = ranges[0][0], ranges[-1][1]

    elif mode == TimeRange.yearly:
        this_month_start = _month_start(base)
        months: List[datetime] = []
        cur = this_month_start
        for _ in range(12):
            months.append(cur)
            cur = _month_start(cur - timedelta(days=1))
        months = list(reversed(months))
        for m_start in months:
            m_end = _next_month(m_start)
            ranges.append((m_start, m_end, _label_month(m_start)))
        start, end = ranges[0][0], ranges[-1][1]

    else:
        raise HTTPException(status_code=400, detail="Invalid mode")

    try:
        # Pull data once for the window
        records = get_health_data_by_range(username, start, end)

        # Filter by metric to avoid mixing
        records = [r for r in records if str(r.get("metric", "")).lower() == metric_key.lower()]

        if not records:
            empty_y = _empty_value(agg)  # type: ignore
            return {
                "graph": [{"x": label, "y": empty_y} for (_, _, label) in ranges],
                "metric": metric_key,
                "mode": mode,
                "anchor_date": base.isoformat(),
                "message": "No data found for the specified period",
            }

        graph = _collect_values_by_bucket(records, metric, ranges)
        return {"graph": graph, "metric": metric_key, "mode": mode, "anchor_date": base.isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing health data: {str(e)}")
