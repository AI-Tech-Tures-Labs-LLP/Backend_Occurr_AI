# remap_from_db.py
from fastapi import APIRouter, FastAPI, Query, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel,Field
from datetime import datetime, timedelta, timezone
from pymongo import MongoClient
from typing import Optional, Dict, Any, List
from pymongo import MongoClient
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv
load_dotenv()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/auth/token")


router = APIRouter()

# Access your databasclient["techjewel"]
# ---------------- CONFIG ----------------
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
# client = MongoClient(MONGO_URL)
DB_NAME = "techjewel"
COLLECTION_NAME = "health_data_entries"
# ----------------------------------------



client = MongoClient(MONGO_URL)
db = client[DB_NAME]
col = db[COLLECTION_NAME]



class ChainRemapRequest(BaseModel):
    metric: str = Field(..., description="The metric to remap (e.g., 'steps')")
    username: Optional[str] = Field(None, description="Optional username to filter by")
    target_start_date: str = Field(..., description="Target start date in YYYY-MM-DD format")



# === FIELD NAMES matching your DB ===
FIELD_START = "timestamp"
FIELD_END   = "created_at"

def _to_utc(dt: datetime) -> datetime:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _parse_ymd(d: str) -> datetime:
    return datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)

def _keep_time(old_dt: datetime, new_day_start: datetime) -> datetime:
    old_dt = _to_utc(old_dt); base = _to_utc(new_day_start)
    return base.replace(
        hour=old_dt.hour, minute=old_dt.minute,
        second=old_dt.second, microsecond=old_dt.microsecond
    )

def _parse_date_field(v: Any) -> Optional[datetime]:
    """Accept BSON Date or ISO8601 string; return UTC-aware datetime or None."""
    if isinstance(v, datetime):
        return _to_utc(v)
    if isinstance(v, str):
        s = v.strip()
        try:
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            s = s.replace(" ", "T")
            return _to_utc(datetime.fromisoformat(s))
        except Exception:
            return None
    return None

def _shift_doc_db_fields(doc: Dict[str, Any], src_start: datetime, tgt_start: datetime) -> Dict[str, Any]:
    """Shift one document using FIELD_START/FIELD_END."""
    d = dict(doc)
    d.pop("_id", None)

    old_start = _parse_date_field(d.get(FIELD_START))
    old_end   = _parse_date_field(d.get(FIELD_END))
    if old_start is None:
        return d

    # day offset from source window start
    offset_days = (old_start.date() - src_start.date()).days
    new_day_base = tgt_start + timedelta(days=offset_days)

    new_start = _keep_time(old_start, new_day_base)
    d[FIELD_START] = new_start

    if old_end is not None:
        d[FIELD_END] = new_start + (old_end - old_start)

    # refresh created_at to now? -> NO. We already moved FIELD_END.
    # If you still want an audit stamp, add a separate field.
    return d

@router.post("/chain_remap")
def chain_remap(req: ChainRemapRequest):
    base_query: Dict[str, Any] = {"metric": req.metric}
    if req.username:
        base_query["username"] = req.username

    # Pull docs that at least have FIELD_START
    cursor = col.find({**base_query, FIELD_START: {"$exists": True}}, projection={FIELD_START: 1})
    docs_with_start = list(cursor)
    if not docs_with_start:
        raise HTTPException(status_code=404, detail=f"No data found for metric='{req.metric}' (username={req.username or '*'}) with {FIELD_START} present.")

    # Parse starts to compute current source window
    parsed_pairs: List[tuple] = []  # (id, parsed_start)
    invalid_meta = 0
    for d in docs_with_start:
        ps = _parse_date_field(d.get(FIELD_START))
        if ps is None:
            invalid_meta += 1
            continue
        parsed_pairs.append((d["_id"], ps))

    if not parsed_pairs:
        raise HTTPException(status_code=400, detail=f"All documents with {FIELD_START} failed to parse as dates.")

    parsed_pairs.sort(key=lambda p: p[1])
    source_start = parsed_pairs[0][1]
    source_end   = parsed_pairs[-1][1]
    source_days  = (source_end.date() - source_start.date()).days + 1

    # Get full set for that window (can’t range on strings safely, so fetch by query and filter in app)
    src_full = list(col.find(base_query))
    valid_src: List[Dict[str, Any]] = []
    skipped_in_window = 0
    for d in src_full:
        sd = _parse_date_field(d.get(FIELD_START))
        if sd is None:
            continue
        if source_start.date() <= sd.date() <= source_end.date():
            valid_src.append(d)
        else:
            skipped_in_window += 1

    if not valid_src:
        raise HTTPException(status_code=400, detail="No valid documents in computed source window.")

    # Parse target start; compute target end to match length
    target_start = _parse_ymd(req.target_start_date)
    target_end = target_start + timedelta(days=source_days - 1)

    # Remap
    remapped = [_shift_doc_db_fields(d, src_start=source_start, tgt_start=target_start) for d in valid_src]

    # Insert first
    inserted = len(col.insert_many(remapped).inserted_ids)

    # Delete originals by _id (robust even if stored as strings)
    original_ids = [d["_id"] for d in valid_src]
    deleted = col.delete_many({"_id": {"$in": original_ids}}).deleted_count

    # Samples
    def _iso(v: Any) -> Optional[str]:
        dt = _parse_date_field(v)
        return dt.isoformat() if dt else None

    sample_before = {
        FIELD_START: _iso(valid_src[0].get(FIELD_START)),
        FIELD_END: _iso(valid_src[0].get(FIELD_END)),
        "value": valid_src[0].get("value"),
    }
    sample_after = {
        FIELD_START: _iso(remapped[0].get(FIELD_START)),
        FIELD_END: _iso(remapped[0].get(FIELD_END)),
        "value": remapped[0].get("value"),
    }

    return {
        "metric": req.metric,
        "username": req.username or "*",
        "source_window": f"{source_start.date()} → {source_end.date()}",
        "target_window": f"{target_start.date()} → {target_end.date()}",
        "inserted": inserted,
        "deleted": deleted,
        "skipped_in_meta_scan": invalid_meta,
        "skipped_outside_window": skipped_in_window,
        "sample_before": sample_before,
        "sample_after": sample_after
    }
