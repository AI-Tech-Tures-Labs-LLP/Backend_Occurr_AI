from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime
from bson import ObjectId
from pymongo import MongoClient
import os
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from pytz import UTC
from fastapi import Query
from fastapi.security import OAuth2PasswordBearer


from app.api.v2.auth import decode_token


router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# --- Mongo setup ---
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
client = MongoClient(MONGO_URL)
db = client["techjewel"]
user_dates_collection = db["user_dates"]

class UserDateEntry(BaseModel):
    date: Optional[datetime] = None  # if not provided, default to now


@router.post("/user_dates/save")
async def save_user_date(
    entry: UserDateEntry,
    token: str = Depends(oauth2_scheme)
):
    """
    Save a user-provided date (or current UTC if missing)
    into the user_dates collection, username taken from token.
    """
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)

    date_to_save = entry.date or datetime.utcnow()
    if user_dates_collection.find_one({"username": username}):
        raise HTTPException(status_code=400, detail="Date entry already exists for this user. Use update instead.")

    doc = {
        "username": username,
        "date": date_to_save,
        "created_at": datetime.utcnow()
    }

    try:
        result = user_dates_collection.insert_one(doc)
        return {
            "message": "User date saved successfully",
            "id": str(result.inserted_id),
            "username": username,
            "date": date_to_save.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- PUT: Update the date ---
@router.put("/user_dates/update")
async def update_user_date(
    entry: UserDateEntry,
    token: str = Depends(oauth2_scheme)
):
    """
    Update the saved date for the current user in user_dates collection.
    Username is taken from token.
    """
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)

    date_to_save = entry.date or datetime.utcnow()

    try:
        result = user_dates_collection.update_one(
            {"username": username},
            {"$set": {"date": date_to_save, "updated_at": datetime.utcnow()}},
            upsert=True,  # optional: create if not exists
        )

        return {
            "message": "User date updated successfully",
            "username": username,
            "date": date_to_save.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user_dates/latest")
async def get_latest_user_date(token: str = Depends(oauth2_scheme)):
    """
    Fetch the latest saved date for a user.
    """
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)

    doc = user_dates_collection.find_one(
        {"username": username},
        sort=[("date", -1), ("created_at", -1)]
    )
    if not doc:
        return {"username": username, "date": None, "message": "No saved date"}
    dt = doc.get("date")
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return {"username": username, "date": dt.isoformat()}
    if isinstance(dt, str):
        return {"username": username, "date": dt}
    return {"username": username, "date": None}