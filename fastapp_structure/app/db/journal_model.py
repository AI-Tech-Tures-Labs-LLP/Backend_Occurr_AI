


from pymongo import MongoClient
from datetime import datetime, timedelta
from bson import ObjectId
from app.db.database import MONGO_URL

client = MongoClient(MONGO_URL)
db = client["techjewel"]
journals_collection = db["journal_entries"]

def save_journal_entry(username, entry_type, prompt, response="", tags=None, mood=None, extra_fields=None):
    entry = {
        "username": username,
        "timestamp": datetime.utcnow(),
        "type": entry_type,  # "triggered", "scheduled", or "conversation"
        "prompt": prompt,
        "response": response,
        "tags": tags or [],
        "mood": mood,
    }
    if extra_fields:
        entry.update(extra_fields)
    journals_collection.insert_one(entry)

def patch_journal(journal_id: str, update_data: dict):
    update_data["updated_at"] = datetime.utcnow()
    journals_collection.update_one(
        {"_id": ObjectId(journal_id)},
        {"$set": update_data}
    )

def get_journals_by_user_month(username: str):
    now = datetime.utcnow()
    start = datetime(now.year, now.month, 1)
    end = datetime(now.year + 1, 1, 1) if now.month == 12 else datetime(now.year, now.month + 1, 1)
    return list(journals_collection.find({
        "username": username,
        "timestamp": {"$gte": start, "$lt": end}
    }).sort("timestamp", -1))

def get_journals_by_day(username: str, date_str: str):
    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return []

    start = datetime(target_date.year, target_date.month, target_date.day)
    end = start + timedelta(days=1)

    return list(journals_collection.find({
        "username": username,
        "timestamp": {"$gte": start, "$lt": end}
    }).sort("timestamp", -1))
