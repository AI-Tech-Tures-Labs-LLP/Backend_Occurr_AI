


from pymongo import MongoClient
from datetime import datetime, timedelta
from bson import ObjectId
from app.db.database import MONGO_URL

client = MongoClient(MONGO_URL)
db = client["techjewel"]
journals_collection = db["journal_entries"]

# def save_journal_entry(username, entry_type, prompt, response="", tags=None, mood=None, extra_fields=None):
#     entry = {
#         "username": username,
#         "timestamp": datetime.utcnow(),
#         "type": entry_type,  # "triggered", "scheduled", or "conversation"
#         "prompt": prompt,
#         "response": response,
#         "tags": tags or [],
#         "mood": mood,
#     }
#     if extra_fields:
#         entry.update(extra_fields)
#     journals_collection.insert_one(entry)

def save_journal_entry(username, entry_type, prompt, response="", tags=None, mood=None, extra_fields=None, task_id=None):
    entry = {
        "username": username,
        "timestamp": datetime.utcnow(),
        "type": entry_type,  # "triggered", "scheduled", or "conversation"
        "prompt": prompt,
        "response": response,
        "tags": tags or [],
        "mood": mood,
    }

    if task_id:
        entry["task_id"] = task_id  # Link the task to the journal

    if extra_fields:
        entry.update(extra_fields)
    
    # Insert the journal entry into the database
    result = journals_collection.insert_one(entry)

    return str(result.inserted_id)


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


def get_or_create_daily_journal(username: str):
    # Retrieve journal for today
    now = datetime.utcnow()
    today_start = datetime(now.year, now.month, now.day)
    today_end = today_start + timedelta(days=1)

    existing_journal = journals_collection.find_one({
        "username": username,
        "timestamp": {"$gte": today_start, "$lt": today_end}
    })

    if existing_journal:
        return existing_journal

    # No journal for today, create a new one
    new_journal = {
        "username": username,
        "content": "",  # Empty content initially
        "tags": [],
        "source": "auto",
        "timestamp": today_start,
        "chat_id": None,
        "context": {}
    }

    result = journals_collection.insert_one(new_journal)
    return journals_collection.find_one({"_id": result.inserted_id})