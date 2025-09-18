# app/db/task_model.py

from pymongo import ASCENDING, DESCENDING, MongoClient
from bson import ObjectId
from datetime import datetime,timedelta
import os
from app.db.database import MONGO_URL



client = MongoClient(MONGO_URL)
db = client["techjewel"]

task_collection = db["tasks"]

# Create indexes for optimization
task_collection.create_index([("username", ASCENDING), ("trigger_time", DESCENDING)])
task_collection.create_index([("completed", ASCENDING)])
task_collection.create_index([("type", ASCENDING)])

# Task document structure
def build_task_doc(
    username: str,
    type: str,
    title: str,
    trigger_time: datetime,
    expires_at: datetime,
    metadata: dict = None
) -> dict:
    return {
        "username": username,
        "type": type,  # 'meal', 'meditation', 'reflection', etc.
        "title": title,
        "trigger_time": trigger_time,
        "expires_at": expires_at,  
        "completed": False,
        "notified": False,
        "completed_at": None,
        "journal_entry_id": None,
        "metadata": metadata or {},
        "created_at": datetime.utcnow()
    }
