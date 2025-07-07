from pymongo import MongoClient
from datetime import datetime
from app.db.database import MONGO_URL,users_collection

client = MongoClient(MONGO_URL)
db = client["techjewel"]

breathing_collection = db["breathing_exercises"]

# Create indexes for optimization
breathing_collection.create_index([("username", 1), ("created_at", -1)])

# Breathing exercise document structure
def build_breathing_doc(username: str, duration: int, created_at: datetime) -> dict:
    return {
        "username": username,
        "date": created_at.date(),
        "duration": duration,
        "created_at": created_at
    }

def save_breathing_exercise(username: str, duration: int):
    now = datetime.utcnow()
    doc = build_breathing_doc(username, duration, now)

    # Insert the breathing exercise into the database
    existing_doc = breathing_collection.find_one({
        "username": username,
        "date": now.date()
    })

    if existing_doc:
        # Update existing document if it exists for the same date
        breathing_collection.update_one(
            {"_id": existing_doc["_id"]},
            {"$set": {"duration": duration, "created_at": now}}
        )
        return str(existing_doc["_id"])
    
    else:
        # Insert a new document if it doesn't exist
        result = breathing_collection.insert_one(doc)
        
        # Update user's total breathing time
        users_collection.update_one(
            {"username": username},
            {"$inc": {"total_breathing_time": duration}}
        )

    return str(result.inserted_id)  # Return the ID of the inserted document
