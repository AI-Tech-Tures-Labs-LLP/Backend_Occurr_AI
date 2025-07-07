from pymongo import MongoClient
from datetime import datetime, timedelta
from bson import ObjectId

from app.db.database import MONGO_URL, users_collection



client = MongoClient(MONGO_URL)
db = client["techjewel"]

notifications_collection = db["notifications"]



def save_notification(username: str, title: str, body: str, read: bool, task_id: str = None, alert_id: str = None):
    # Create the notification document
    notification = {
        "username": username,
        "title": title,
        "body": body,
        "read": False,  # Default to unread
        "timestamp": datetime.utcnow()  # Timestamp when the notification is sent
    }
    
    # Convert ObjectId to string if task_id or alert_id are provided
    if task_id:
        notification["task_id"] = str(task_id) if isinstance(task_id, ObjectId) else task_id
    if alert_id:
        notification["alert_id"] = str(alert_id) if isinstance(alert_id, ObjectId) else alert_id

    # Clean up the data to remove None values (don't store empty fields in MongoDB)
    notification = {key: value for key, value in notification.items() if value is not None}

    # Insert the notification into the database
    try:
        result = notifications_collection.insert_one(notification)
        return str(result.inserted_id)  # Return the ID of the inserted notification document
    except Exception as e:
        print(f"Error inserting notification: {e}")
        return None