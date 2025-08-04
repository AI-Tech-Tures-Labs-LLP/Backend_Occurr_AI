from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from app.api.auth.auth import decode_token
from app.db.database import users_collection
from app.db.task_model import task_collection
from bson import ObjectId
from datetime import datetime, timedelta
from app.utils.task_helper import generate_daily_tasks_from_profile, complete_task, check_and_notify_pending_tasks_for_all_users
from app.core.chatbot_engine import client
from dotenv import load_dotenv
load_dotenv()
router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class TaskRequest(BaseModel):
    username: str
    task_type: str  # e.g., "exercise", "meditation", "breathing"
    trigger_time: Optional[str] = None  # ISO format datetime string, e.g., "2023-10-01T12:00:00Z"
    duration: Optional[int] = None  # in seconds
    date: Optional[str] = None  # ISO format date string, e.g., "2023-10-01"

class TaskResponse(BaseModel):
    task_id: str
    username: str
    task_type: str
    trigger_time: str  # ISO format datetime string, e.g., "2023-10-01T12:00:00Z"
    duration: Optional[int] = None  # in seconds
    date: Optional[str] = None  # ISO format date string, e.g., "2023-10-01"
    created_at: str  # ISO format datetime string, e.g., "2023-10-01T12:00:00Z"


@router.get("/task/get_tasks")
def get_today_tasks(token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)

    user = users_collection.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Calculate start and end of today (UTC)
    now = datetime.utcnow()
    start_of_day = datetime(now.year, now.month, now.day)
    end_of_day = start_of_day + timedelta(days=1)

    # Fetch all tasks scheduled for today without limiting the fields
    tasks = list(task_collection.find(
        {
            "username": username,
            "trigger_time": {"$gte": start_of_day, "$lt": end_of_day}
        }
    ).sort("trigger_time", 1))  # Sort by trigger_time

    # Convert ObjectId to string for serialization (if needed)
    for task in tasks:
        task["_id"] = str(task["_id"])

    return {"tasks": tasks}




@router.get("/task/get_all_tasks")
def get_all_tasks(token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)

    user = users_collection.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    tasks = list(task_collection.find(
        {
            "username": username,
            # "trigger_time": {"$gte": start_of_day, "$lt": end_of_day}
        }
    ).sort("trigger_time", 1))  # Sort by trigger_time

    # Convert ObjectId to string for serialization (if needed)
    for task in tasks:
        task["_id"] = str(task["_id"])

    return {"task": tasks}



@router.get("/task/get_task_by_id")
def get_task_by_id(task_id: str, token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)

    user = users_collection.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")       
    tasks = list(task_collection.find(
        {"_id": ObjectId(task_id), "username": username}
    ).sort("trigger_time", 1))  # Sort by trigger_time

    # Convert ObjectId to string for serialization (if needed)
    for task in tasks:
        task["_id"] = str(task["_id"])
        
    
    # task = task_collection.find_one({"_id": ObjectId(task_id), "username": username}, {"_id": 0, "date": 1, "task_type": 1, "trigger_time": 1, "duration": 1, "completed": 1})
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task": task}



from datetime import datetime, timedelta

@router.get("/task/get_tasks_by_date")
def get_tasks_by_date(date: str, token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)

    user = users_collection.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Convert the date string to datetime object
    try:
        date_obj = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    
    # Calculate start and end of day
    start_of_day = date_obj.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = date_obj.replace(hour=23, minute=59, second=59, microsecond=999999)

    # Query tasks based on trigger_time within the day range
   
    tasks = list(task_collection.find(
        {
            "trigger_time": {"$gte": start_of_day, "$lt": end_of_day},
            "username": username
            # "trigger_time": {"$gte": start_of_day, "$lt": end_of_day}
        }
    ).sort("trigger_time", 1))  # Sort by trigger_time

    # Convert ObjectId to string for serialization (if needed)
    for task in tasks:
        task["_id"] = str(task["_id"])

    return {"tasks": tasks}



@router.get("/task/get_tasks_by_type")
def get_tasks_by_type(task_type: str, token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)

    user = users_collection.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    tasks = list(task_collection.find(
        {
            "username": username,
            "type": task_type
            # "trigger_time": {"$gte": start_of_day, "$lt": end_of_day}
        }
    ).sort("trigger_time", 1))  # Sort by trigger_time

    # Convert ObjectId to string for serialization (if needed)
    for task in tasks:
        task["_id"] = str(task["_id"])
    # tasks = list(task_collection.find({"username": username, "task_type": task_type}, {"_id": 0, "date": 1, "task_type": 1, "trigger_time": 1, "duration": 1, "completed": 1}).sort("trigger_time", 1))
    return {"tasks": tasks}


@router.get("/task/streak")
def get_task_streak(token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)

    user = users_collection.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Calculate streak
    # tasks = list(task_collection.find({"username": username, "completed": True}, {"_id": 0, "date": 1}).sort("date", 1))
    tasks = list(task_collection.find(
        {
            "username": username,
           "completed": True
            # "trigger_time": {"$gte": start_of_day, "$lt": end_of_day}
        }
    ).sort("trigger_time", 1))  # Sort by trigger_time

    # Convert ObjectId to string for serialization (if needed)
    for task in tasks:
        task["_id"] = str(task["_id"])

    if not tasks:
        return {"streak": 0}

    streak = 0
    for i in range(1, len(tasks)):
        if tasks[i]["date"] == (datetime.strptime(tasks[i - 1]["date"], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"):
            streak += 1
        else:
            break

    return {"streak": streak}


@router.put("/task/complete_task")
def complete_task_with_content(task_id: str, task_content: str, image_url: Optional[str] = None, token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)

    user = users_collection.find_one({"username": username})
    if not user:      
        raise HTTPException(status_code=404, detail="User not found")
    
    completed_task = complete_task(username, task_id, task_content, image_url)
    if not completed_task:
        raise HTTPException(status_code=404, detail="Task not found or already completed")
    return {'result': completed_task,'status': 'success'}


