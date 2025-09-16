from fastapi import APIRouter, Depends, HTTPException, Query,Body
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
from typing import Optional
from bson import ObjectId
from datetime import datetime, time
# from utils.saveimage import save_image_locally

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import Optional
import os, uuid, mimetypes, boto3
from app.api.v2.health_data import get_user_anchor_date
from fastapi.security import OAuth2PasswordBearer   

# ... existing imports ...
router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- S3 config ---
S3_BUCKET = os.getenv("AWS_S3_BUCKET")
S3_REGION = os.getenv("AWS_REGION", "ap-south-1")
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=S3_REGION,
)



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


# @router.get("/task/get_tasks")
# def get_today_tasks(token: str = Depends(oauth2_scheme)):
#     valid, username = decode_token(token)
#     if not valid:
#         raise HTTPException(status_code=401, detail=username)

#     user = users_collection.find_one({"username": username})
#     if not user:
#         raise HTTPException(status_code=404, detail="User not found")

#     # Calculate start and end of today (UTC)
    
#     start_of_day = datetime(now.year, now.month, now.day)
#     end_of_day = start_of_day + timedelta(days=1)

#     # Fetch all tasks scheduled for today without limiting the fields
#     tasks = list(task_collection.find(
#         {
#             "username": username,
#             "trigger_time": {"$gte": start_of_day, "$lt": end_of_day}
#         }
#     ).sort("trigger_time", 1))  # Sort by trigger_time

#     # Convert ObjectId to string for serialization (if needed)
#     for task in tasks:
#         task["_id"] = str(task["_id"])

#     return {"tasks": tasks}


@router.get("/task/get_tasks")
def get_today_tasks(token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)

    user = users_collection.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # --- Get base date from user_dates collection ---
    base = get_user_anchor_date(username)  # <-- helper we wrote earlier
    # Normalize to start/end of that day (UTC)
    start_of_day = base.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = start_of_day + timedelta(days=1)

    # Fetch all tasks scheduled for that "day"
    tasks = list(task_collection.find(
        {
            "username": username,
            "trigger_time": {"$gte": start_of_day, "$lt": end_of_day}
        }
    ).sort("trigger_time", 1))

    # Convert ObjectId to str for JSON serialization
    for task in tasks:
        if "_id" in task:
            task["_id"] = str(task["_id"])

    return {
        "anchor_date": base.isoformat(),  # helpful for client
        "tasks": tasks
    }



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

    # Fetch only completed tasks, sorted
    tasks = list(task_collection.find(
        {"username": username, "completed": True}
    ).sort("created_at", 1))

    for task in tasks:
        task["_id"] = str(task["_id"])

    if not tasks:
        return {"streak": 0}

    streak = 0 # at least one day if tasks exist
    for i in range(1, len(tasks)):
        prev_day = tasks[i - 1]["created_at"].date()
        this_day = tasks[i]["created_at"].date()

        if this_day == prev_day + timedelta(days=1):
            streak += 1
        else:
            streak = 1  # reset streak if gap is found

    return {"streak": streak}
















# S3 upload version
# task_api.py


def upload_image_to_s3(file: UploadFile, key_prefix: str = "journal/meals/") -> tuple[str, str]:
    """
    Uploads an image to S3 without ACLs and returns:
      (s3_key, presigned_url)
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed")

    ext = os.path.splitext(file.filename or "")[1] or mimetypes.guess_extension(file.content_type) or ".jpg"
    key = f"{key_prefix}{uuid.uuid4().hex}{ext}"

    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    # ✅ no ACL here
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=data,
        ContentType=file.content_type,
    )

    # generate presigned GET url (default 1h, you can tune ExpiresIn)
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=3600
    )
    return key, url


# @router.put("/task/complete_task")
# def complete_task_with_chat(
#     # NOTE: Use Form/File so Swagger allows multipart
#     task_id: str = Form(...),
#     task_content: str | None = Form(None),
#     image_file: UploadFile | None = File(None),
#     # image_url: str | None = Form(None),
#     token: str = Depends(oauth2_scheme)
# ):
#     # 1) Auth
#     try:
#         valid, username = decode_token(token)
#         if not valid:
#             raise HTTPException(status_code=401, detail=username)
#     except Exception:
#         raise HTTPException(status_code=401, detail="Invalid token")

#     # 2) User check
#     user = users_collection.find_one({"username": username})
#     if not user:
#         raise HTTPException(status_code=404, detail="User not found")

#     # 3) If file provided, upload to S3 and prefer that URL
#     # if image_file:
#     #     try:
#     #         s3_key, presigned_url = upload_image_to_s3(image_file)
#     #         image_url = presigned_url   # use this in vision + journal
#     #     except Exception as e:
#     #         raise HTTPException(status_code=400, detail=f"S3 upload failed: {e}")

#     from app.utils.task_helper import complete_task
#     result = complete_task(
#         username=username,
#         task_id=task_id,
#         task_content=task_content,
#         image_file=image_file,   # pass the file object
#     )
#     # 4) Continue to your helper (it saves to journal + does vision)
#     if not result:
#         raise HTTPException(status_code=404, detail="Task not found or already completed")
#     return {"status": "success", "result": result}
    


@router.put("/task/complete_task")
def complete_task_with_chat(
    task_id: str = Form(...),
    task_content: str | None = Form(None),
    image_file: UploadFile | None = File(None),
    token: str = Depends(oauth2_scheme)
):
    # 1) Auth
    try:
        valid, username = decode_token(token)
        if not valid:
            raise HTTPException(status_code=401, detail=username)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

    # 2) User check
    user = users_collection.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # 3) Do the work
    try:
        result = complete_task(
            username=username,
            task_id=task_id,
            task_content=task_content,
            image_file=image_file,   # pass the UploadFile
        )
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        # Log and return clean error
        print("complete_task error:", e)
        raise HTTPException(status_code=500, detail="Failed to complete task")

    if not result:
        raise HTTPException(status_code=404, detail="Task not found or already completed")
    return {"status": "success", "result": result}