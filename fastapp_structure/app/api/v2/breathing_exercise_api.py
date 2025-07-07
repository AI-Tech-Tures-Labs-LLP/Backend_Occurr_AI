from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import OAuth2PasswordBearer

from pydantic import BaseModel
from typing import Optional, Dict, Any  
from app.api.auth.auth import decode_token
from app.db.database import users_collection
from app.db.breathing_exercise import breathing_collection
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/auth/token")


router = APIRouter()

class BreathingExerciseRequest(BaseModel):
    username: str
    exercise_type: str
    duration: int  # in seconds
    date: Optional[str] = None  # ISO format date string, e.g., "2023-10-01"

class BreathingExerciseResponse(BaseModel):
    username: str
    exercise_type: str
    duration: int  # in seconds
    date: str  # ISO format date string, e.g., "2023-10-01"
    created_at: str  # ISO format datetime string, e.g., "2023-10-01T12:00:00Z"


@router.post("/breathing_exercise/save", response_model=BreathingExerciseResponse)
def save_breathing_exercise(
    request: BreathingExerciseRequest
):
    # Save the breathing exercise data to the database
    exercise_data = request.dict()
    exercise_data["created_at"] = "2023-10-01T12:00:00Z"  # Mock created_at for response
    breathing_collection.insert_one(exercise_data)
    return BreathingExerciseResponse(**exercise_data)


@router.get("/breathing_exercise/get")
def get_breathing_exercises(
    username: str = Query(..., description="Username of the user"),
    date: Optional[str] = Query(None, description="Date in ISO format (YYYY-MM-DD) to filter exercises")
):
    # Fetch breathing exercises for the user, optionally filtered by date
    query = {"username": username}
    if date:
        query["date"] = date
    exercises = list(breathing_collection.find(query))
    if not exercises:
        raise HTTPException(status_code=404, detail="No breathing exercises found for this user")
    return {"exercises": exercises}



@router.get("/breathing_exercise/get_all")
def get_all_breathing_exercises():
    # Fetch all breathing exercises from the database
    exercises = list(breathing_collection.find())
    if not exercises:
        raise HTTPException(status_code=404, detail="No breathing exercises found")
    return {"exercises": exercises}


@router.get("/breathing_exercise/streak")
def get_breathing_exercise_streak(
    username: str = Query(..., description="Username of the user"),
    date: Optional[str] = Query(None, description="Date in ISO format (YYYY-MM-DD) to filter exercises")
):
    # Fetch breathing exercise streak for the user, optionally filtered by date
    query = {"username": username}
    if date:
        query["date"] = date
    exercises = list(breathing_collection.find(query))
    if not exercises:
        raise HTTPException(status_code=404, detail="No breathing exercises found for this user")
    # Calculate streak (mock implementation)
    streak = len(exercises)
    return {"username": username, "streak": streak}