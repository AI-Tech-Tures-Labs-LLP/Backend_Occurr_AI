
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
# from app.api.v1.auth import decode_token
from app.api.auth.auth import decode_token
from app.db.database import set_journal_time, get_journal_time, get_all_users_and_times


router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class TimeRequest(BaseModel):
    time: str  # Format: "HH:MM"

@router.post("/settings/journal-time")
def update_journal_time(req: TimeRequest, token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)
    set_journal_time(username, req.time)
    return {"message": f"Journal time set to {req.time}"}

@router.get("/settings/journal-time")
def fetch_journal_time(token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)
    return {"journal_time": get_journal_time(username)}

