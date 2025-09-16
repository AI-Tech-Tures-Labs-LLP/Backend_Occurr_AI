from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import List
# from app.api.v1.auth import decode_token
from app.api.auth.auth import decode_token
from app.db.database import conversations_collection
from app.core.chatbot_engine import main

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class ChatMessage(BaseModel):
    message: str
    mode: str = "friendlymode"

class ChatRequest(BaseModel):
    messages: List[dict]
    mode: str = "friendlymode"

def save_message(username, role, content):
    conversations_collection.update_one(
        {"username": username},
        {"$push": {"history": {"role": role, "content": content}}},
        upsert=True
    )

def get_recent_history(username, limit=6):
    doc = conversations_collection.find_one({"username": username})
    if doc:
        return doc.get("history", [])[-limit:]
    return []





@router.post("/chat")
def chat(msg: ChatMessage, token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)

    # Wrap the string into a list of valid message dicts
    messages = [{"role": "user", "content": msg.message}]
    bot_response = main(messages, msg.mode)

    save_message(username, "user", msg.message)
    save_message(username, "assistant", bot_response)

    return {"response": bot_response, "history": get_recent_history(username)}


@router.post("/smart-chat/query")
def smart_chat(req: ChatRequest, token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)

    # If messages is empty or malformed, inject a default message
    if not req.messages or not isinstance(req.messages[0], dict) or "role" not in req.messages[0]:
        req.messages = [{"role": "user", "content": "Hi"}]

    bot_response = main(req.messages, req.mode)

    # Save last user message
    last_user_msg = next((m for m in reversed(req.messages) if m["role"] == "user"), None)
    if last_user_msg:
        save_message(username, "user", last_user_msg["content"])
    save_message(username, "assistant", bot_response)

    return {"response": bot_response, "history": get_recent_history(username)}


