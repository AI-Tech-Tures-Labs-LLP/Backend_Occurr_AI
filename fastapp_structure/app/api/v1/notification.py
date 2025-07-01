from fastapi import APIRouter, Depends, HTTPException, Body, Query
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import random
from openai import OpenAI

# from app.api.v1.auth import decode_token
from app.api.auth.auth import decode_token
from app.db.database import users_collection
from app.db.health_data_model import alert_collection
from bson import ObjectId
import os
from app.core.chatbot_engine import client
from dotenv import load_dotenv
load_dotenv()


router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

client = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"),base_url=os.getenv("OPENAI_API_BASE_URL"))


class AlertResponseRequest(BaseModel):
    response: str

# ✅ Conversation state store
@router.get("/get_notification")
def get_notifications(token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)

    user = users_collection.find_one({"username": username}, {"notifications": 1})
    return user.get("notifications", [])


@router.post("/mark_notification_read")
def mark_read(index: int = Body(...), token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)

    users_collection.update_one(
        {"username": username},
        {"$set": {f"notifications.{index}.read": True}}
    )
    return {"message": "Marked as read"}



@router.get("/notifications/unread")
def get_unread_notifications(token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)

    user = users_collection.find_one({"username": username}, {"notifications": 1})
    all_notifications = user.get("notifications", [])
    unread = [n for n in all_notifications if not n.get("read", False)]

    return {
        "count": len(unread),
        "notifications": unread
    }

@router.get("/alerts/")
def get_alerts(token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)

    alerts = alert_collection.find({"username": username})
    alerts_list = list(alerts)

    # Convert ObjectId to str
    for alert in alerts_list:
        alert["_id"] = str(alert["_id"])

    return alerts_list

# @router.get("/alerts/{alert_id}")
# def get_alert_by_id(alert_id: str, token: str = Depends(oauth2_scheme)):
#     valid, username = decode_token(token)
#     if not valid:
#         raise HTTPException(status_code=401, detail=username)

#     alert = alert_collection.find_one({"_id": ObjectId(alert_id), "username": username})
#     if not alert:
#         raise HTTPException(status_code=404, detail="Alert not found")

#     return alert



system_prompt = {
    "role": "system",
    "content": (
        "You are a friendly and supportive health assistant. "
        "Always respond with one clear, friendly, short question that helps the user reflect on their health. "
        "Use 'why' or 'what' style questions depending on the metric discussed. "
        "Keep tone warm, helpful, and non-judgmental. Never give generic advice or long responses. "
        "Do NOT explain. Just ask a question like:\n"
        "- Why do you think your oxygen levels dropped recently?\n"
        "- What affected your activity level today?\n"
        "- Did anything affect your sleep quality or duration?"
    )
}

@router.post("/alerts/{alert_id}/response")
def save_alert_response(
    alert_id: str,
    req: AlertResponseRequest,
    token: str = Depends(oauth2_scheme)
):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail="Unauthorized")

    alert = alert_collection.find_one({"_id": ObjectId(alert_id), "username": username})
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    # Determine question based on alert metric
    metric = alert.get("metric", "general")
    metric_prompt = {
        "heartRate": "What might have caused your heart rate to go high?",
        "spo2": "Why do you think your oxygen levels dropped recently?",
        "steps": "What affected your activity level today?",
        "sleep": "Did anything affect your sleep quality or duration?",
    }.get(metric, "How are you feeling about this recent health update?")

    chat = alert.get("chat", [])
    if not chat:
    # First time opening the chat — send metric-based prompt
        assistant_entry = {"role": "assistant", "content": metric_prompt}
        chat = [assistant_entry]

        alert_collection.update_one({"_id": ObjectId(alert_id)}, {"$set": {"chat": chat}})
        return {
            "message": "First prompt initiated.",
            "assistant_reply": metric_prompt
        }

    user_turns = len([m for m in chat if m["role"] == "user"])
    if user_turns >= 4:
        return {"message": "✅ Thank you for sharing the reasons , Hope this helps you reflect on your health.", "assistant_reply": None}

    user_msg = {"role": "user", "content": req.response}
    print("🔍 User Prompt:", req.response)

    alert_collection.update_one(
        {"_id": ObjectId(alert_id)},
        {
            "$push": {"chat": user_msg},
            "$set": {
                "responded": True,
                "responded_at": datetime.utcnow()
            }
        }
    )

    chat.append(user_msg)
    try:
        gpt_response = client.chat.completions.create(
            model=os.getenv("OPENAI_API_MODEL", "gpt-4"),
            messages=[system_prompt] + chat,
            max_tokens=50,
            temperature=0.4
        )
        assistant_msg = gpt_response.choices[0].message.content.strip()
        print("🤖 Assistant Reply:", assistant_msg)
    except Exception as e:
        print("⚠️ GPT continuation failed:", e)
        assistant_msg = "Thanks for your response. We'll keep monitoring your health."

    assistant_entry = {"role": "assistant", "content": assistant_msg}
    alert_collection.update_one(
        {"_id": ObjectId(alert_id)},
        {"$push": {"chat": assistant_entry}}
    )

    return {
        "message": "Response saved.",
        "assistant_reply": assistant_msg
    }
