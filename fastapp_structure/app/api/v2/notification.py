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
from app.db.notification_model import notifications_collection
from bson import ObjectId
import os
from app.core.chatbot_engine import client
from dotenv import load_dotenv
load_dotenv()

import firebase_admin
from firebase_admin import credentials, messaging
from functools import lru_cache


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
 
    notifications = list(notifications_collection.find({"username": username}))
 
    # Convert ObjectId to string if needed
    for n in notifications:
        n["_id"] = str(n["_id"])
        n["task_id"] = str(n["task_id"]) if "task_id" in n else None
 
    return {"notifications": notifications}


@router.post("/mark_notification_read")
def mark_read(notification_id: str = Body(...), token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)

    result = notifications_collection.update_one(
        {"_id": ObjectId(notification_id), "username": username},
        {"$set": {"read": True}}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Notification not found")

    return {"message": "Notification marked as read"}


@router.get("/notifications/unread")
def get_unread_notifications(token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)

    unread = list(notifications_collection.find(
        {"username": username, "read": False}
    ))

    # Convert ObjectId to string for JSON compatibility
    for n in unread:
        n["_id"] = str(n["_id"])

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


# ✅ Initialize Firebase app once (singleton)
@lru_cache()
def init_firebase():
    try:
        if not firebase_admin._apps:  # Check if already initialized
            cred = credentials.Certificate("app/secrets/firebase-service-account.json")  # path to your JSON file
            firebase_admin.initialize_app(cred)
            print("✅ Firebase initialized successfully")
        return True
    except Exception as e:
        print(f"⚠️ Firebase initialization error: {e}")
        return False

# ✅ Send push via FCM v1 API
def send_push_notification_v1(token: str, title: str, body: str, screen: str = "HomeScreen", data: dict = None):
    if not init_firebase():
        print("⚠️ Firebase not initialized, cannot send notification")
        return {"error": "Firebase not initialized"}

    # Prepare data payload
    payload_data = {
        "click_action": "FLUTTER_NOTIFICATION_CLICK",
        "screen": screen
    }
    
    # Add any additional data
    if data:
        payload_data.update(data)

    # Construct message
    message = messaging.Message(
        token=token,
        notification=messaging.Notification(
            title=title,
            body=body
        ),
        data=payload_data,
        android=messaging.AndroidConfig(
            priority="high",
            notification=messaging.AndroidNotification(
                sound="default",
                click_action="FLUTTER_NOTIFICATION_CLICK"
            )
        ),
        apns=messaging.APNSConfig(
            headers={"apns-priority": "10"},
            payload=messaging.APNSPayload(
                aps=messaging.Aps(
                    sound="default",
                    category=screen  # optional for iOS
                )
            )
        )
    )

    try:
        # Send notification
        response = messaging.send(message)
        print(f"✅ Push notification sent: {response}")
        return {"message_id": response}
    except Exception as e:
        print(f"⚠️ Error sending push notification: {e}")
        return {"error": str(e)}


# ✅ FCM Token Management Endpoints
@router.post("/update_fcm_token")
def update_fcm_token(fcm_data: dict = Body(...), token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)
    
    fcm_token = fcm_data.get("fcm_token")
    if not fcm_token:
        raise HTTPException(status_code=400, detail="FCM token is required")
    
    # Update the user's FCM token in the database
    result = users_collection.update_one(
        {"username": username},
        {"$set": {"fcm_token": fcm_token}}
    )
    
    print(f"🔔 FCM token updated for user {username}: {fcm_token[:10]}...")
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Send a test notification to verify the token works
    try:
        send_push_notification_v1(
            token=fcm_token,
            title="Notifications Enabled",
            body="You will now receive health alerts even when the app is closed.",
            screen="HomeScreen"
        )
        print(f"✅ Test notification sent to {username}")
    except Exception as e:
        print(f"⚠️ Error sending test notification: {e}")
        # We don't want to fail the request if the test notification fails
    
    return {"message": "FCM token updated successfully"}

@router.post("/clear_fcm_token")
def clear_fcm_token(token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)
    
    # Remove the FCM token from the user's record
    result = users_collection.update_one(
        {"username": username},
        {"$unset": {"fcm_token": ""}}
    )
    
    print(f"🔔 FCM token cleared for user {username}")
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"message": "FCM token cleared successfully"}

# ✅ Utility function to send notification to a specific user
def send_notification_to_user(username, title, body, screen="HomeScreen", data=None):
    # Get the user's FCM token
    user = users_collection.find_one({"username": username})
    
    if not user or "fcm_token" not in user or not user["fcm_token"]:
        print(f"⚠️ Cannot send notification: No FCM token for user {username}")
        return False
    
    try:
        # Send the push notification
        result = send_push_notification_v1(
            token=user["fcm_token"],
            title=title,
            body=body,
            screen=screen,
            data=data
        )
        print(f"✅ Notification sent to {username}: {result}")
        return True
    except Exception as e:
        print(f"⚠️ Error sending notification to {username}: {e}")
        return False

# ✅ Create a notification for new alerts
def create_notification_for_alert(username, alert_id, title, message, alert_type="health"):
    try:
        notification = {
            "username": username,
            "title": title,
            "message": message,
            "type": alert_type,
            "category": "alerts",
            "data": {"alert_id": str(alert_id)},
            "created_at": datetime.utcnow(),
            "read": False
        }
        
        result = notifications_collection.insert_one(notification)
        print(f"✅ Notification created for alert: {result.inserted_id}")
        
        # Send push notification
        send_notification_to_user(
            username=username,
            title=title,
            body=message,
            screen="AlertDetail",
            data={"alert_id": str(alert_id)}
        )
        
        return str(result.inserted_id)
    except Exception as e:
        print(f"⚠️ Error creating notification: {e}")
        return None

# Example: Modify your create_alert endpoint to send notifications
"""
@router.post("/create_alert")
def create_alert(alert_data: dict = Body(...), token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Your existing code to create an alert
    # ...
    
    # After creating the alert, send a notification
    metric = alert_data.get("metric", "health")
    create_notification_for_alert(
        username=username,
        alert_id=alert_id,  # From your alert creation code
        title=f"{metric.title()} Alert",
        message=f"We detected an issue with your {metric} levels. Tap to learn more.",
        alert_type=metric
    )
    
    return {"message": "Alert created", "alert_id": str(alert_id)}
"""