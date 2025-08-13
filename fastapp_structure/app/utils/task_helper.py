from app.db.task_model import task_collection,build_task_doc
from app.db.database import users_collection
from app.db.journal_model import journals_collection, get_or_create_daily_journal,save_journal_entry
from app.db.notification_model import notifications_collection, save_notification
from datetime import datetime, timedelta
from bson import ObjectId
from app.utils.firebase_push import send_push_notification_v1 
from app.db.breathing_exercise import breathing_collection
from openai import OpenAI
import os
from typing import Optional
    #     return datetime.combine(today, datetime.max.time())
import re
from datetime import datetime, time

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE_URL"))


# Generate daily tasks based on user profile and schedule
# def generate_daily_tasks_from_profile(user):
#     username = user["username"]
#     today = datetime.utcnow().date()
#     schedule = user.get("schedule", {})
#     task_templates = []

#     # def dt_today_at(time_str):
#     #     t = datetime.strptime(time_str, "%H:%M").time()
#     #     return datetime.combine(today,t)

#     # def dt_end_of_day():


#     def dt_today_at(time_str):
#         # Fallback if None or not a string
#         if not isinstance(time_str, str):
#             print(f"⚠️ Invalid time format (not string): {time_str}. Using default 09:00")
#             time_str = "09:00"

#         time_str = time_str.strip()

#         # Validate format using regex
#         if not re.match(r"^(?:[01]?\d|2[0-3]):[0-5]\d$", time_str):
#             print(f"⚠️ Time string '{time_str}' is not in valid HH:MM format. Using default 09:00")
#             time_str = "09:00"

#         try:
#             t = datetime.strptime(time_str, "%H:%M").time()
#         except ValueError:
#             print(f"⚠️ Error parsing time: {time_str}. Using default 09:00")
#             t = datetime.strptime("09:00", "%H:%M").time()

#         return datetime.combine(today, t)


#     def dt_end_of_day():
#         return datetime.combine(today, datetime.max.time())

#     # Breakfast Task
#     if "breakfast_time" in schedule:
#         task_templates.append({
#             "title": "Log your breakfast",
#             "type": "meal",
#             "trigger_time": dt_today_at(schedule["breakfast_time"]),
#             "metadata": {"meal": "breakfast"},
#             "expires_at": dt_end_of_day()
#         })

#     # Lunch Task
#     if "lunch_time" in schedule:
#         task_templates.append({
#             "title": "Log your lunch",
#             "type": "meal",
#             "trigger_time": dt_today_at(schedule["lunch_time"]),
#             "metadata": {"meal": "lunch"},
#             "expires_at": dt_end_of_day()
#         })

#     # Dinner Task
#     if "dinner_time" in schedule:
#         task_templates.append({
#             "title": "Log your dinner",
#             "type": "meal",
#             "trigger_time": dt_today_at(schedule["dinner_time"]),
#             "metadata": {"meal": "dinner"},
#             "expires_at": dt_end_of_day()
#         })

#     # Meditation Task
#     if "meditation_time" in schedule:
#         task_templates.append({
#             "title": "Do your daily meditation",
#             "type": "meditation",
#             "trigger_time": dt_today_at(schedule["meditation_time"]),
#             "expires_at": dt_end_of_day()
#         })


#     # Exercise Task
#     if "exercise_time" in schedule or "workout_time" in schedule:
#         task_templates.append({
#             "title": "Log your exercise",
#             "type": "exercise",
#             "trigger_time": dt_today_at(schedule.get("exercise_time", schedule.get("workout_time"))),
#             "expires_at": dt_end_of_day()
#         })

#    #Breathing Task
#     # Check for breathing time or breathing exercise time
#     if "breathing_time" in schedule or "breathing_exercise_time" in schedule:
#         task_templates.append({
#             "title": "Do your breathing exercises",
#             "type": "breathing",
#             "trigger_time": dt_today_at(schedule.get("breathing_time", schedule.get("breathing_exercise_time"))),
#             "expires_at": dt_end_of_day()
#         })

#     # Reflect Task (before sleep)
#     if "sleep_time" in schedule:
#         task_templates.append({
#             "title": "Reflect on your workday",
#             "type": "reflection",
#             "trigger_time": dt_today_at(schedule["sleep_time"]) - timedelta(hours=1),
#             "expires_at": dt_end_of_day()
#         })


#     for task in task_templates:
#         # ✅ Check if a similar task already exists today
#         existing = task_collection.find_one({
#             "username": username,
#             "title": task["title"],
#             "trigger_time": task["trigger_time"]
#         })

#         if existing:
#             # print(f"⏩ Task already exists for {username}: {task['title']} at {task['trigger_time']}")
#             continue

#         task_doc = build_task_doc(
#             username=username,
#             type=task["type"],
#             title=task["title"],
#             trigger_time=task["trigger_time"],
#             expires_at=task["expires_at"],
#             metadata=task.get("metadata")
#         )
#         task_collection.insert_one(task_doc)
#         print(f"✅ Created task for {username}: {task['title']} at {task['trigger_time']}")

def generate_daily_tasks_from_profile(user):
    from datetime import datetime, time, timedelta
    import re

    username = user["username"]
    today = datetime.utcnow().date()
    schedule = user.get("schedule", {})
    task_templates = []

    def dt_today_at(time_str):
        if not isinstance(time_str, str):
            time_str = "09:00"
        time_str = time_str.strip()

        if not re.match(r"^(?:[01]?\d|2[0-3]):[0-5]\d$", time_str):
            time_str = "09:00"

        try:
            t = datetime.strptime(time_str, "%H:%M").time()
        except ValueError:
            t = datetime.strptime("09:00", "%H:%M").time()

        return datetime.combine(today, t)

    def dt_end_of_day():
        return datetime.combine(today, datetime.max.time())

    def guess_type_from_key(key):
        if "meal" in key or "breakfast" in key or "lunch" in key or "dinner" in key:
            return "meal"
        if "meditation" in key:
            return "meditation"
        if "exercise" in key or "workout" in key:
            return "exercise"
        if "breathing" in key:
            return "breathing"
        if "sleep" in key or "reflection" in key:
            return "reflection"
        if "journal" in key:
            return "journaling"
        if "focus" in key or "study" in key:
            return "focus"
        return "general"

    def format_title_from_key(key):
        label = key.replace("_time", "").replace("_", " ").strip().capitalize()
        if "meal" in key or "breakfast" in key or "lunch" in key or "dinner" in key:
            return f"Log your {label}"
        if "meditation" in key:
            return "Do your daily meditation"
        if "exercise" in key or "workout" in key:
            return "Log your exercise"
        if "breathing" in key:
            return "Do your breathing exercises"
        if "sleep" in key or "reflection" in key:
            return "Reflect on your workday"
        if "journal" in key:
            return "Write your journal entry"
        return f"Do your {label}"

    for key, time_str in schedule.items():
        if not key.endswith("_time"):
            continue  # ignore non-time keys

        task_type = guess_type_from_key(key)
        title = format_title_from_key(key)
        trigger_time = dt_today_at(time_str)

        # Optional: offset reflection 1 hour before sleep
        if task_type == "reflection":
            trigger_time -= timedelta(hours=1)

        task_templates.append({
            "title": title,
            "type": task_type,
            "trigger_time": trigger_time,
            "expires_at": dt_end_of_day()
        })

    # Create tasks in DB if not already present
    for task in task_templates:
        existing = task_collection.find_one({
            "username": username,
            "title": task["title"],
            "trigger_time": task["trigger_time"]
        })

        if existing:
            continue

        task_doc = build_task_doc(
            username=username,
            type=task["type"],
            title=task["title"],
            trigger_time=task["trigger_time"],
            expires_at=task["expires_at"],
        )

        task_collection.insert_one(task_doc)
        print(f"✅ Created task for {username}: {task['title']} at {task['trigger_time']}")

# Complete a task and update the journal
# from datetime import datetime, time
# from typing import Optional


from datetime import datetime, time
from typing import Optional

def complete_task(username: str, task_id: str, task_content: Optional[str] = None, image_url: Optional[str] = None):
    now = datetime.utcnow()
    today = now.date()
    start = datetime.combine(today, time.min)
    end = datetime.combine(today, time.max)

    # Fetch the task by ID and username
    task = task_collection.find_one({"_id": ObjectId(task_id), "username": username})
    if not task:
        raise ValueError("Task not found")

    # Check if the task has expired
    expires_at = task.get("expires_at")
    if expires_at and now > expires_at:
        raise ValueError("Task has expired and cannot be completed.")

    # Breathing: prevent multiple per day
    if task["type"] == "breathing" and task_content is None:
        already_logged = breathing_collection.find_one({
            "username": username,
            "timestamp": {"$gte": start, "$lte": end}
        })
        if already_logged:
            return {
                "message": "Breathing already logged today.",
                "journal_id": None,
                "assistant_reply": "No further actions needed.",
                "completed": False
            }

    # Find/create today's journal
    journal = journals_collection.find_one({
        "username": username,
        "type": "conversation",
        "timestamp": {"$gte": start, "$lte": end}
    })

    if not journal:
        new_journal = {
            "username": username,
            "type": "conversation",
            "timestamp": now,
            "entries": [],
            "mood": None
        }
        journal_id = journals_collection.insert_one(new_journal).inserted_id
    else:
        journal_id = journal["_id"]

    def append_to_journal(entries):
        journals_collection.update_one(
            {"_id": journal_id},
            {"$push": {"entries": {"$each": entries}}}
        )

    conv_count = task.get("conv_count", 0)
    MAX_TURNS = 4  # max user replies

    # If task_content is None, ask the initial assistant message
    if task_content is None:
        assistant_msg = f"Can you tell me more about: {task['title']}?"
        append_to_journal([{
            "role": "assistant", "content": assistant_msg, "timestamp": now
        }])

        task_collection.update_one(
            {"_id": task["_id"]},
            {"$set": {"status": "in_progress", "conv_count": conv_count, "last_prompt": assistant_msg}}
        )

        return {
            "message": "Started chat for task.",
            "journal_id": str(journal_id),
            "assistant_reply": assistant_msg,
            "completed": False
        }

    # If task_content is provided, generate a dynamic follow-up question using GPT
    if task_content:
        try:
            # Request GPT model to generate a follow-up question based on task_content
            follow_response = client.chat.completions.create(
                model=os.getenv("OPENAI_API_MODEL"),  # Ensure the model is correct
                messages=[
                    {"role": "system", "content": (
                        "You are a friendly journaling assistant. Based on the user's response, "
                        "ask a short helpful question that continues the conversation."
                    )},
                    {"role": "user", "content": task_content}
                ],
                max_tokens=50,
                temperature=0.6
            )
            
            # Check if GPT returned a valid response
            if follow_response.choices and follow_response.choices[0].message.content.strip():
                follow_up = follow_response.choices[0].message.content.strip()
            else:
                # If no valid response, ask for more details
                follow_up = "Could you clarify what you meant by that? Can you provide more details?"
            
        except Exception as e:
            print("Follow-up GPT error:", e)
            follow_up = "Sorry, I didn't catch that. Can you share more about your task?"

        entries = [
            {"role": "user", "content": task_content.strip(), "timestamp": now},
            {"role": "assistant", "content": follow_up, "timestamp": now}
        ]
        conv_count += 1
    # Optional: handle meal image
    if task["type"] == "meal" and image_url:
        try:
            vision_response = client.chat.completions.create(
                model="gpt-4",  # Ensure the model is valid
                messages=[
                    {"role": "system", "content": "You're a nutrition assistant. Estimate calories and food details from image."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "What can you tell me about this meal?"},
                        {"type": "image_url", "image_url": image_url}
                    ]}
                ],
                max_tokens=100
            )
            vision_info = vision_response.choices[0].message.content.strip()
            entries.append({"role": "assistant", "content": vision_info, "timestamp": now})
        except Exception as e:
            print("GPT vision error:", e)
            entries.append({"role": "assistant", "content": "Error processing image.", "timestamp": now})

    # Decide if follow-up or complete
    is_complete = conv_count >= MAX_TURNS
    if not is_complete:
        follow_up = follow_up or "Sorry, I didn't catch that. Can you share more about your task?"

    # Save entries to journal
    append_to_journal(entries)

    task_updates = {
        "conv_count": conv_count,
        "journal_entry_id": str(journal_id)
    }

    if is_complete:
        task_updates.update({
            "completed": True,
            "status": "completed",
            "completed_at": now
        })
    else:
        task_updates.update({
            "status": "in_progress",
            "last_prompt": follow_up
        })

    task_collection.update_one(
        {"_id": task["_id"]},
        {"$set": task_updates}
    )

    return {
        "message": "Task completed!" if is_complete else "Task updated.",
        "journal_id": str(journal_id),
        "assistant_reply": follow_up,
        "completed": is_complete
    }








# Check and notify pending tasks for all users

def check_and_notify_pending_tasks_for_all_users():
    now = datetime.utcnow()
    users = users_collection.find()

    for user in users:
        pending_tasks = task_collection.find({
            "username": user["username"],
            "trigger_time": {"$lte": now},
            "completed": False,
            "notified": {"$ne": True},
            "expires_at": {"$gte": now}
        })

        for task in pending_tasks:
            task_type = task.get("type")
            meal = task.get("metadata", {}).get("meal")

            if task_type == "meal" and meal:
                message = f"🍽️ Don’t forget to log your {meal}!"
            elif task_type == "meditation":
                message = "🧘 Time for your daily meditation. Take a moment to breathe."
            elif task_type == "exercise":
                message = "🏋️ Ready to log your workout? Let’s move!"
            elif task_type == "breathing":
                message = "🌬️ Don’t forget your breathing exercise session."
            elif task_type == "reflection":
                message = "📝 Reflect on your day before bed. Want to jot something down?"
            else:
                message = "⏰ Reminder: You have a pending task to complete."

            send_push_notification(
                username=task["username"],
                title="Task Reminder",
                body=message,
                task_id=str(task["_id"])  # ✅ Include task_id
            )

            task_collection.update_one({"_id": task["_id"]}, {"$set": {"notified": True}})
            print(f"✅ Reminder sent and task marked notified: {task['_id']}")




# Send push notification to user and save notification record
# def send_push_notification(username: str, title: str, body: str, task_id=None, alert_id=None):
#     user = users_collection.find_one({"username": username}, {"device_token": 1})

#     # Try to send push
#     if user and "device_token" in user:
#         token = user["device_token"]
#         print(f"📲 Sending push notification to {username}: {title} — {body}")
#         # TODO: Add real FCM/Expo integration here
#     else:
#         print(f"⚠️ No device token found for user {username}. Skipping push.")

#     # Always save the notification
#     save_notification(
#         username=username,
#         title=title,
#         body=body,
#         read=False,
#         task_id=task_id,
#         alert_id=alert_id
#     )


def send_push_notification(username: str, title: str, body: str, task_id=None, alert_id=None):
    user = users_collection.find_one({"username": username}, {"device_token": 1})

    if user and "device_token" in user:
        token = user["device_token"]
        print(f"📲 Sending push notification to {username}: {title} — {body}")

        try:
            send_push_notification_v1(
                token=token,
                title=title,
                body=body,
                screen="HomeScreen" if alert_id else "TaskScreen"
            )
        except Exception as e:
            print(f"❌ Failed to send push to {username}: {e}")
    else:
        print(f"⚠️ No device token found for user {username}. Skipping push.")

    # ✅ Always save the notification in DB
    save_notification(
        username=username,
        title=title,
        body=body,
        read=False,
        task_id=task_id,
        alert_id=alert_id
    )



def get_or_create_daily_journal(username: str):
    today = datetime.utcnow().date()
    journal = journals_collection.find_one({
        "username": username,
        "date": today
    })

    if not journal:
        # Create a new daily journal entry if it doesn't exist
        journal = get_or_create_daily_journal(username)

    return journal