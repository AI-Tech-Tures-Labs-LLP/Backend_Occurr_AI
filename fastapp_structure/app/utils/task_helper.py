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

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE_URL"))


# Generate daily tasks based on user profile and schedule
def generate_daily_tasks_from_profile(user):
    username = user["username"]
    today = datetime.utcnow().date()

    schedule = user.get("schedule", {})
    task_templates = []

    # def dt_today_at(time_str):
    #     t = datetime.strptime(time_str, "%H:%M").time()
    #     return datetime.combine(today,t)

    # def dt_end_of_day():
    #     return datetime.combine(today, datetime.max.time())
    def dt_today_at(time_str):
        if not time_str:
            # Handle the case when time_str is None (set a default time)
            # For example, default to 09:00
            time_str = "09:00"
        t = datetime.strptime(time_str, "%H:%M").time()
        return datetime.combine(today, t)

    def dt_end_of_day():
        return datetime.combine(today, datetime.max.time())

    # Breakfast Task
    if "breakfast_time" in schedule:
        task_templates.append({
            "title": "Log your breakfast",
            "type": "meal",
            "trigger_time": dt_today_at(schedule["breakfast_time"]),
            "metadata": {"meal": "breakfast"},
            "expires_at": dt_end_of_day()
        })

    # Lunch Task
    if "lunch_time" in schedule:
        task_templates.append({
            "title": "Log your lunch",
            "type": "meal",
            "trigger_time": dt_today_at(schedule["lunch_time"]),
            "metadata": {"meal": "lunch"},
            "expires_at": dt_end_of_day()
        })

    # Dinner Task
    if "dinner_time" in schedule:
        task_templates.append({
            "title": "Log your dinner",
            "type": "meal",
            "trigger_time": dt_today_at(schedule["dinner_time"]),
            "metadata": {"meal": "dinner"},
            "expires_at": dt_end_of_day()
        })

    # Meditation Task
    if "meditation_time" in schedule:
        task_templates.append({
            "title": "Do your daily meditation",
            "type": "meditation",
            "trigger_time": dt_today_at(schedule["meditation_time"]),
            "expires_at": dt_end_of_day()
        })


    # Exercise Task
    if "exercise_time" in schedule or "workout_time" in schedule:
        task_templates.append({
            "title": "Log your exercise",
            "type": "exercise",
            "trigger_time": dt_today_at(schedule.get("exercise_time", schedule.get("workout_time"))),
            "expires_at": dt_end_of_day()
        })

   #Breathing Task
    # Check for breathing time or breathing exercise time
    if "breathing_time" in schedule or "breathing_exercise_time" in schedule:
        task_templates.append({
            "title": "Do your breathing exercises",
            "type": "breathing",
            "trigger_time": dt_today_at(schedule.get("breathing_time", schedule.get("breathing_exercise_time"))),
            "expires_at": dt_end_of_day()
        })

    # Reflect Task (before sleep)
    if "sleep_time" in schedule:
        task_templates.append({
            "title": "Reflect on your workday",
            "type": "reflection",
            "trigger_time": dt_today_at(schedule["sleep_time"]) - timedelta(hours=1),
            "expires_at": dt_end_of_day()
        })




    for task in task_templates:
        # ✅ Check if a similar task already exists today
        existing = task_collection.find_one({
            "username": username,
            "title": task["title"],
            "trigger_time": task["trigger_time"]
        })

        if existing:
            # print(f"⏩ Task already exists for {username}: {task['title']} at {task['trigger_time']}")
            continue

        task_doc = build_task_doc(
            username=username,
            type=task["type"],
            title=task["title"],
            trigger_time=task["trigger_time"],
            expires_at=task["expires_at"],
            metadata=task.get("metadata")
        )
        task_collection.insert_one(task_doc)
        print(f"✅ Created task for {username}: {task['title']} at {task['trigger_time']}")


# Complete a task and update the journal
def complete_task(username: str, task_id: str, task_content: str, image_url: str = None):
    task = task_collection.find_one({"_id": ObjectId(task_id), "username": username})
    if not task or task.get("completed"):
        raise ValueError("Task not found or already completed.")

    now = datetime.utcnow()
    today = now.date()
    timestamp = now

    # 🔐 Block duplicate breathing logs
    if task["type"] == "breathing":
        start = datetime.combine(today, datetime.min.time())
        end = datetime.combine(today, datetime.max.time())
        already_logged = breathing_collection.find_one({
            "username": username,
            "timestamp": {"$gte": start, "$lte": end}
        })
        if already_logged:
            return {
                "message": "Breathing session already recorded for today.",
                "journal_id": None,
                "assistant_reply": None
            }

    # ✅ Use task title as assistant prompt
    assistant_prompt = f"Can you tell me more about: {task['title']}?"
    user_reply = task_content.strip()

    # Build chat conversation so far
    conversation = [
        {"role": "assistant", "content": assistant_prompt, "timestamp": timestamp},
        {"role": "user", "content": user_reply, "timestamp": timestamp}
    ]

    # 📊 Optional: Handle food image if task is 'meal'
    if task["type"] == "meal" and image_url:
        try:
            vision_response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {"role": "system", "content": "You're a nutrition assistant. Estimate calories and food details from image."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "What can you tell me about this meal?"},
                        {"type": "image_url", "image_url": image_url}
                    ]}
                ],
                max_tokens=100
            )
            nutrition_info = vision_response.choices[0].message.content.strip()
            conversation.append({"role": "assistant", "content": nutrition_info, "timestamp": timestamp})
        except Exception as e:
            print(f"GPT Vision error: {e}")

    # 🤔 Generate AI follow-up
    follow_up = None
    try:
        followup_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": (
                    "You're a friendly journaling assistant. After each user response, ask a short helpful question that continues the conversation based on the task title."
                )},
                {"role": "assistant", "content": assistant_prompt},
                {"role": "user", "content": user_reply}
            ],
            max_tokens=50,
            temperature=0.6
        )
        follow_up = followup_response.choices[0].message.content.strip()
        conversation.append({"role": "assistant", "content": follow_up, "timestamp": timestamp})
    except Exception as e:
        print("GPT follow-up error:", e)

    # 📚 Save to today's journal (conversation type)
    journal = journals_collection.find_one({
        "username": username,
        "date": today,
        "type": "conversation"
    })

    if journal:
        journals_collection.update_one(
            {"_id": journal["_id"]},
            {"$push": {"entries": {"$each": conversation}}}
        )
        journal_id = journal["_id"]
    else:
        new_journal = {
            "username": username,
            "type": "conversation",
            "date": today,
            "timestamp": timestamp,
            "entries": conversation,
            "mood": None
        }
        result = journals_collection.insert_one(new_journal)
        journal_id = result.inserted_id

    # ✅ Mark task complete
    task_collection.update_one(
        {"_id": task["_id"]},
        {"$set": {
            "completed": True,
            "completed_at": timestamp,
            "journal_entry_id": str(journal_id)
        }}
    )

    return {
        "message": "Task completed and journal updated.",
        "journal_id": str(journal_id),
        "assistant_reply": follow_up
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