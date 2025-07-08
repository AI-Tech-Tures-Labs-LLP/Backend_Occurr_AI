from datetime import datetime
from app.db.journal_model import save_journal_entry, get_journals_by_day
from app.utils.journal_prompt_generator import generate_journal_prompt, get_time_of_day
from app.db.database import get_all_users_and_times,users_collection  # ensure this returns list of dicts
 # ensure this is the correct collection for users
from apscheduler.schedulers.background import BackgroundScheduler

def check_journal_times():

    now_str = datetime.now().strftime("%H:%M")
    today_str = datetime.now().strftime("%Y-%m-%d")

    users = users_collection.find({"journal_time": now_str})

    for user in users:
        username = user["username"]

        # ⛔ Skip if already journaled today
        journal_entries = get_journals_by_day(username, today_str)
        if journal_entries:
            continue

        # ✅ Get prompt from GPT (or fallback if GPT fails)
        prompt = generate_journal_prompt(
            category="scheduled",
            username=username,
            time_of_day=get_time_of_day()
        )

        if not prompt:
            print(f"⚠️ Failed to generate prompt for {username}, using default.")
            prompt = f"How are you feeling this {get_time_of_day()}?"

        # 📬 Send in-app notification
        users_collection.update_one(
            {"username": username},
            {"$push": {
                "notifications": {
                    "message": prompt,
                    "type": "journal_reminder",
                    "read": False,
                    "redirect": "/journal/conversation",
                    "created_at": datetime.utcnow()
                }
            }}
        )
        print(f"📬 Journal reminder sent to {username}: {prompt}")
