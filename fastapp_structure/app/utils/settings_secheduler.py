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

    # now = datetime.now().strftime("%H:%M")
    # users = get_all_users_and_times()  # should return list of { username, journal_time }

    # for user in users:
    #     journal_time = user.get("journal_time")
    #     username = user.get("username")

    #     if not username or not journal_time:
    #         continue

    #     if journal_time == now:
    #         try:
    #             prompt = generate_journal_prompt("scheduled", username=username, time_of_day=get_time_of_day())
    #         except Exception as e:
    #             print(f"⚠️ Failed to generate prompt for {username}: {e}")
    #             prompt = f"How are you feeling this {get_time_of_day()}?"

    #         save_journal_entry(
    #             username=username,
    #             entry_type="scheduled",
    #             prompt=prompt,
    #             response="",
    #             tags=["auto-scheduled"],
    #             mood="neutral"
    #         )
    #         print(f"✅ Journal prompt scheduled for {username} at {now}")

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(check_journal_times, trigger="interval", minutes=1, id="check_journal_times", misfire_grace_time=60)
    scheduler.start()
