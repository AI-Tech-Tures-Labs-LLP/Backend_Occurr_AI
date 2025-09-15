from app.db.task_model import task_collection,build_task_doc
from app.db.database import users_collection
from app.db.journal_model import journals_collection, get_or_create_daily_journal,save_journal_entry
from app.db.notification_model import notifications_collection, save_notification
from datetime import datetime, timedelta
from bson import ObjectId
from app.utils.firebase_push import send_push_notification_v1
from app.db.breathing_exercise import breathing_collection
from groq import Groq
from typing import Any, Dict, List, Optional
import os
from typing import Optional
    #     return datetime.combine(today, datetime.max.time())
import re
from datetime import datetime, time
from openai import OpenAI
import os, uuid, base64, mimetypes
import requests  # <-- add this
from datetime import datetime, time, timezone
import boto3
 

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEYS"),base_url=os.getenv("OPENAI_API_BASE_URLS"))
client = Groq(api_key=os.getenv("OPENAI_API_KEY"))
 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

 
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
            return key.replace("_time", "").replace("_", " ").strip().lower()
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
        day_start = datetime.combine(today, datetime.min.time())
        day_end   = datetime.combine(today, datetime.max.time())
        day_start = datetime.combine(today, datetime.min.time())
        day_end   = datetime.combine(today, datetime.max.time())
        existing = task_collection.find_one({
            "username": username,
            "title": task["title"],
            "trigger_time": {"$gte": day_start, "$lte": day_end},
            "trigger_time": {"$gte": day_start, "$lte": day_end},
        })
 
        if existing:
            # If the time changed, update it; else ski
            if existing.get("trigger_time") != task["trigger_time"]:
                task_collection.update_one(
                    {"_id": existing["_id"]},
                    {"$set": {
                        "trigger_time": task["trigger_time"],
                        "expires_at": task["expires_at"],
                        "updated_at": datetime.utcnow(),
                        "meta.schedule_sync_reason": "daily_regen_time_adjust"
                    }}
                )
            # If the time changed, update it; else skip
            if existing.get("trigger_time") != task["trigger_time"]:
                task_collection.update_one(
                    {"_id": existing["_id"]},
                    {"$set": {
                        "trigger_time": task["trigger_time"],
                        "expires_at": task["expires_at"],
                        "updated_at": datetime.utcnow(),
                        "meta.schedule_sync_reason": "daily_regen_time_adjust"
                    }}
                )
            continue
 
        # create new
 
        # create new
        task_doc = build_task_doc(
            username=username,
            type=task["type"],
            title=task["title"],
            trigger_time=task["trigger_time"],
            expires_at=task["expires_at"],
        )
        task_collection.insert_one(task_doc)

        print(f"✅ Created task for {username}: {task['title']} at {task['trigger_time']}")
 

# ==== ENV HELPERS ====
def get_openai_client() -> OpenAI:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")
    # return OpenAI(api_key=key)
    return OpenAI(api_key=key, base_url="https://api.deepseek.com")

def get_groq_client() -> Groq:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required")
    return Groq(api_key=key)

# ---- prompts ----
NUTRITION_SYSTEM_PROMPT = (
    "You are an expert nutritionist. Look carefully at the food items in the image "
    "and estimate total calories. List each item with an estimated calorie count in this format:\n"
    "1. Item 1 - N calories\n2. Item 2 - N calories\n...\n"
    "Add a brief note about assumptions/portion sizes."
)

JOURNAL_FOLLOWUP_SYS = (
    "You are a friendly journaling assistant.\n"
    "Use the full conversation context below to ask ONE short follow-up question "
    "that genuinely advances the topic. Refer to specific details the user mentioned. "
    "Avoid repeating questions already asked. Keep it under 25 words."
)

# ---- helpers (paste once in your module) ----
def _bytes_from_image_url(image_url: str) -> tuple[bytes, str]:
    """Supports https:// and data: URLs."""
    if image_url.startswith("data:"):
        header, b64data = image_url.split(",", 1)
        # e.g. data:image/png;base64,...
        mime = header.split(";")[0].split("data:")[1] or "image/png"
        return base64.b64decode(b64data), mime
    resp = requests.get(image_url, timeout=30)
    resp.raise_for_status()
    mime = resp.headers.get("content-type", "image/jpeg")
    return resp.content, mime

def upload_image_to_s3(
    image_url: str,
    *,
    bucket: str,
    key_prefix: str = "journal",
    region: str | None = None,
    public_base: str | None = None,   # okay to keep; just won’t use ACLs
    presign_ttl: int = 7 * 24 * 3600, # 7 days
) -> str:
    # fetch bytes (supports https:// and data: URLs)
    if image_url.startswith("data:"):
        header, b64data = image_url.split(",", 1)
        mime = header.split(";")[0].split("data:")[1] or "image/png"
        data = base64.b64decode(b64data)
    else:
        r = requests.get(image_url, timeout=30)
        r.raise_for_status()
        data = r.content
        mime = r.headers.get("content-type", "image/jpeg")

    ext = mimetypes.guess_extension(mime) or ".jpg"
    key = f"{key_prefix}/{datetime.now(timezone.utc).strftime('%Y/%m/%d')}/{uuid.uuid4().hex}{ext}"

    s3 = boto3.client("s3", region_name=region)

    # IMPORTANT: no ACL here (ACLs are disabled)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        ContentType=mime,
    )

    # If you’ve configured a bucket policy/CloudFront to allow public reads,
    # you can still return a clean URL; otherwise use a presigned URL.
    if public_base:
        # This will only be publicly readable if your bucket policy allows GetObject.
        return f"{public_base.rstrip('/')}/{key}"

    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=presign_ttl,
    )

def analyze_meal_with_openai(image_https_url: str,
                             *,
                             model: str = "gpt-4o-mini",
                             extra_text: str = "Analyze this meal." ) -> str:
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        temperature=0.3,
        max_tokens=600,
        messages=[
            {"role": "system", "content": NUTRITION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": extra_text},
                    {"type": "image_url", "image_url": {"url": image_https_url}},
                ],
            },
        ],
    )
    return (resp.choices[0].message.content or "").strip()

# ---- context utils for previous-turn-aware follow-ups ----
def _normalize_entry_to_text(entry: dict) -> str:
    """Flatten a journal entry to plain text for LLM context (hide raw URLs)."""
    text = (entry.get("content") or "").strip()
    atts = entry.get("attachments") or []
    if atts:
        kinds = sorted({a.get("type", "file") for a in atts})
        text = (text + f"\n[attachments present: {', '.join(kinds)}]").strip()
    return text

def build_context_messages(journal_doc: dict, new_user_text: str, *, max_turns: int = 12) -> list[dict]:
    """Build Groq messages from prior journal entries (last `max_turns`) + the new user text."""
    msgs = [{"role": "system", "content": JOURNAL_FOLLOWUP_SYS}]
    entries = (journal_doc or {}).get("entries", [])
    convo = []
    for e in entries:
        r = e.get("role")
        if r in ("user", "assistant"):
            txt = _normalize_entry_to_text(e)
            if txt:
                convo.append({"role": r, "content": txt})
    msgs.extend(convo[-max_turns:])
    if new_user_text:
        msgs.append({"role": "user", "content": new_user_text.strip()})
    return msgs

# ---- your function (S3 + Vision integrated + context-aware Groq follow-up) ----
def complete_task(
    username: str,
    task_id: str,
    task_content: Optional[str] = None,
    image_url: Optional[str] = None
):
    """
    Conversation-style task completion that:
      - Creates/fetches today's 'conversation' journal
      - Logs user text (if any) and generates a context-aware Groq follow-up
      - If a meal task and an image URL is provided: uploads photo to S3, saves S3 URL in journal,
        and runs OpenAI Vision (gpt-4o-mini) on that S3 URL
      - Updates task conv_count/status and returns assistant reply
    """
    now = datetime.utcnow()
    today = now.date()
    start = datetime.combine(today, time.min)
    end = datetime.combine(today, time.max)

    # 1) Fetch the task
    task = task_collection.find_one({"_id": ObjectId(task_id), "username": username})
    if not task:
        raise ValueError("Task not found")

    # 2) Expiry guard
    expires_at = task.get("expires_at")
    if expires_at and now > expires_at:
        raise ValueError("Task has expired and cannot be completed.")

    # 3) Breathing: prevent multiple per day
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

    # 4) Find/create today's journal
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
        journal = {**new_journal, "_id": journal_id}
    else:
        journal_id = journal["_id"]

    def append_to_journal(entries_to_add: List[Dict[str, Any]]):
        if entries_to_add:
            journals_collection.update_one(
                {"_id": journal_id},
                {"$push": {"entries": {"$each": entries_to_add}}}
            )

    # ✅ Always initialize
    entries: List[Dict[str, Any]] = []
    follow_up: Optional[str] = None
    conv_count = task.get("conv_count", 0)
    MAX_TURNS = 4

    # ----- (A) Handle user text (context-aware Groq follow-up) -----
    if task_content:
        try:
            groq_client = get_groq_client()
            journal_doc = journals_collection.find_one({"_id": journal_id}, {"entries": 1}) or {"entries": []}
            ctx_messages = build_context_messages(journal_doc, task_content, max_turns=12)

            groq_resp = groq_client.chat.completions.create(
                model=os.getenv("OPENAI_API_KEYS", "llama-3.1-70b-versatile"),
                messages=ctx_messages,
                max_tokens=64,
                temperature=0.6,
            )
            follow_up = (groq_resp.choices[0].message.content or "").strip()
            if not follow_up:
                follow_up = "Could you clarify that a bit more?"
        except Exception as e:
            print("Groq follow-up error:", e)
            follow_up = "Thanks! Anything else you'd like to add?"

        entries.extend([
            {"role": "user", "content": task_content.strip(), "timestamp": now},
            {"role": "assistant", "content": follow_up, "timestamp": now},
        ])
        conv_count += 1

    # ----- (B) Handle meal image (S3 + OpenAI Vision) -----
    if task["type"] in ["meal", "snack", "breakfast", "lunch", "dinner"] and image_url:
        print("Processing meal image for vision analysis...===========")
        try:
            # 1) Upload to S3 and get a URL we can store/call with
            aws_region = os.getenv("AWS_REGION", "ap-south-1")
            s3_bucket  = os.getenv("AWS_S3_BUCKET")
            if not s3_bucket:
                raise RuntimeError("S3_BUCKET env var is required")
            s3_public_base = os.getenv("S3_PUBLIC_BASE")  # optional

            s3_url = upload_image_to_s3(
                image_url,
                bucket=s3_bucket,
                key_prefix="journal",
                region=aws_region,
                public_base=s3_public_base,
            )

            # 2) Attach image to journal entry (user bubble) with the S3 URL
            entries.append({
                "role": "user",
                "content": "Here's my meal photo" if not task_content else "Attached meal photo",
                "timestamp": now,
                "attachments": [{"type": "image", "url": s3_url, "timestamp": now}],
            })

            # 3) Vision analysis with OpenAI using the same S3 URL
            analysis = analyze_meal_with_openai(
                image_https_url=s3_url,
                model=os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini"),
                extra_text="Please analyze this meal for total calories, macros, items, and a brief health tip. If unsure, state assumptions.",
            )
            if not analysis:
                analysis = "I couldn't analyze the image right now."

            entries.append({"role": "assistant", "content": analysis, "timestamp": now})
            follow_up = (follow_up or analysis)

        except Exception as e:
            print("Meal image pipeline error:", e)
            entries.append({
                "role": "assistant",
                "content": "I couldn't process the image. Please send a valid HTTPS or data URL.",
                "timestamp": now
            })

    # ----- (C) If neither text nor image, start chat (context-aware opener) -----

    if not task_content and not image_url:
        assistant_msg = f"Can you tell me more about: {task['title']}?"
        entries.append({"role": "assistant", "content": assistant_msg, "timestamp": now})
        task_collection.update_one(
            {"_id": task["_id"]},
            {"$set": {"status": "in_progress", "conv_count": conv_count, "last_prompt": assistant_msg}}
        )
        append_to_journal(entries)
        return {
            "message": "Started chat for task.",
            "journal_id": str(journal_id),
            "assistant_reply": assistant_msg,
            "completed": False
        }

    # if not task_content and not image_url:
    #     try:
    #         groq_client = get_groq_client()
    #         journal_doc = journals_collection.find_one({"_id": journal_id}, {"entries": 1}) or {"entries": []}
    #         ctx_messages = build_context_messages(journal_doc, "Start a helpful check-in.", max_turns=12)
    #         gresp = groq_client.chat.completions.create(
    #             model=os.getenv("OPENAI_API_MODEL", "llama-3.1-70b-versatile"),
    #             messages=ctx_messages,
    #             max_tokens=64,
    #             temperature=0.6,
    #         )
    #         assistant_msg = (gresp.choices[0].message.content or "").strip() or f"Can you tell me more about: {task['title']}?"
    #     except Exception:
    #         assistant_msg = f"Can you tell me more about: {task['title']}?"

    #     entries.append({"role": "assistant", "content": assistant_msg, "timestamp": now})
    #     task_collection.update_one(
    #         {"_id": task["_id"]},
    #         {"$set": {"status": "in_progress", "conv_count": conv_count, "last_prompt": assistant_msg}}
    #     )
    #     append_to_journal(entries)
    #     return {
    #         "message": "Started chat for task.",
    #         "journal_id": str(journal_id),
    #         "assistant_reply": assistant_msg,
    #         "completed": False
    #     }

    # ----- (D) Persist journal entries accumulated above -----
    append_to_journal(entries)

    # ----- (E) Compute completion + update task -----
    is_complete = conv_count >= MAX_TURNS
    task_updates: Dict[str, Any] = {
        "conv_count": conv_count,
        "journal_entry_id": str(journal_id),
    }

    if is_complete:
        task_updates.update({
            "completed": True,
            "status": "completed",
            "completed_at": now
        })
    else:
        if follow_up:
            task_updates.update({"status": "in_progress", "last_prompt": follow_up})
        else:
            task_updates.update({"status": "in_progress"})

    task_collection.update_one({"_id": task["_id"]}, {"$set": task_updates})

    return {
        "message": "Task completed!" if is_complete else "Task updated.",
        "journal_id": str(journal_id),
        "assistant_reply": follow_up or "Noted.",
        "completed": is_complete
    } 

# def complete_task(
#     username: str,
#     task_id: str,
#     task_content: Optional[str] = None,
#     image_url: Optional[str] = None
# ):
#     """
#     Conversation-style task completion that:
#       - Creates/fetches today's 'conversation' journal
#       - Logs user text (if any) and generates a Groq follow-up
#       - If a meal task and an image URL is provided: saves photo in journal and runs OpenAI Vision (gpt-4o-mini)
#       - Updates task conv_count/status and returns assistant reply
#     """
#     now = datetime.utcnow()
#     today = now.date()
#     start = datetime.combine(today, time.min)
#     end = datetime.combine(today, time.max)

#     # 1) Fetch the task
#     task = task_collection.find_one({"_id": ObjectId(task_id), "username": username})
#     if not task:
#         raise ValueError("Task not found")

#     # 2) Expiry guard
#     expires_at = task.get("expires_at")
#     if expires_at and now > expires_at:
#         raise ValueError("Task has expired and cannot be completed.")

#     # 3) Breathing: prevent multiple per day
#     if task["type"] == "breathing" and task_content is None:
#         already_logged = breathing_collection.find_one({
#             "username": username,
#             "timestamp": {"$gte": start, "$lte": end}
#         })
#         if already_logged:
#             return {
#                 "message": "Breathing already logged today.",
#                 "journal_id": None,
#                 "assistant_reply": "No further actions needed.",
#                 "completed": False
#             }

#     # 4) Find/create today's journal
#     journal = journals_collection.find_one({
#         "username": username,
#         "type": "conversation",
#         "timestamp": {"$gte": start, "$lte": end}
#     })

#     if not journal:
#         new_journal = {
#             "username": username,
#             "type": "conversation",
#             "timestamp": now,
#             "entries": [],
#             "mood": None
#         }
#         journal_id = journals_collection.insert_one(new_journal).inserted_id
#     else:
#         journal_id = journal["_id"]

#     def append_to_journal(entries_to_add):
#         if entries_to_add:
#             journals_collection.update_one(
#                 {"_id": journal_id},
#                 {"$push": {"entries": {"$each": entries_to_add}}}
#             )

#     # ✅ Always initialize
#     entries: list[dict[str, Any]] = []
#     follow_up: Optional[str] = None
#     conv_count = task.get("conv_count", 0)
#     MAX_TURNS = 4

#     # ----- (A) Handle user text (Groq follow-up) -----
#     if task_content:
#         try:
#             # Use your Groq text model for conversational follow-ups
#             groq_resp = client.chat.completions.create(
#                 model=os.getenv("GROQ_TEXT_MODEL", "llama-3.1-70b-versatile"),
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": (
#                             "You are a friendly journaling assistant. Based on the user's response, "
#                             "ask a short helpful question that continues the conversation."
#                         )
#                     },
#                     {"role": "user", "content": task_content}
#                 ],
#                 max_tokens=64,
#                 temperature=0.6,
#             )
#             follow_up = (groq_resp.choices[0].message.content or "").strip()
#             if not follow_up:
#                 follow_up = "Could you clarify what you meant by that? Can you provide more details?"
#         except Exception as e:
#             print("Follow-up GPT error:", e)
#             follow_up = "Sorry, I didn't catch that. Can you share more about your task?"

#         entries.extend([
#             {"role": "user", "content": task_content.strip(), "timestamp": now},
#             {"role": "assistant", "content": follow_up, "timestamp": now},
#         ])
#         conv_count += 1

#     # ----- (B) Handle meal image (OpenAI Vision) -----
#     if task["type"] in ["meal", "snack", "breakfast", "lunch", "dinner"] and image_url:
#         print("Processing meal image for vision analysis...===========")

#         try:
#     # Attach image to journal entry (user bubble)
#             entries.append({
#                 "role": "user",
#                 "content": "Here's my meal photo" if not task_content else "Attached meal photo",
#                 "timestamp": now,
#                 "attachments": [{"type": "image", "url": image_url, "timestamp": now}]
#             })

#             # IMPORTANT: call OpenAI (not Groq), with correct model + content array
#             vresp = openai_client.chat.completions.create(
#                 model="gpt-4o-mini",  # or "gpt-4o" if you have access
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": (
#                             "You are a nutrition assistant. Analyze the meal photo and respond with:\n"
#                             "- Estimated total calories\n"
#                             "- Protein/Carbs/Fats (grams)\n"
#                             "- Main items with portion estimates\n"
#                             "- One brief health tip or swap\n"
#                             "If unsure, clearly state assumptions."
#                         )
#                     },
#                     {
#                         "role": "user",
#                         "content": [
#                             {"type": "text", "text": "Please analyze this meal."},
#                             {"type": "image_url", "image_url": {"url": image_url}}
#                         ]
#                     }
#                 ],
#                 max_tokens=400,
#                 temperature=0.5
#             )

#             analysis = (vresp.choices[0].message.content or "").strip()
#             if not analysis:
#                 analysis = "I couldn't analyze the image right now."
#             entries.append({"role": "assistant", "content": analysis, "timestamp": now})

#             # ensure the API response is surfaced as assistant_reply
#             follow_up = (follow_up or analysis)

#         except Exception as e:
#             print("OpenAI vision error:", e)
#             entries.append({
#                 "role": "assistant",
#                 "content": "I couldn't process the image. Please send a valid HTTPS or base64 data URL.",
#                 "timestamp": now
#             })
        

#     # ----- (C) If neither text nor image, start chat -----
#     if not task_content and not image_url:
#         assistant_msg = f"Can you tell me more about: {task['title']}?"
#         entries.append({"role": "assistant", "content": assistant_msg, "timestamp": now})
#         task_collection.update_one(
#             {"_id": task["_id"]},
#             {"$set": {"status": "in_progress", "conv_count": conv_count, "last_prompt": assistant_msg}}
#         )
#         append_to_journal(entries)
#         return {
#             "message": "Started chat for task.",
#             "journal_id": str(journal_id),
#             "assistant_reply": assistant_msg,
#             "completed": False
#         }

#     # ----- (D) Persist journal entries accumulated above -----
#     append_to_journal(entries)

#     # ----- (E) Compute completion + update task -----
#     is_complete = conv_count >= MAX_TURNS
#     task_updates: dict[str, Any] = {
#         "conv_count": conv_count,
#         "journal_entry_id": str(journal_id),
#     }

#     if is_complete:
#         task_updates.update({
#             "completed": True,
#             "status": "completed",
#             "completed_at": now
#         })
#     else:
#         if follow_up:
#             task_updates.update({"status": "in_progress", "last_prompt": follow_up})
#         else:
#             task_updates.update({"status": "in_progress"})

#     task_collection.update_one({"_id": task["_id"]}, {"$set": task_updates})

#     return {
#         "message": "Task completed!" if is_complete else "Task updated.",
#         "journal_id": str(journal_id),
#         "assistant_reply": follow_up or "Noted.",
#         "completed": is_complete
#     }

 
 
 
 
 
 
 
 
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
 