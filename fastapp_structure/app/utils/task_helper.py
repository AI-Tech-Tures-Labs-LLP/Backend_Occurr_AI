from __future__ import annotations
from fastapi import UploadFile
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
from datetime import datetime, date, time, timezone
import boto3

import anyio  # comes via Starlette/AnyIO
from anyio import from_thread

# from vision_model import 
import os, io, base64,uuid,json,torch,requests,anyio,mimetypes  
import streamlit as st
from PIL import Image
from openai import OpenAI

# --- Local vision (captioning) ---
from transformers import pipeline


from io import BytesIO
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone, time
from bson import ObjectId
from PIL import Image

import torch


from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException

# ---- Project-specific imports ----

 

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
            return "your daily meditation"
        if "exercise" in key or "workout" in key:
            return "Log your exercise"
        if "breathing" in key:
            return "your breathing exercises"
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
def get_groq_client() -> Groq:
    api_key = os.getenv("OPENAI_API_KEY") 
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY (or OPENAI_API_KEY)")
    # DeepSeek exposes OpenAI-compatible API
    return client(api_key=api_key)


def ds_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY") 
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY (or OPENAI_API_KEY)")
    # DeepSeek exposes OpenAI-compatible API
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")



NUTRITION_SYSTEM_PROMPT = (
  """You are an expert nutritionist. Given a short meal description, estimate total calories.
List each item with an estimated calorie count in this exact format:
1. Item 1 - N calories
2. Item 2 - N calories
...
Add a brief note about assumptions/portion sizes."""
)
USER_TEMPLATE = (
    "Image caption: {caption}\n"
    "{extra}\n"
    "Return a short analysis with bullet points for items, a single total kcal estimate, and macros."
)

def analyze_meal_with_openai(caption: str, extra: str = "", model: str = None) -> str:
    client = ds_client()
    prompt = USER_TEMPLATE.format(caption=caption.strip(), extra=(extra.strip() or ""))
    resp = client.chat.completions.create(
        model=model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        temperature=0.2,
        messages=[
            {"role": "system", "content": NUTRITION_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )
    return (resp.choices[0].message.content or "").strip()

# ---------------- HF Captioning (BLIP) ----------------
_DATA_URL_RE = re.compile(r"^data:(?P<mime>[\w/+.-]+);base64,(?P<b64>.+)$", re.I)
_CAPTIONER = None  # cached pipeline

def to_pil_image(image_input: Any) -> Image.Image:
    """Coerce UploadFile / bytes / data-url / http(s) url / local path / PIL.Image -> PIL.Image (RGB)."""
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    # Starlette UploadFile or file-like with .file
    if hasattr(image_input, "file"):
        try:
            image_input.file.seek(0)
        except Exception:
            pass
        return Image.open(image_input.file).convert("RGB")

    # Raw bytes
    if isinstance(image_input, (bytes, bytearray)):
        return Image.open(BytesIO(image_input)).convert("RGB")

    # String inputs
    if isinstance(image_input, str):
        m = _DATA_URL_RE.match(image_input)
        if m:
            raw = base64.b64decode(m.group("b64"))
            return Image.open(BytesIO(raw)).convert("RGB")
        if image_input.startswith(("http://", "https://")):
            r = requests.get(image_input, timeout=30)
            r.raise_for_status()
            return Image.open(BytesIO(r.content)).convert("RGB")
        # assume filesystem path
        return Image.open(image_input).convert("RGB")

    raise ValueError("Unsupported image input type; provide URL/path/bytes/base64/PIL/UploadFile")

def get_captioner():
    global _CAPTIONER
    if _CAPTIONER is None:
        device = 0 if torch.cuda.is_available() else -1
        _CAPTIONER = pipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-base",
            device=device,
        )
    return _CAPTIONER

def caption_image(image_input: Any) -> str:
    img = to_pil_image(image_input)
    cap = get_captioner()(img)[0]["generated_text"].strip()
    return cap

def analyze_meal(image_input: Any, *, extra: str = "") -> str:
    """
    image_input can be UploadFile / URL / data-URL / bytes / path / PIL.Image
    Returns the nutrition analysis string (DeepSeek).
    """
    cap = caption_image(image_input)
    return analyze_meal_with_openai(
        caption=cap,
        extra=extra or "Estimate total calories and macros. If uncertain, say so briefly.",
    )

# ---------------- S3 Upload (async) ----------------
async def upload_image_to_s3(
    image_file: UploadFile,
    bucket: str,
    key_prefix: str = "journal",
    region: str | None = None,
    public_base: str | None = None,
    presign_ttl: int = 7 * 24 * 3600,
) -> str:
    """
    Reads UploadFile (await .read()), uploads bytes to S3 without ACLs, and returns:
      - clean public URL if public_base is provided (assumes bucket policy/CF handles public read)
      - else presigned GET URL valid for presign_ttl seconds
    """
    if not image_file:
        raise ValueError("image_file is required")

    data = await image_file.read()
    mime = (image_file.content_type or "application/octet-stream").split(";")[0].strip()
    ext = mimetypes.guess_extension(mime) or ".jpg"
    key = f"{key_prefix}/{datetime.now(timezone.utc).strftime('%Y/%m/%d')}/{uuid.uuid4().hex}{ext}"

    s3 = boto3.client("s3", region_name=region)

    def _put():
        s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=mime)

    # boto3 is blocking -> run in thread
    await anyio.to_thread.run_sync(_put)

    if public_base:
        return f"{public_base.rstrip('/')}/{key}"

    # Presigned URL
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=presign_ttl,
    )


JOURNAL_FOLLOWUP_SYS = (
    "You are a friendly journaling assistant.\n"
    "Use the full conversation context below to ask ONE short follow-up question "
    "that genuinely advances the topic. Refer to specific details the user mentioned. "
    "Avoid repeating questions already asked. Keep it under 25 words."
)

# ---- helpers (paste once in your module) ----


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



def _ensure_datetime_utc(v: Any) -> datetime:
    """
    Coerce a value into a timezone-aware UTC datetime.
    - datetime with tz -> convert to UTC
    - naive datetime   -> assume UTC
    - date             -> start of day UTC
    """
    if isinstance(v, datetime):
        return v.astimezone(timezone.utc) if v.tzinfo else v.replace(tzinfo=timezone.utc)
    if isinstance(v, date):
        return datetime.combine(v, time.min).replace(tzinfo=timezone.utc)
    # fallback to "now" if missing/invalid
    return datetime.now(tz=timezone.utc)

def _day_bounds_utc(dt: datetime) -> tuple[datetime, datetime]:
    """
    Given a UTC datetime, return start/end of that (UTC) day as aware datetimes.
    """
    dt_utc = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    start = datetime(dt_utc.year, dt_utc.month, dt_utc.day, 0, 0, 0, tzinfo=timezone.utc)
    end   = datetime(dt_utc.year, dt_utc.month, dt_utc.day, 23, 59, 59, 999999, tzinfo=timezone.utc)
    return start, end

# ---- your function (S3 + Vision integrated + context-aware Groq follow-up) ----



def complete_task(
    username: str,
    task_id: str,
    task_content: Optional[str] = None,
    image_file: Optional[Any] = None,
):
    """
    Conversation-style task completion that:
      - Creates/fetches today's 'conversation' journal
      - Logs user text (if any) with task_id and generates a context-aware follow-up
      - If a meal task and an image is provided: uploads photo to S3, saves S3 URL in journal,
        and runs image analysis
      - Updates task conv_count/status and returns assistant reply

    NEW:
      - Persist task_id on every journal entry
      - When generating follow-up, prefer context from journal entries that match this task_id;
        if none exist, fallback to all entries in today's journal.
    """
    # 1) Fetch the task
    task = task_collection.find_one({"_id": ObjectId(task_id), "username": username})
    if not task:
        raise ValueError("Task not found")

    task_id_str = str(task["_id"])

    # Normalize created_at to aware UTC datetime and compute day bounds
    created_at = _ensure_datetime_utc(task.get("created_at") or datetime.now(tz=timezone.utc))
    now = created_at  # keep as datetime (NOT .date())
    start, end = _day_bounds_utc(now)

    # 2) Breathing: prevent multiple per day
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

    # 3) Find/create today's journal
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

    # Initialize
    entries: List[Dict[str, Any]] = []
    follow_up: Optional[str] = None
    conv_count = task.get("conv_count", 0)
    MAX_TURNS = 4

    meal_types = {"meal", "snack", "breakfast", "lunch", "dinner"}

    # ----- (A) Handle image path (S3 + analysis) -----
    if ((task.get("type") in meal_types) and image_file and task_content) or \
       ((task.get("type") not in meal_types) and image_file):
        print("Processing meal image for vision analysis...===========")
        try:
            # 1) Upload to S3 and get a URL
            aws_region = os.getenv("AWS_REGION", "ap-south-1")
            s3_bucket  = os.getenv("AWS_S3_BUCKET")
            if not s3_bucket:
                raise RuntimeError("S3_BUCKET env var is required")
            s3_public_base = os.getenv("S3_PUBLIC_BASE")  # optional

            from anyio import from_thread
            s3_url: str = from_thread.run(
                lambda: upload_image_to_s3(
                    image_file=image_file,
                    bucket=s3_bucket,
                    key_prefix="journal",
                    region=aws_region,
                    public_base=s3_public_base,
                )
            )
            if not isinstance(s3_url, str) or not s3_url:
                raise RuntimeError("Uploader did not return URL string")

            # 2) Append user entry with image (persist task_id)
            try:
                image_file.file.seek(0)
            except Exception:
                pass

            user_caption = "Attached meal photo" if task_content else "Here's my meal photo"
            entries.append({
                "role": "user",
                "content": user_caption if task.get("type") in meal_types else (task_content or user_caption),
                "timestamp": now,
                "task_id": task_id_str,  # NEW
                "attachments": [{"type": "image", "url": s3_url, "timestamp": now}],
            })

            # 3) Analyze image (optionally include extra text)
            analysis = analyze_meal(
                image_input=image_file,
                extra=f", user_input {task_content} ,Analyze this meal for items, total kcal, macros (protein, carbs, fat), and one health tip .",
            )
            if not analysis:
                analysis = "I couldn't analyze the image right now."

            # 4) Append assistant analysis (persist task_id)
            entries.append({
                "role": "assistant",
                "content": analysis,
                "timestamp": now,
                "task_id": task_id_str,  # NEW
            })
            follow_up = analysis
            conv_count += 1

        except Exception as e:
            print("Meal image pipeline error:", e)
            entries.append({
                "role": "assistant",
                "content": "I couldn't process the image. Please send a valid HTTPS or data URL.",
                "timestamp": now,
                "task_id": task_id_str,  # NEW
            })
            follow_up = "I couldn't process the image. Please send a valid HTTPS or data URL."

    # ----- (B) Handle text-only flow (context-aware follow-up) -----
    if task_content and not image_file:
        print("Generating follow-up...===========")
        try:
            journal_doc = journals_collection.find_one({"_id": journal_id}, {"entries": 1}) or {"entries": []}
            all_entries = journal_doc.get("entries", [])

            # NEW: restrict context to this task_id if any entries exist for it
            same_task_entries = [e for e in all_entries if e.get("task_id") == task_id_str]
            context_entries = same_task_entries if same_task_entries else all_entries

            ctx_messages = build_context_messages(
                {"entries": context_entries},
                task_content,
                max_turns=12
            )

            groq_resp = client.chat.completions.create(
                model=os.getenv("OPENAI_API_MODEL"),
                messages=ctx_messages,
                max_tokens=64,
                temperature=0,
            )
            follow_up = (groq_resp.choices[0].message.content or "").strip()
            if not follow_up:
                follow_up = "Could you clarify that a bit more?"
        except Exception as e:
            print("Follow-up error:", e)
            follow_up = "Thanks! Anything else you'd like to add?"

        # Persist user + assistant turns with task_id
        entries.extend([
            {"role": "user", "content": task_content.strip(), "timestamp": now, "task_id": task_id_str},
            {"role": "assistant", "content": follow_up, "timestamp": now, "task_id": task_id_str},
        ])
        conv_count += 1

    # ----- (C) If neither text nor image, start chat (context-aware opener) -----
    if not task_content and not image_file:
        assistant_msg = f"Can you tell me more about: {task['title']}?"
        # Persist assistant prompt with task_id
        entries.append({
            "role": "assistant",
            "content": assistant_msg,
            "timestamp": now,
            "task_id": task_id_str,  # NEW
        })
        append_to_journal(entries)
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

    # ----- (D) Persist accumulated journal entries -----
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
            task_updates.update({
                "status": "in_progress",
                "last_prompt": follow_up,
                "user_reply": task_content
            })
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
#     image_file: Optional[Any] = None,
# ):
#     """
#     Conversation-style task completion that:
#       - Creates/fetches today's 'conversation' journal
#       - Logs user text (if any) and generates a context-aware Groq follow-up
#       - If a meal task and an image URL is provided: uploads photo to S3, saves S3 URL in journal,
#         and runs OpenAI Vision (gpt-4o-mini) on that S3 URL
#       - Updates task conv_count/status and returns assistant reply
#     """
#     # now = datetime.utcnow()
#     # today = now.date()
#     # start = datetime.combine(today, time.min)
#     # end = datetime.combine(today, time.max)

#     # 1) Fetch the task
#     task = task_collection.find_one({"_id": ObjectId(task_id), "username": username})
#     if not task:
#         raise ValueError("Task not found")
    

#      # Normalize created_at to aware UTC datetime
#     created_at = _ensure_datetime_utc(task.get("created_at") or datetime.now(tz=timezone.utc))
#     now = created_at  # keep as datetime (NOT .date())
#     start, end = _day_bounds_utc(now)



#     # now = task["created_at"].date()
#     # today = now
#     # start = datetime.combine(today, time.min)
#     # end = datetime.combine(today, time.max)


#     # 2) Expiry guard
#     # expires_at = task.get("expires_at")
#     # if expires_at and now > expires_at:
#     #     return("Task has expired and cannot be completed.")

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
#         journal = {**new_journal, "_id": journal_id}
#     else:
#         journal_id = journal["_id"]

#     def append_to_journal(entries_to_add: List[Dict[str, Any]]):
#         if entries_to_add:
#             journals_collection.update_one(
#                 {"_id": journal_id},
#                 {"$push": {"entries": {"$each": entries_to_add}}}
#             )

#     # ✅ Always initialize
#     entries: List[Dict[str, Any]] = []
#     follow_up: Optional[str] = None
#     conv_count = task.get("conv_count", 0)
#     MAX_TURNS = 4

#     # ----- (A) Handle user text (context-aware Groq follow-up) -----
#     if (task.get("type") in {"meal", "snack", "breakfast", "lunch", "dinner"} and image_file and task_content) or (task.get("type") not in {"meal", "snack", "breakfast", "lunch", "dinner"} and image_file ):
#             print("Processing meal image for vision analysis...===========")
#             try:
#                 # 1) Upload to S3 and get a URL we can store/call withS
#                 aws_region = os.getenv("AWS_REGION", "ap-south-1")
#                 s3_bucket  = os.getenv("AWS_S3_BUCKET")
#                 if not s3_bucket:
#                     raise RuntimeError("S3_BUCKET env var is required")
#                 s3_public_base = os.getenv("S3_PUBLIC_BASE")  # optional

#                 # Bridge to async uploader from sync thread; pass kwargs via lambda
#                 from anyio import from_thread
#                 s3_url: str = from_thread.run(
#                     lambda: upload_image_to_s3(
#                         image_file=image_file,
#                         bucket=s3_bucket,
#                         key_prefix="journal",
#                         region=aws_region,
#                         public_base=s3_public_base,
#                     )
#                 )
#                 if not isinstance(s3_url, str) or not s3_url:
#                     raise RuntimeError("Uploader did not return URL string")

#                 # 2) Attach image to journal entry (user bubble)
#                 entries.append({
#                     "role": "user",
#                     "content": "Here's my meal photo" if not task_content else "Attached meal photo",
#                     "timestamp": now,
#                     "attachments": [{"type": "image", "url": s3_url, "timestamp": now}],
#                 })

#                 # 3) Caption+analyze using local file (faster; no re-download). Reset pointer in case it moved.
#                 try:
#                     image_file.file.seek(0)
#                 except Exception:
#                     pass

#                 analysis = analyze_meal(
#                     image_input=image_file,
#                     extra=f", user_input {task_content} ,Analyze this meal for items, total kcal, macros (protein, carbs, fat), and one health tip .",
#                 )
#                 if not analysis:
#                     analysis = "I couldn't analyze the image right now."

#                 entries.append({"role": "assistant", "content": analysis, "timestamp": now})
#                 follow_up = follow_up or analysis
#                 conv_count += 1

#             except Exception as e:
#                 print("Meal image pipeline error:", e)
#                 entries.append({
#                     "role": "assistant",
#                     "content": "I couldn't process the image. Please send a valid HTTPS or data URL.",
#                     "timestamp": now
#                 })
#                 follow_up = follow_up or "I couldn't process the image. Please send a valid HTTPS or data URL."


# # ----- (A) Handle user text (context-aware Groq follow-up) -----
#     if task_content and not image_file:
#         print("Generating Groq follow-up...===========")
#         try:
#             journal_doc = journals_collection.find_one({"_id": journal_id}, {"entries": 1}) or {"entries": []}
#             ctx_messages = build_context_messages(journal_doc, task_content, max_turns=12)

#             groq_resp = client.chat.completions.create(
#                 model=os.getenv("OPENAI_API_MODEL"),
#                 messages=ctx_messages,
#                 max_tokens=64,
#                 temperature=0,
#             )
#             follow_up = (groq_resp.choices[0].message.content or "").strip()
#             print("Generated Groq follow-up:", follow_up)
#             if not follow_up:
#                 follow_up = "Could you clarify that a bit more?"
#         except Exception as e:
#             print("Groq follow-up error:", e)
#             follow_up = "Thanks! Anything else you'd like to add?"

#         entries.extend([
#             {"role": "user", "content": task_content.strip(), "timestamp": now},
#             {"role": "assistant", "content": follow_up, "timestamp": now},
#         ])
#         conv_count += 1

#     # ----- (B) Handle meal image (S3 + OpenAI Vision) -----
    
#     # ----- (C) If neither text nor image, start chat (context-aware opener) -----

#     if not task_content and not image_file:
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
#     task_updates: Dict[str, Any] = {
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
#             task_updates.update({"status": "in_progress", "last_prompt": follow_up,"user_reply": task_content})
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
 