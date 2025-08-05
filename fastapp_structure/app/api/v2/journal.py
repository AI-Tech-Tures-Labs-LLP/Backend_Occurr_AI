from fastapi import APIRouter, Depends, HTTPException, Body, Query
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Optional
from datetime import datetime,time
import random
import openai  # ✅ Required for GPT

# from app.api.v1.auth import decode_token
from app.api.auth.auth import decode_token
from app.db.journal_model import save_journal_entry, get_journals_by_user_month, get_journals_by_day, patch_journal,get_journals_by_range
from app.utils.journal_summary_generator import summarize_journals
from app.utils.journal_prompt_generator import generate_journal_prompt, get_time_of_day

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# ✅ Conversation state store
user_journal_sessions = {}

# ✅ Journal Chat Input
class JournalChatInput(BaseModel):
    user_input: Optional[str] = None


# @router.post("/journal/conversation")
# async def journal_conversation(
#     body: JournalChatInput,
#     token: str = Depends(oauth2_scheme)
# ):
#     valid, username = decode_token(token)
#     if not valid:
#         raise HTTPException(status_code=401, detail=username)

#     user_input = body.user_input or ""
#     time_of_day = get_time_of_day()

#     if username not in user_journal_sessions:
#         questions = [
#             generate_journal_prompt("food", username=username, time_of_day=time_of_day),
#             generate_journal_prompt("personal", username=username, time_of_day=time_of_day),
#             generate_journal_prompt("work_study", username=username, time_of_day=time_of_day),
#             generate_journal_prompt("sleep", username=username, time_of_day=time_of_day),
#         ]
#         user_journal_sessions[username] = {
#             "mood": None,
#             "answers": [],
#             "step": -1,
#             "questions": questions,
#             "extra_notes": [],
#             "last_question": ""
#         }
#         return {
#             "question": "👋 Hello! How are you feeling today? Select a mood: 😄 😊 😐 😢 😡 🥱",
#             "step": 0
#         }

#     session = user_journal_sessions[username]
#     step = session["step"]

#     if step == -1:
#         session["mood"] = user_input.strip() or "😐"
#         session["step"] = 0
#         return {"question": session["questions"][0], "step": 1}

#     if step < 4:
#         if user_input:
#             session["answers"].append(user_input)
#         session["step"] += 1
#         if session["step"] < 4:
#             return {"question": session["questions"][session["step"]], "step": session["step"] + 1}
#         else:
#             extra_q = generate_journal_prompt("extra", username=username, time_of_day=time_of_day)
#             session["last_question"] = extra_q
#             return {"question": extra_q, "step": session["step"]}

#     if step >= 4:
#         last_question = session.get("last_question", "")
#         if user_input and user_input.strip().lower() not in ["no", "nothing", "nope", "i'm good", "na","noo"]:
#             session["extra_notes"].append(f"Q: {last_question}\nA: {user_input.strip()}")
#         else:
#             entry = {
#                 "food_intake": session["answers"][0] if len(session["answers"]) > 0 else "",
#                 "personal": session["answers"][1] if len(session["answers"]) > 1 else "",
#                 "work_or_study": session["answers"][2] if len(session["answers"]) > 2 else "",
#                 "sleep": session["answers"][3] if len(session["answers"]) > 3 else "",
#                 "extra_note": " ".join(session["extra_notes"]),
#             }
#             save_journal_entry(
#                 username=username,
#                 entry_type="conversation",
#                 prompt="Conversational Journal",
#                 response=str(entry),
#                 tags=[],
#                 mood=session["mood"],
#                 extra_fields=entry
#             )
#             user_journal_sessions.pop(username, None)
#             return {"message": "✅ Thank you for journaling today! Your notes have been saved."}

#         next_question = generate_journal_prompt("extra", username=username, time_of_day=time_of_day)
#         session["last_question"] = next_question
#         session["step"] += 1
#         return {"question": next_question, "step": session["step"]}

@router.post("/journal/conversation")
async def journal_conversation(
    body: JournalChatInput,
    token: str = Depends(oauth2_scheme)
):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)

    user_input = body.user_input or ""
    time_of_day = get_time_of_day()

    if username not in user_journal_sessions:
        # Async generation of prompts one by one
        questions = [
            await generate_journal_prompt("food", username=username, time_of_day=time_of_day),
            await generate_journal_prompt("personal", username=username, time_of_day=time_of_day),
            await generate_journal_prompt("work_study", username=username, time_of_day=time_of_day),
            await generate_journal_prompt("sleep", username=username, time_of_day=time_of_day),
        ]
        user_journal_sessions[username] = {
            "mood": None,
            "answers": [],
            "step": -1,
            "questions": questions,
            "extra_notes": [],
            "last_question": ""
        }
        return {
            "question": "👋 Hello! How are you feeling today? Select a mood: 😄 😊 😐 😢 😡 🥱",
            "step": 0
        }

    session = user_journal_sessions[username]
    step = session["step"]

    if step == -1:
        session["mood"] = user_input.strip() or "😐"
        session["step"] = 0
        return {"question": session["questions"][0], "step": 1}

    if step < 4:
        if user_input:
            session["answers"].append(user_input)
        session["step"] += 1
        if session["step"] < 4:
            return {"question": session["questions"][session["step"]], "step": session["step"] + 1}
        else:
            extra_q = await generate_journal_prompt("extra", username=username, time_of_day=time_of_day)
            session["last_question"] = extra_q
            return {"question": extra_q, "step": session["step"]}

    if step >= 4:
        last_question = session.get("last_question", "")
        if user_input and user_input.strip().lower() not in ["no", "nothing", "nope", "i'm good", "na", "noo"]:
            session["extra_notes"].append(f"Q: {last_question}\nA: {user_input.strip()}")
        else:
            entry = {
                "food_intake": session["answers"][0] if len(session["answers"]) > 0 else "",
                "personal": session["answers"][1] if len(session["answers"]) > 1 else "",
                "work_or_study": session["answers"][2] if len(session["answers"]) > 2 else "",
                "sleep": session["answers"][3] if len(session["answers"]) > 3 else "",
                "extra_note": " ".join(session["extra_notes"]),
            }
            # Optional: await this if you convert to async later
            save_journal_entry(
                username=username,
                entry_type="conversation",
                prompt="Conversational Journal",
                response=str(entry),
                tags=[],
                mood=session["mood"],
                extra_fields=entry
            )
            user_journal_sessions.pop(username, None)
            return {"message": "✅ Thank you for journaling today! Your notes have been saved."}

        next_question = await generate_journal_prompt("extra", username=username, time_of_day=time_of_day)
        session["last_question"] = next_question
        session["step"] += 1
        return {"question": next_question, "step": session["step"]}


# ✅ Patch Journal
class PatchConversationJournalRequest(BaseModel):
    journal_id: str
    food_intake: str = None
    personal: str = None
    work_or_study: str = None
    sleep: str = None
    extra_note: str = None

@router.patch("/journal/conversation")
def patch_conversation_journal(req: PatchConversationJournalRequest, token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)

    update_fields = {k: v for k, v in req.dict().items() if k != "journal_id" and v is not None}
    if not update_fields:
        raise HTTPException(status_code=400, detail="No fields to update.")

    patch_journal(req.journal_id, update_fields)
    return {"message": "Conversational journal updated successfully"}

# ✅ Monthly Summary
@router.get("/journal/summary/month")
def get_monthly_summary(token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)

    entries = get_journals_by_user_month(username)
    for entry in entries:
        entry["_id"] = str(entry["_id"])

    responses = [e["response"] for e in entries if e.get("response")]
    summary = summarize_journals(responses)

    return {
        "count": len(responses),
        "summary": summary,
        "entries": entries
    }

# ✅ Day Summary
@router.get("/journal/by-day")
def get_journals_for_day(
    date: str = Query(..., example="2025-05-20"),
    token: str = Depends(oauth2_scheme)
):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)

    entries = get_journals_by_day(username, date)
    for entry in entries:
        entry["_id"] = str(entry["_id"])

    return {
        "date": date,
        "count": len(entries),
        "entries": entries
    }
 

@router.get("/journal/by-range")
def get_journals_for_range(
    start_date: str = Query(..., example="2025-05-20"),
    end_date: str = Query(..., example="2025-05-22"),
    token: str = Depends(oauth2_scheme)
):
    # Validate the token and decode to get the username
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)

    # Ensure that the start and end dates are valid
    try:
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Please use YYYY-MM-DD.")

    # Ensure full day range (00:00 to 23:59) for both start and end dates
    start_of_day = datetime.combine(start_date_obj, time.min)  # 00:00:00
    end_of_day = datetime.combine(end_date_obj, time.max)  # 23:59:59.999999

    print(f"start_of_day: {start_of_day}")
    print(f"end_of_day: {end_of_day}")

    # Query the journals within the specified range
    entries = get_journals_by_range(username, start_of_day, end_of_day)
    
    # Ensure entries is not None
    if not entries:
        entries = []

    # Ensure the entries' _id fields are returned as strings
    for entry in entries:
        entry["_id"] = str(entry["_id"])

    return {
        "start_date": start_date,
        "end_date": end_date,
        "entries": entries
    }

# Helper function to retrieve journals within a range (update this based on your database)
