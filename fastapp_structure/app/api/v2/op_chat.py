

# -*- coding: utf-8 -*-
"""
SECONDARY INTENT DETECTION + FOLLOW-UP CONTEXT MEMORY
- Remembers last topic, collection, and date range per conversation
- Injects that context into the next turn for classification & Mongo routing
- Uses previous QA pair to resolve pronouns/implicit references
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from functools import lru_cache
from dotenv import load_dotenv
from groq import Groq
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os, json, re, time

# ---------------- External app imports ----------------
# (these are expected to exist in your codebase)
from app.db.database import users_collection, conversations_collection
from app.db.health_data_model import alert_collection, health_data_collection
from app.db.journal_model import journals_collection
from app.api.auth.auth import decode_token
from app.utils.optimized_code_rag import query_documents, prewarm_indexes
from app.core.advance_chatbot import apply_personality
from app.api.v2.adv_chat import *  # get_or_create_conversation, save_message, get_recent_history, extract_date_context, generate_ai_mongo_query_with_fallback, fetch_personal_data, build_comprehensive_context
from app.utils.optimized_code_rag import generate_answer_from_context,format_mongo_answer_llm
# ---------------- Setup ----------------
load_dotenv()

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

FAISS_FOLDER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "faiss_indexes")
)

client = Groq(
    api_key=os.getenv("OPENAI_API_KEY")
    # base_url=os.getenv("OPENAI_API_BASE_URL")
)
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL"),
    model=os.getenv("OPENAI_API_MODEL")
)

hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# ---------------- In-memory stores ----------------
conversation_store: Dict[str, List[Dict[str, Any]]] = {}
loaded_indexes: Dict[str, Any] = {}
pending_journal_confirmations: Dict[str, Dict[str, Any]] = {}

# NEW: Lightweight per-conversation memory for topic/collection/date
conversation_context = defaultdict(lambda: {
    "topic": None,          # e.g., "heart_rate"
    "collection": None,     # e.g., "health_data_collection"
    "date_ctx": None        # {"date": "..."} or {"start_date": "...", "end_date": "..."}
})

# ---------------- Models ----------------
class TagTimeRange(BaseModel):
    tag: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class ChatRequest(BaseModel):
    question: str
    tags: Optional[List[TagTimeRange]] = None
    context: Optional[str] = None
    date: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    history: List[Dict[str, Any]]
    conversation_id: str
    query_type: str
    data_sources: List[str]

# ---------------- Constants ----------------
TAG_COLLECTION_MAP = {
    "heart_rate": "health_data_collection",
    "heartrate": "health_data_collection",
    "steps": "health_data_collection",
    "sleep": "health_data_collection",
    "spo2": "health_data_collection",
    "calories": "health_data_collection",
    "food_intake": "journals_collection",
    "meditation": "journals_collection",
    "work_or_study": "journals_collection",
    "personal": "journals_collection",
    "mood": "journals_collection",
    "extra_note": "journals_collection",
    "alert": "alert_collection",
    "notification": "notifications_collection",
    "reflection": "journals_collection",
    "breathing": "breathing_collection",
    "task": "task_collection"
}

DEFAULT_SYSTEM_PROMPT = """
You are an intelligent health and lifestyle assistant.

Your job is to understand the user's message and determine its purpose.
Based on the message, take the appropriate action:
──────────────────────────────────────────────
1. 🔍 INTENT DETECTION:
Classify the user's message as one of the following:

- "mongo_query": A question about the user's own health/activity data (e.g., "What was my heart rate yesterday?")
- "knowledge_query": A general health-related question (e.g., "What is a healthy sleep duration?")
- "journal_entry": The user is logging something like food, sleep, mood, personal reflections, work, or meditation.
- "alert_response": A short response to a health alert or reminder (e.g., "Yes", "Done", "I just ate").
- "greeting": A greeting (e.g., "hi", "good morning").
- "smalltalk": Casual conversation not meant to be logged or analyzed.

──────────────────────────────────────────────
2. 📦 COLLECTION DETECTION for "mongo_query":
If the intent is "mongo_query", also identify which MongoDB collection is relevant.
Infer the collection based on the topic of the question and include it in the output as the "collection" field.

Use the following guide to map questions to collections:

- "health_data_collection" → heart rate, steps, sleep, calories, spo2
- "journals_collection" → journal, mood, dream, food, meditation, reflection
- "alert_collection" → alert, health warning, abnormal reading
- "notifications" → notification, reminder, message
- "tasks" → task, to-do, pending item, checklist
- "breathing_exercises" → breathing, inhale, exhale, session
- "conversations_collection" → chat history, past messages
- "users_collection" → profile, email, username, account info

Include this as:
"collection": "<collection_name>"
👉 For journal or health data queries that involve a specific **date only**, use this MongoDB date filter to **ignore time**:
{
  "$expr": {
    "$eq": [
      { "$dateToString": { "format": "%Y-%m-%d", "date": "$timestamp" } },
      "2025-07-23"
    ]
  }
}

──────────────────────────────────────────────
3. 📓 JOURNAL ENTRY DETECTION:
If the message logs a real-life activity, classify it as a "journal_entry".
Further tag the category as one of:
- "meditation"
- "food_intake"
- "sleep"
- "work_or_study"
- "personal"
- "extra_note"

Examples:
- "I meditated for 10 minutes" → journal_entry, meditation
- "Had eggs for breakfast" → journal_entry, food_intake
- "I studied math today" → journal_entry, work_or_study

──────────────────────────────────────────────
4. 🔁 ALERT RESPONSE:
If the message is a simple response like "Yes", "No", "Just did", and the assistant previously asked a reminder or warning — treat it as a response to that prompt.
Flag it as "alert_response", and connect it to the recent alert.

──────────────────────────────────────────────
5. 🧠 QUESTION HANDLING:
If the user asks a question:

- If it relates to their own data (e.g., "What was my heart rate yesterday?", "What is the summary of my journal today?"), it’s a "mongo_query".
- If it's general health advice (e.g., "What is a healthy heart rate?"), it’s a "knowledge_query".
- If the user asks for a **summary of their journal entries**, respond with a formatted and human-friendly summary using the provided journal content.
  Example format:
  {
    "reply": "📝 **Today's Journal Summary**:\\n• Food: eggs and toast\\n• Sleep: 7.5 hours\\n• Mood: relaxed",
    "intent": "mongo_query",
    "collection": "journals_collection"
  }

──────────────────────────────────────────────
6. 💬 RESPONSE STYLE:
- Always be conversational, warm, supportive, and helpful.
- For journal entries or alert responses, confirm the entry was saved.
- For questions, give helpful and specific answers.
- For smalltalk, keep it friendly and open-ended.

──────────────────────────────────────────────
7. 📤 OUTPUT FORMAT:
Always return a JSON string like:
{
  "reply": "Got it! I logged your meditation.",
  "intent": "journal_entry",
  "tag": "meditation"
}

If intent is "mongo_query", also include:
"collection": "<target_collection>"
"""

# ---------------- Follow-up & topic detection helpers ----------------
YES_TOKENS = {"yes","yeah","yup","ok","okay","sure","please","yes please","do it","go ahead","confirm","y","k"}
OFFER_PATTERNS = re.compile(r"(would you like me to|should i|do you want me to|want me to|shall i|i can)\s+(check|look|pull|fetch|get)", re.IGNORECASE)

def is_yes_like(text: str) -> bool:
    return any(tok in text.strip().lower() for tok in YES_TOKENS)

def assistant_offered_check(text: str) -> bool:
    return bool(OFFER_PATTERNS.search(text or ""))

def detect_collection_from_query(query: str) -> str:
    query = query.lower()
    collection_keywords = {
        "health_data_collection": ["steps","heart rate","heartrate","pulse","sleep","calories","spo2","oxygen","activity"],
        "journals_collection": ["journal","mood","reflection","dream","note","food","meal","meditation","entry","log"],
        "alert_collection": ["alert","abnormal","warning","trigger","anomaly"],
        "notifications": ["notification","reminder","ping","unread","messages"],
        "tasks": ["task","todo","checklist","pending","completed","due"],
        "breathing_exercises": ["breathing","inhale","exhale","hold","breathwork","session"],
        "conversations_collection": ["chat","conversation","message history"],
        "users_collection": ["profile","email","account","username","user data"]
    }
    for coll, kws in collection_keywords.items():
        if any(kw in query for kw in kws):
            return coll
    return "journals_collection"

# Topic maps
TOPIC_KEYWORDS = {
    "heart_rate": ["heart rate","heartrate","pulse","bpm"],
    "steps": ["steps","walk","walking","pedometer"],
    "sleep": ["sleep","nap","asleep","bed"],
    "spo2": ["spo2","oxygen","o2","blood oxygen"],
    "calories": ["calorie","calories","kcal","energy"],
    # journals:
    "food_intake": ["food","ate","meal","breakfast","lunch","dinner","snack"],
    "meditation": ["meditat","breathe","breathwork","mindful","zen"],
    "work_or_study": ["work","study","office","project","meeting"],
    "mood": ["feel","mood","happy","sad","anxious","angry","excited"],
    "sleep_journal": ["sleep log","slept"],
}
TOPIC_COLLECTION = {
    "heart_rate": "health_data_collection",
    "heartrate": "health_data_collection",
    "steps": "health_data_collection",
    "sleep": "health_data_collection",
    "spo2": "health_data_collection",
    "calories": "health_data_collection",
    "food_intake": "journals_collection",
    "meditation": "journals_collection",
    "work_or_study": "journals_collection",
    "mood": "journals_collection",
    "sleep_journal": "journals_collection",
}
PRONOUN_FOLLOWUP = re.compile(
    r"^(when|where|how|what|was|is|my|did|does|do|then|and|also|again|so)\b|^(when|where|how|what)\b.*(it|that|this)$",
    re.IGNORECASE
)

def detect_topic(text: str) -> Optional[str]:
    t = text.lower()
    for topic, kws in TOPIC_KEYWORDS.items():
        if any(kw in t for kw in kws):
            return topic
    return None

def update_conversation_context(convo_id: str, *, topic: Optional[str]=None,
                                collection: Optional[str]=None,
                                date_ctx: Optional[dict]=None) -> None:
    st = conversation_context[convo_id]
    if topic:
        st["topic"] = topic
    if collection:
        st["collection"] = collection
    if date_ctx:
        st["date_ctx"] = date_ctx

def ctx_collection_fallback(convo_id: str) -> Optional[str]:
    st = conversation_context[convo_id]
    if st["collection"]:
        return st["collection"]
    if st["topic"] and st["topic"] in TOPIC_COLLECTION:
        return TOPIC_COLLECTION[st["topic"]]
    return None

def carry_forward_question(raw_q: str, convo_id: str) -> str:
    """
    If the user asks a follow-up like 'When was it too high?'
    inject last topic hints + last time window to disambiguate.
    """
    state = conversation_context[convo_id]
    looks_followup = bool(PRONOUN_FOLLOWUP.search(raw_q.strip()))
    found_topic = detect_topic(raw_q)

    if looks_followup or not found_topic:
        parts = []
        if state["topic"]:
            parts.append(f"(Previous topic: {state['topic'].replace('_',' ')})")
        if state["date_ctx"]:
            if "date" in state["date_ctx"]:
                parts.append(f"(date={state['date_ctx']['date']})")
            else:
                sd = state["date_ctx"].get("start_date"); ed = state["date_ctx"].get("end_date")
                if sd or ed:
                    parts.append(f"(range {sd or '?'} → {ed or '?'})")
        if parts:
            return f"{' '.join(parts)}\n{raw_q}"
    return raw_q

# ---------------- Prev-QA helpers ----------------
def get_last_qa_pair(history: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    last_answer = None; last_question = None
    for m in reversed(history or []):
        if m.get("role") == "assistant" and last_answer is None:
            last_answer = m.get("content","")
        elif m.get("role") == "user" and last_answer is not None:
            last_question = m.get("content",""); break
    if last_question and last_answer:
        return {"question": last_question.strip()[:800], "answer": last_answer.strip()[:800]}
    return None

def with_prev_qa(q: str, history: List[Dict[str, Any]]) -> str:
    prev = get_last_qa_pair(history)
    if not prev:
        return q
    return (
        f"{q}\n\n[Context for disambiguation]\n"
        f"Previous Question: {prev['question']}\n"
        f"Previous Answer: {prev['answer']}\n"
        f"[End context]"
    )

# ---------------- FAISS helpers ----------------
def _has_faiss(dirpath: str) -> bool:
    return os.path.exists(os.path.join(dirpath,"index.faiss")) and os.path.exists(os.path.join(dirpath,"index.pkl"))

def _safe_read(path: str) -> str:
    try:
        with open(path,"r",encoding="utf-8",errors="ignore") as f:
            return f.read().strip()
    except Exception:
        return ""

@lru_cache(maxsize=1)
def _index_descriptions() -> dict:
    descs={}; base=FAISS_FOLDER_PATH
    if not os.path.isdir(base):
        return descs
    for name in os.listdir(base):
        d=os.path.join(base,name)
        if not (os.path.isdir(d) and _has_faiss(d)): 
            continue
        text=""
        meta=os.path.join(d,"meta.json")
        if os.path.exists(meta):
            try:
                meta_obj = json.loads(_safe_read(meta) or "{}")
                text = meta_obj.get("description","")
            except Exception:
                pass
        if not text:
            for fn in ("description.txt","README.md","readme.txt"):
                fp=os.path.join(d,fn)
                if os.path.exists(fp):
                    text=_safe_read(fp)
                    if text: break
        if not text:
            text=name.replace("_"," ").replace("-"," ")
        descs[name]=text
    return descs

@lru_cache(maxsize=1)
def _desc_embeddings() -> dict:
    return {n: tuple(hf_embeddings.embed_query(txt)) for n,txt in _index_descriptions().items()}

def pick_best_index(question:str)->Optional[str]:
    q=np.array(hf_embeddings.embed_query(question),dtype="float32"); qn=np.linalg.norm(q) or 1.0
    best=None; bestsim=-1.0
    for n,v in _desc_embeddings().items():
        v=np.array(v,dtype="float32")
        sim=float(np.dot(q,v)/(np.linalg.norm(v) or 1.0)/qn)
        if sim>bestsim:
            bestsim=sim; best=n
    return best

# ---------------- Journal tag detection ----------------
def detect_journal_tag_from_text(text:str)->str:
    t=text.lower()
    if any(w in t for w in["meditat","mindful","breathe","zen"]): return "meditation"
    if any(w in t for w in["ate","food","meal","lunch","dinner","breakfast","snack"]): return "food_intake"
    if any(w in t for w in["sleep","slept","nap","tired","bed"]): return "sleep"
    if any(w in t for w in["work","study","learn","project","meeting","office"]): return "work_or_study"
    if any(w in t for w in["feel","mood","happy","sad","angry","anxious","excited"]): return "mood"
    if any(w in t for w in["reflect","thought","realize","understand","insight"]): return "reflection"
    return "personal"

# ---------------- Core handler ----------------
def handle_user_message(request: ChatRequest, username: str) -> ChatResponse:
    start_time_of_api_hit = time.perf_counter()
    convo_object_id = get_or_create_conversation(request.conversation_id, username)
    convo_id = str(convo_object_id)

    history = conversation_store.get(convo_id, [])

    # Build augmented question using last topic/date window + prev QA
    augmented_user_q = carry_forward_question(request.question, convo_id)
    augmented_user_q = with_prev_qa(augmented_user_q, history)

    # Save original user message
    save_message(convo_id, "user", request.question)

    # Intercept "yes/ok" confirmations to earlier offers
    last_assistant_msg = ""
    for m in reversed(history):
        if m.get("role") == "assistant":
            last_assistant_msg = m.get("content", "")
            break

    if is_yes_like(request.question) and assistant_offered_check(last_assistant_msg):
        inferred_collection = detect_collection_from_query(last_assistant_msg + " " + request.question)
        inferred_collection = inferred_collection or ctx_collection_fallback(convo_id)

        context_info = {
            "date": request.date,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "conversation_id": convo_id,
            "collection": inferred_collection,
        }
        if not request.start_date and not request.end_date:
            context_info.update(extract_date_context(request.question))

        mongo_query = generate_ai_mongo_query_with_fallback(augmented_user_q, username, context_info)
        personal_data = fetch_personal_data(mongo_query, username)
        personal_context = build_comprehensive_context(personal_data, username)

        if personal_context and personal_context.strip() != "No recent data available for your query.":
            final_reply = apply_personality(personal_context, "friendly")
        else:
            final_reply = apply_personality(
                "I couldn’t find recent synced data. Want help checking your device/app sync settings?",
                "friendly"
            )

        # Update context memory
        maybe_topic = detect_topic(last_assistant_msg + " " + request.question)
        update_conversation_context(
            convo_id,
            topic=maybe_topic,
            collection=inferred_collection,
            date_ctx={k: v for k, v in context_info.items() if k in ("date","start_date","end_date")}
        )

        history.append({"role": "assistant", "content": final_reply})
        conversation_store[convo_id] = history[-10:]
        save_message(convo_id, "assistant", final_reply)

        return ChatResponse(
            reply=final_reply,
            history=conversation_store[convo_id],
            conversation_id=convo_id,
            query_type="mongo_query",
            data_sources=[inferred_collection] if inferred_collection else []
        )

    # ---- LLM classification (with optional prev QA context) ----
    messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
    prev_qa = get_last_qa_pair(history)
    if prev_qa:
        messages.append({
            "role": "system",
            "content": (
                "Previous QA context (use only to resolve pronouns/implicit references):\n"
                f"Q: {prev_qa['question']}\nA: {prev_qa['answer']}\n— End —"
            )
        })
    messages.append({"role": "user", "content": augmented_user_q})

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_API_MODEL", "gpt-4"),
        messages=messages,
        temperature=0.5
    )
    raw_llm_reply = response.choices[0].message.content.strip()

    # Parse response JSON (robust to ```json fences)
    cleaned_reply = re.sub(r"^```(?:json)?|```$", "", raw_llm_reply.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned_reply)
        raw_reply = parsed.get("reply", raw_llm_reply)
        intent = parsed.get("intent", "knowledge_query")
        tag = parsed.get("tag")
        collection = parsed.get("collection")
        print(f"🧾 LLM classified intent: {intent}, tag: {tag}, collection: {collection}")
    except json.JSONDecodeError:
        raw_reply = raw_llm_reply
        intent = "knowledge_query"
        tag = None
        collection = None

    context_reply = None

    # ---- Journal confirmation flow ----
    if convo_id in pending_journal_confirmations:
        user_reply = request.question.strip().lower()
        journal = pending_journal_confirmations.pop(convo_id)

        if user_reply in ["yes", "yeah", "yup", "ok", "sure"]:
            now = datetime.utcnow()
            start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + timedelta(days=1)

            new_entry = {
                "tag": journal["tag"],
                "text": journal["text"],
                "timestamp": now
            }

            existing_journal = journals_collection.find_one({
                "username": username,
                "timestamp": {"$gte": start_of_day, "$lt": end_of_day}
            })

            if existing_journal:
                update_result = journals_collection.update_one(
                    {"_id": existing_journal["_id"]},
                    {
                        "$push": {"entries": new_entry},
                        "$set": {"last_modified": now}
                    }
                )
                if update_result.modified_count > 0:
                    reply = apply_personality(f"📝 Added a new '{journal['tag']}' entry to today's journal.", "friendly")
                else:
                    reply = apply_personality("⚠️ Tried updating your journal but nothing changed.", "friendly")
            else:
                new_journal = {
                    "username": username,
                    "entries": [new_entry],
                    "timestamp": now,
                    "last_modified": now,
                    "conversation_id": convo_id
                }
                try:
                    insert_result = journals_collection.insert_one(new_journal)
                    if insert_result.inserted_id:
                        print("✅ Inserted journal ID:", insert_result.inserted_id)
                        reply = apply_personality(f"✅ Got it! Your '{journal['tag']}' entry has been saved to today's journal.", "friendly")
                    else:
                        reply = apply_personality("⚠️ Something went wrong while saving your journal.", "friendly")
                except Exception as e:
                    print(f"❌ Error inserting journal: {e}")
                    reply = apply_personality("⚠️ Something went wrong while saving your journal. Please try again.", "friendly")
        else:
            reply = apply_personality("Okay, I won't save it. Let me know if you want to record anything else.", "friendly")

        save_message(convo_id, "user", request.question)
        save_message(convo_id, "assistant", reply)

        history.append({"role": "user", "content": request.question})
        history.append({"role": "assistant", "content": reply})
        conversation_store[convo_id] = history[-10:]

        return ChatResponse(
            reply=reply,
            history=conversation_store[convo_id],
            conversation_id=convo_id,
            query_type="journal_entry",
            data_sources=[journal["tag"]] if journal.get("tag") else []
        )

    # ---- Handle intents ----
    if intent == "journal_entry":
        if not tag:
            tag = detect_journal_tag_from_text(request.question)

        pending_journal_confirmations[convo_id] = {
            "username": username,
            "tag": tag or "extra_note",
            "text": request.question
        }

        # Remember topic/collection for follow-ups
        mapped_coll = TOPIC_COLLECTION.get(tag or "", None)
        update_conversation_context(convo_id, topic=tag, collection=mapped_coll)

        final_reply = apply_personality(
            f"Would you like me to save this as a '{tag or 'note'}' journal entry for today? Just say 'yes' or 'no'.",
            "friendly"
        )

        save_message(convo_id, "assistant", final_reply)
        history.append({"role": "user", "content": request.question})
        history.append({"role": "assistant", "content": final_reply})
        conversation_store[convo_id] = history[-10:]

        return ChatResponse(
            reply=final_reply,
            history=conversation_store[convo_id],
            conversation_id=convo_id,
            query_type=intent,
            data_sources=[tag] if tag else []
        )

    elif intent == "mongo_query":
        print("🔄 Searching MongoDB data...")
        used_sources = []

        if request.tags:
            all_contexts = []
            for tag_obj in request.tags:
                tg = tag_obj.tag
                coll = TAG_COLLECTION_MAP.get(tg)
                if not coll:
                    continue

                context_info = {
                    "start_date": tag_obj.start_date,
                    "end_date": tag_obj.end_date,
                    "collection": coll,
                    "conversation_id": convo_id
                }

                mongo_query = generate_ai_mongo_query_with_fallback(tg, username, context_info)
                personal_data = fetch_personal_data(mongo_query, username)
                # print(f"📦 MongoDB Results for tag '{tg}':\n{json.dumps(personal_data, indent=2, default=str)}")
                summary = build_comprehensive_context(personal_data, username)

                if summary and "No recent data" not in summary:
                    # all_contexts.append(f"🗂️ **{tg.title()} Summary:**\n{summary}")
                    used_sources.append(tg)

            if all_contexts:
                formated_answer = format_mongo_answer_llm(augmented_user_q, all_contexts)
                context_reply = apply_personality("\n\n".join(formated_answer), "friendly")
        else:
            print("No tags provided, using LLM-detected context.")
            context_info = {
                "date": request.date,
                "start_date": request.start_date,
                "end_date": request.end_date,
                "conversation_id": convo_id
            }
            if not request.start_date and not request.end_date:
                context_info.update(extract_date_context(request.question))

            collection = detect_collection_from_query(request.question) or ctx_collection_fallback(convo_id)
            context_info["collection"] = collection

            print("🔍 Processing MongoDB query with context:", context_info)

            mongo_query = generate_ai_mongo_query_with_fallback(augmented_user_q, username, context_info)
            print(f"🧠 AI Mongo Query: {mongo_query}")

            personal_data = fetch_personal_data(mongo_query, username)
            # print(f"📦 MongoDB Results: {personal_data}")

            personal_context = build_comprehensive_context(personal_data, username)
            print("Personal_Context:")

            if personal_context and personal_context.strip() != "No recent data available for your query.":
                print("Personal Context:", personal_context)
                formated_answer = format_mongo_answer_llm(augmented_user_q, personal_context)
                print("Formatted Answer:", formated_answer)
                context_reply = apply_personality(formated_answer, "friendly")
            else:
                context_reply = apply_personality(
                    "I couldn't find any recent data for that. Is your device or tracker synced properly? Let me know if you'd like help troubleshooting. 😊",
                    "friendly"
                )

            # Remember topic/collection/date for next turn
            maybe_topic = detect_topic(request.question) or detect_topic(augmented_user_q)
            update_conversation_context(
                convo_id,
                topic=maybe_topic,
                collection=collection,
                date_ctx={k: v for k, v in context_info.items() if k in ("date","start_date","end_date")}
            )

    elif intent == "knowledge_query":
        print("🔄 Searching knowledge base...")
        try:
            start_time = time.perf_counter()
            best_index_name = pick_best_index(augmented_user_q)  # fast + cached

            if best_index_name:
                index_path = os.path.join(FAISS_FOLDER_PATH, best_index_name)
                if (
                    os.path.exists(os.path.join(index_path, "index.faiss"))
                    and os.path.exists(os.path.join(index_path, "index.pkl"))
                ):
                    print(f"🔍 Querying FAISS: {best_index_name}")
                    kb_snippets = query_documents(augmented_user_q, index_path)

                    if kb_snippets:
                        formated_answer = generate_answer_from_context(augmented_user_q, kb_snippets, llm)
                        context_reply = apply_personality("\n\n".join(formated_answer), "friendly")
                    else:
                        context_reply = raw_reply or "I'm here to help!"
            else:
                context_reply = apply_personality(
                    "I couldn't find any relevant information in the knowledge base.",
                    "friendly",
                )

            elapsed_time = time.perf_counter() - start_time
            print(f"⏱️ Main Elapsed time for knowledge base query: {elapsed_time:.4f} seconds")
        except Exception as e:
            print(f"❌ Knowledge base error: {e}")

        # Update topic memory (no collection)
        maybe_topic = detect_topic(request.question) or detect_topic(augmented_user_q)
        if maybe_topic:
            update_conversation_context(convo_id, topic=maybe_topic)

    # ---- Finalize reply text ----
    if intent == "mongo_query" and context_reply:
        final = context_reply
    elif intent == "knowledge_query" and context_reply:
        final = context_reply  # prefer KB context
    else:
        final = raw_reply or "I'm here to help!"

    def style_light(txt: str) -> str:
        return txt if txt.startswith(("✅","📝","💡","⚠️")) else f"💡 {txt}"

    final_reply = style_light(final)

    # Save & return
    history.append({"role": "user", "content": request.question})
    history.append({"role": "assistant", "content": final_reply})
    conversation_store[convo_id] = history[-10:]

    print("Context Reply", context_reply)
    print("🧾 Reply:", raw_reply)

    save_message(convo_id, "assistant", final_reply)
    recent = get_recent_history(convo_id)
    end_time_of_api_hit = time.perf_counter()
    elapsed_time = end_time_of_api_hit - start_time_of_api_hit
    print(f"⏱️ API hit duration: {elapsed_time:.4f} seconds")

    print(f"🏷️ Intent: {intent}, Tag: {tag}, Collection: {collection}")

    return ChatResponse(
        reply=final_reply,
        history=recent['history'],
        conversation_id=convo_id,
        query_type=intent,
        data_sources=[tag] if tag else []
    )

# ---------------- FastAPI route ----------------
@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest, token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid token")
    return handle_user_message(req, username)
