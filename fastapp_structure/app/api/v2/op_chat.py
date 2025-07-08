from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from bson import ObjectId
from datetime import datetime, timedelta
import os, json, re
from openai import OpenAI as OpenAIClient

# Database imports
from app.db.database import users_collection, conversations_collection
from app.db.health_data_model import alert_collection, health_data_collection
from app.db.journal_model import journals_collection
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.api.auth.auth import decode_token
import time
from app.utils.optimized_code_rag import load_faiss_index,query_documents
from app.core.advance_chatbot import  apply_personality
from dotenv import load_dotenv
from app.api.v2.adv_chat import *

load_dotenv()

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

FAISS_FOLDER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "faiss_indexes")
)

client = OpenAIClient(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL")
)
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL"),
    model=os.getenv("OPENAI_API_MODEL")
)

loaded_indexes = {}

class ChatRequest(BaseModel):
    question: str
    tags: Optional[List[str]] = None
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

conversation_store = {}

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
- If it relates to their own data (my steps, my sleep, yesterday, today), it’s a "mongo_query".
- If it's general (e.g., "What is a healthy heart rate?"), it’s a "knowledge_query".

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

import re
def detect_collection_from_query(query: str) -> str:
    query = query.lower()

    collection_keywords = {
        "health_data_collection": ["steps", "heart rate", "sleep", "calories", "spo2", "oxygen", "activity"],
        "journals_collection": ["journal", "mood", "reflection", "dream", "note", "food", "meditation", "entry", "log"],
        "alert_collection": ["alert", "abnormal", "warning", "trigger", "anomaly"],
        "notifications": ["notification", "reminder", "ping", "unread", "messages"],
        "tasks": ["task", "todo", "checklist", "pending", "completed", "due"],
        "breathing_exercises": ["breathing", "inhale", "exhale", "hold", "breathwork", "session"],
        "conversations_collection": ["chat", "conversation", "message history"],
        "users_collection": ["profile", "email", "account", "username", "user data"]
    }

    for collection, keywords in collection_keywords.items():
        if any(kw in query for kw in keywords):
            return collection

    return "journals_collection"  # default fallback


def handle_user_message(request: ChatRequest, username: str) -> ChatResponse:
    convo_id = request.conversation_id or str(ObjectId())
    history = conversation_store.get(convo_id, [])

    # 🧠 Step 1: Ask LLM to classify the message
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_API_MODEL", "gpt-4"),
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": request.question}
        ],
        temperature=0.5
    )

    raw_reply = response.choices[0].message.content.strip()

    # 🧼 Step 2: Clean and parse the response (handle triple-backtick JSON)
    cleaned_reply = re.sub(r"^```(?:json)?|```$", "", raw_reply.strip(), flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(cleaned_reply)
        reply = parsed.get("reply", raw_reply)
        intent = parsed.get("intent", "unknown")
        tag = parsed.get("tag")
        collection = parsed.get("collection")
    except json.JSONDecodeError:
        reply = raw_reply
        intent = "unknown"
        tag = None
        collection = None

    # 💬 Step 3: Store history
    history.append({"role": "user", "content": request.question})
    history.append({"role": "assistant", "content": reply})
    conversation_store[convo_id] = history[-10:]

    # 🗃️ Step 4: Handle mongo_query
    if intent == "mongo_query":
        context_info = {
            "date": request.date,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "conversation_id": convo_id
        }

        # Add NLP-based date extraction if not provided
        if not request.start_date and not request.end_date:
            context_info.update(extract_date_context(request.question))

        # Fallback to keyword detection if LLM didn’t provide collection
        if not collection:
            collection = detect_collection_from_query(request.question)

        print("🔍 Processing MongoDB query with context:", context_info)

        # Step 4b: Build query using collection info
        context_info["collection"] = collection
        mongo_query = generate_ai_mongo_query_with_fallback(request.question, username, context_info)
        print(f"🧠 AI Mongo Query:\n{json.dumps(mongo_query, indent=2)}")

        # Step 4c: Fetch data
        personal_data = fetch_personal_data(mongo_query, username)
        print(f"📦 MongoDB Results:\n{json.dumps(personal_data, indent=2, default=str)}")

        # Step 4d: Generate reply
        personal_context = build_comprehensive_context(personal_data, username)
        if personal_context and "No recent data" not in personal_context:
            reply = apply_personality(personal_context, "friendly")
        else:
            reply = apply_personality(
                "I couldn't find any recent data for that. Is your device or tracker synced properly? Let me know if you'd like help troubleshooting. 😊",
                "friendly"
            )

        save_message(convo_id, "assistant", reply)

    # 📚 Step 5: Handle knowledge_query
    elif intent == "knowledge_query":
        kb_context = []
        print("🔄 Searching knowledge base...")
        try:
            index_descriptions = load_index_descriptions_from_folders(FAISS_FOLDER_PATH)
            best_index_name = find_best_matching_index(request.question, index_descriptions)

            if best_index_name:
                index_path = os.path.join(FAISS_FOLDER_PATH, best_index_name)
                if os.path.exists(os.path.join(index_path, "index.faiss")) and os.path.exists(os.path.join(index_path, "index.pkl")):
                    print(f"🔍 Querying FAISS: {best_index_name}")
                    kb_snippets = query_documents(request.question, index_path)
                    if kb_snippets:
                        kb_context.extend(kb_snippets)
                        reply = apply_personality("\n\n".join(kb_context), "friendly")
                        save_message(convo_id, "assistant", reply)
            else:
                print("⚠️ No match found for this knowledge query.")
        except Exception as e:
            print(f"❌ Knowledge base error: {e}")

    return ChatResponse(
        reply=reply,
        history=history,
        conversation_id=convo_id,
        query_type=intent,
        data_sources=[tag] if tag else []
    )


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest, token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid token")

    return handle_user_message(req, username)
