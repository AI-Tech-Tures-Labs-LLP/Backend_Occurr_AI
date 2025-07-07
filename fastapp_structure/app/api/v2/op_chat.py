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
2. 📓 JOURNAL ENTRY DETECTION:
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
3. 🔁 ALERT RESPONSE:
If the message is a simple response like "Yes", "No", "Just did", and the assistant previously asked a reminder or warning — treat it as a response to that prompt.
Flag it as "alert_response", and connect it to the recent alert.

──────────────────────────────────────────────
4. 🧠 QUESTION HANDLING:
If the user asks a question:
- If it relates to their own data (my steps, my sleep, yesterday, today), it’s a "mongo_query".
- If it's general (e.g., "What is a healthy heart rate?"), it’s a "knowledge_query".

──────────────────────────────────────────────
5. 💬 RESPONSE STYLE:
- Always be conversational, warm, supportive, and helpful.
- For journal entries or alert responses, confirm the entry was saved.
- For questions, give helpful and specific answers.
- For smalltalk, keep it friendly and open-ended.

6. 📦 OUTPUT FORMAT:
Always return a JSON string like:
{
  "reply": "Got it! I logged your meditation.",
  "intent": "journal_entry",
  "tag": "meditation"
}
"""

def handle_user_message(request: ChatRequest, username: str) -> ChatResponse:
    convo_id = request.conversation_id or str(ObjectId())
    history = conversation_store.get(convo_id, [])

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_API_MODEL", "gpt-4"),
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": request.question}
        ],
        temperature=0.5
    )

    raw_reply = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw_reply)
        reply = parsed.get("reply", raw_reply)
        intent = parsed.get("intent", "unknown")
        tag = parsed.get("tag")
    except json.JSONDecodeError:
        reply = raw_reply
        intent = "unknown"
        tag = None

    history.append({"role": "user", "content": request.question})
    history.append({"role": "assistant", "content": reply})
    conversation_store[convo_id] = history[-10:]

    

    if intent == "mongo_query":
        context_info = {
            "date": request.date,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "conversation_id": convo_id
        }
        print("🔍 Processing MongoDB query...",{context_info})
        if not request.start_date and not request.end_date:
            context_info.update(extract_date_context(request.question))

        print("📊 Fetching personal data from MongoDB...")
        mongo_queries = generate_ai_mongo_query_with_fallback(request.question, username, context_info)
        print(f"🧠 AI MongoDB Query: {json.dumps(mongo_queries, indent=2)}")

        personal_data = fetch_personal_data(mongo_queries, username)
        print(f"📦 MongoDB Query Results: {json.dumps(personal_data, indent=2, default=str)}")

        personal_context = build_comprehensive_context(personal_data, username)

        # Respond based on available data
        if personal_context and "No recent data" not in personal_context:
            reply = apply_personality(personal_context, "friendly")
        else:
            reply = apply_personality("""
                💡Health Insight💡

                It looks like there isn't any recent heart rate data available for the last 7 days. To help track this, make sure your wearable device or health app is synced and recording properly. If you'd like, I can guide you on how to check or troubleshoot your device. Let me know how I can assist! 😊
                            """, "friendly")

        save_message(convo_id, "assistant", reply)


    elif intent == "knowledge_query":
        kb_context = []
        print("🔄 Searching knowledge base...")
        try:
            index_descriptions = load_index_descriptions_from_folders(FAISS_FOLDER_PATH)
            best_index_name = find_best_matching_index(request.question, index_descriptions)

            if best_index_name:
                index_path = os.path.join(FAISS_FOLDER_PATH, best_index_name)

                if os.path.exists(os.path.join(index_path, "index.faiss")) and os.path.exists(os.path.join(index_path, "index.pkl")):
                    print(f"🔍 Querying FAISS index: {best_index_name}")
                    kb_snippets = query_documents(request.question, index_path)
                    if kb_snippets:
                        kb_context.extend(kb_snippets)
                        print(f"✅ Retrieved {len(kb_snippets)} snippets from {best_index_name}")
                        reply = apply_personality("\n\n".join(kb_context), "friendly")
                        save_message(convo_id, "assistant", reply)
            else:
                print("⚠️ No good match found for the query.")

        except Exception as e:
            print(f"❌ Knowledge base search failed: {e}")

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
