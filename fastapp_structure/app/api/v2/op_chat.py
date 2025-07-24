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
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
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



# Initialize once globally
hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)


loaded_indexes = {}
# In-memory store to track pending journal confirmations
pending_journal_confirmations = {}


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

conversation_store = {}



TAG_COLLECTION_MAP = {
    "heart_rate": "health_data_collection",
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
    "breathing":"breathing_collection",
    "task":"task_collection"
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
    "reply": "📝 **Today's Journal Summary**:\n• Food: eggs and toast\n• Sleep: 7.5 hours\n• Mood: relaxed",
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

import re
def detect_collection_from_query(query: str) -> str:
    query = query.lower()

    collection_keywords = {
        "health_data_collection": ["steps", "heart rate", "sleep", "calories", "spo2", "oxygen", "activity"],
        "journals_collection": ["journal", "mood", "reflection", "dream", "note", "food", "meditation", "entry", "log",""],
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





def embed(text: str) -> np.ndarray:
    return np.array(hf_embeddings.embed_query(text))

def choose_relevant_reply(raw_reply: str, context_reply: str, question: str) -> str:
    if not context_reply:
        return raw_reply
    if not raw_reply:
        return context_reply

    try:
        question_vec = embed(question)
        raw_vec = embed(raw_reply)
        context_vec = embed(context_reply)

        raw_score = cosine_similarity([question_vec], [raw_vec])[0][0]
        context_score = cosine_similarity([question_vec], [context_vec])[0][0]

        print(f"🔍 Relevance scores — raw: {raw_score:.4f}, context: {context_score:.4f}")
        return context_reply if context_score >= raw_score else raw_reply

    except Exception as e:
        print(f"⚠️ Fallback to keyword match due to embedding error: {e}")
        raw_score = sum(1 for word in question.lower().split() if word in raw_reply.lower())
        context_score = sum(1 for word in question.lower().split() if word in context_reply.lower())
        return context_reply if context_score >= raw_score else raw_reply
    


def handle_user_message(request: ChatRequest, username: str) -> ChatResponse:

    convo_object_id = get_or_create_conversation(request.conversation_id, username)
    convo_id = str(convo_object_id)
    history = conversation_store.get(convo_id, [])

    # ✅ Check if user is replying to a pending journal confirmation
    if convo_id in pending_journal_confirmations:
        user_reply = request.question.strip().lower()
        journal = pending_journal_confirmations.pop(convo_id)

        if user_reply in ["yes", "yeah", "yup", "ok", "sure"]:
        # Check if journal already exists for today, same tag, same user
            start_of_day = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = datetime.utcnow().replace(hour=23, minute=59, second=59, microsecond=999999)

            existing_journal = journals_collection.find_one({
                "username": username,
                "timestamp": {"$gte": start_of_day, "$lte": end_of_day}
            })

            if existing_journal:
                reply = apply_personality(f"ℹ️ You already have a '{journal['tag']}' journal entry for today. Skipped saving duplicate.", "friendly")
            else:
                journals_collection.insert_one({
                    "username": username,
                    "tag": journal["tag"],
                    "text": journal["text"],
                    "timestamp": datetime.utcnow(),
                    "conversation_id": convo_id
                })
                reply = apply_personality(f"✅ Got it! Your '{journal['tag']}' entry has been saved to today's journal.", "friendly")
        else:
            reply = apply_personality("Okay, I won’t save it. Let me know if you want to record anything else.", "friendly")
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
            data_sources=[]
        )

    # 💬 Save initial user message
    save_message(convo_id, "user", request.question)

    # Step 1: LLM classification
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_API_MODEL", "gpt-4"),
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": request.question}
        ],
        temperature=0.5
    )
    raw_llm_reply = response.choices[0].message.content.strip()

    # Step 2: Parse response
    cleaned_reply = re.sub(r"^```(?:json)?|```$", "", raw_llm_reply.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned_reply)
        raw_reply = parsed.get("reply", raw_llm_reply)
        intent = parsed.get("intent", "unknown")
        tag = parsed.get("tag")
        collection = parsed.get("collection")
    except json.JSONDecodeError:
        raw_reply = raw_llm_reply
        intent = "unknown"
        tag = None
        collection = None

    context_reply = None

    # ✅ Step 3: Journal entry — ask for save confirmation
    if intent == "mongo_query":
        context_reply = None
        used_sources = []

        if request.tags:
            all_contexts = []

            for tag_obj in request.tags:
                tag = tag_obj.tag
                collection = TAG_COLLECTION_MAP.get(tag)
                if not collection:
                    continue

                context_info = {
                    "start_date": tag_obj.start_date,
                    "end_date": tag_obj.end_date,
                    "collection": collection,
                    "conversation_id": convo_id
                }

                mongo_query = generate_ai_mongo_query_with_fallback(tag, username, context_info)
                personal_data = fetch_personal_data(mongo_query, username)
                summary = build_comprehensive_context(personal_data, username)

                if summary and "No recent data" not in summary:
                    all_contexts.append(f"🗂️ **{tag.title()} Summary:**\n{summary}")
                    used_sources.append(tag)

            if all_contexts:
                context_reply = apply_personality("\n\n".join(all_contexts), "friendly")

        else:
            # Fallback: regular single query path
            context_info = {
                "date": request.date,
                "start_date": request.start_date,
                "end_date": request.end_date,
                "conversation_id": convo_id
            }

            if not request.start_date and not request.end_date:
                context_info.update(extract_date_context(request.question))

            collection = detect_collection_from_query(request.question)
            context_info["collection"] = collection

            print("🔍 Processing MongoDB query with context:", context_info)

            mongo_query = generate_ai_mongo_query_with_fallback(request.question, username, context_info)
            print(f"🧠 AI Mongo Query:\n{json.dumps(mongo_query, indent=2)}")

            personal_data = fetch_personal_data(mongo_query, username)
            print(f"📦 MongoDB Results:\n{json.dumps(personal_data, indent=2, default=str)}")

            personal_context = build_comprehensive_context(personal_data, username)
            print("Personal_Context:", personal_context)

            if personal_context and personal_context.strip() != "No recent data available for your query.":
                context_reply = apply_personality(personal_context, "friendly")
            else:
                context_reply = apply_personality(
                    "I couldn't find any recent data for that. Is your device or tracker synced properly? Let me know if you'd like help troubleshooting. 😊",
                    "friendly"
                )

    # if intent == "journal_entry":
    #     pending_journal_confirmations[convo_id] = {
    #         "username": username,
    #         "tag": tag,
    #         "text": request.question
    #     }

    #     final_reply = apply_personality(
    #         f"Would you like me to save this as a '{tag}' journal entry for today? Just say 'yes' or 'no'.",
    #         "friendly"
    #     )

    #     save_message(convo_id, "assistant", final_reply)
    #     history.append({"role": "user", "content": request.question})
    #     history.append({"role": "assistant", "content": final_reply})
    #     conversation_store[convo_id] = history[-10:]

    #     return ChatResponse(
    #         reply=final_reply,
    #         history=conversation_store[convo_id],
    #         conversation_id=convo_id,
    #         query_type=intent,
    #         data_sources=[tag] if tag else []
    #     )

    # # Step 4: Handle mongo_query
    # elif intent == "mongo_query":
    #     context_info = {
    #         "date": request.date,
    #         "start_date": request.start_date,
    #         "end_date": request.end_date,
    #         "conversation_id": convo_id
    #     }

        

    #     if not request.start_date and not request.end_date:
    #         context_info.update(extract_date_context(request.question))

    #     if not collection:
    #         collection = detect_collection_from_query(request.question)

    #     print("🔍 Processing MongoDB query with context:", context_info)
    #     context_info["collection"] = collection

    #     mongo_query = generate_ai_mongo_query_with_fallback(request.question, username, context_info)
    #     print(f"🧠 AI Mongo Query:\n{json.dumps(mongo_query, indent=2)}")

    #     personal_data = fetch_personal_data(mongo_query, username)
    #     print(f"📦 MongoDB Results:\n{json.dumps(personal_data, indent=2, default=str)}")

    #     personal_context = build_comprehensive_context(personal_data, username)
    #     print("Personal_Context:", personal_context)
    #     # if personal_context.strip() != "No recent data available for your query.":
    #     if personal_context and personal_context.strip() != "No recent data available for your query.":

    #         context_reply = apply_personality(personal_context, "friendly")
    #     else:
    #         context_reply = apply_personality(
    #             "I couldn't find any recent data for that. Is your device or tracker synced properly? Let me know if you'd like help troubleshooting. 😊",
    #             "friendly"
    #     )

        # if personal_context and "No recent data" not in personal_context:
        #     context_reply = apply_personality(personal_context, "friendly")
        # else:
        #     context_reply = apply_personality(
        #         "I couldn't find any recent data for that. Is your device or tracker synced properly? Let me know if you'd like help troubleshooting. 😊",
        #         "friendly"
        #     )

        # save_message(convo_id, "assistant", context_reply)


    # Step 5: Handle knowledge_query
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
                        context_reply = apply_personality("\n\n".join(kb_context), "friendly")
                        # save_message(convo_id, "assistant", context_reply)
            else:
                print("⚠️ No match found for this knowledge query.")
        except Exception as e:
            print(f"❌ Knowledge base error: {e}")

    # Step 6: Decide which reply to use
    if intent == "mongo_query" and context_reply:
        final_reply = context_reply
    else:
        final_reply = choose_relevant_reply(
            apply_personality(raw_reply, "friendly"),
            context_reply,
            request.question
        )
    # final_reply = choose_relevant_reply(
    # apply_personality(raw_reply, "friendly"),
    # context_reply,
    # request.question
    #  )

    # Step 6: Store to history
    history.append({"role": "user", "content": request.question})
    history.append({"role": "assistant", "content": final_reply})
    conversation_store[convo_id] = history[-10:]

  # reply generated from mongo/knowledge base
    print("Context Reply", context_reply )
    print("🧾 Reply:", raw_reply)
    print("📚 History:", conversation_store[convo_id])

    
    save_message(convo_id, "assistant", final_reply)
    recent = get_recent_history(convo_id)
    

    # Step 7: Return response
    return ChatResponse(
        reply=final_reply,
        history=recent['history'],
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
