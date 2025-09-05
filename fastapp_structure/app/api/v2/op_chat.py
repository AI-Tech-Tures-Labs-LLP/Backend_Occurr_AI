
# #####______THE SECONDARY INTENT DETECTION

# from fastapi import APIRouter, Depends, HTTPException
# from fastapi.security import OAuth2PasswordBearer
# from pydantic import BaseModel
# from typing import List, Dict, Any, Optional
# from bson import ObjectId
# from datetime import datetime, timedelta
# import os, json, re
# from groq import Groq as OpenAIClient

# # Database imports
# from app.db.database import users_collection, conversations_collection
# from app.db.health_data_model import alert_collection, health_data_collection
# from app.db.journal_model import journals_collection
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from app.api.auth.auth import decode_token
# import time
# from app.utils.optimized_code_rag import query_documents,prewarm_indexes
# from app.core.advance_chatbot import  apply_personality
# from dotenv import load_dotenv
# from langchain_huggingface import HuggingFaceEmbeddings
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# from app.api.v2.adv_chat import *
# from functools import lru_cache
# from groq import Groq


# load_dotenv()

# router = APIRouter()
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# FAISS_FOLDER_PATH = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "faiss_indexes")
# )

# client = Groq(
#     api_key=os.getenv("OPENAI_API_KEY"),
#     base_url=os.getenv("OPENAI_API_BASE_URL")
# )
# llm = ChatOpenAI(
#     api_key=os.getenv("OPENAI_API_KEY"),
#     base_url=os.getenv("OPENAI_API_BASE_URL"),
#     model=os.getenv("OPENAI_API_MODEL")
# )

# # Initialize once globally
# hf_embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2",
#     model_kwargs={'device': 'cpu'},
#     encode_kwargs={'normalize_embeddings': True}
# )

# loaded_indexes = {}
# # In-memory store to track pending journal confirmations
# pending_journal_confirmations = {}

# class TagTimeRange(BaseModel):
#     tag: str
#     start_date: Optional[str] = None
#     end_date: Optional[str] = None

# class ChatRequest(BaseModel):
#     question: str
#     tags: Optional[List[TagTimeRange]] = None
#     context: Optional[str] = None
#     date: Optional[str] = None
#     start_date: Optional[str] = None
#     end_date: Optional[str] = None
#     conversation_id: Optional[str] = None

# class ChatResponse(BaseModel):
#     reply: str
#     history: List[Dict[str, Any]]
#     conversation_id: str
#     query_type: str
#     data_sources: List[str]

# conversation_store = {}

# TAG_COLLECTION_MAP = {
#     "heart_rate": "health_data_collection",
#     "steps": "health_data_collection",
#     "sleep": "health_data_collection",
#     "spo2": "health_data_collection",
#     "calories": "health_data_collection",
#     "food_intake": "journals_collection",
#     "meditation": "journals_collection",
#     "work_or_study": "journals_collection",
#     "personal": "journals_collection",
#     "mood": "journals_collection",
#     "extra_note": "journals_collection",
#     "alert": "alert_collection",
#     "notification": "notifications_collection",
#     "reflection": "journals_collection",
#     "breathing":"breathing_collection",
#     "task":"task_collection"
# }

# DEFAULT_SYSTEM_PROMPT = """
# You are an intelligent health and lifestyle assistant.

# Your job is to understand the user's message and determine its purpose.
# Based on the message, take the appropriate action:
# ──────────────────────────────────────────────
# 1. 🔍 INTENT DETECTION:
# Classify the user's message as one of the following:

# - "mongo_query": A question about the user's own health/activity data (e.g., "What was my heart rate yesterday?")
# - "knowledge_query": A general health-related question (e.g., "What is a healthy sleep duration?")
# - "journal_entry": The user is logging something like food, sleep, mood, personal reflections, work, or meditation.
# - "alert_response": A short response to a health alert or reminder (e.g., "Yes", "Done", "I just ate").
# - "greeting": A greeting (e.g., "hi", "good morning").
# - "smalltalk": Casual conversation not meant to be logged or analyzed.

# ──────────────────────────────────────────────
# 2. 📦 COLLECTION DETECTION for "mongo_query":
# If the intent is "mongo_query", also identify which MongoDB collection is relevant.
# Infer the collection based on the topic of the question and include it in the output as the "collection" field.

# Use the following guide to map questions to collections:

# - "health_data_collection" → heart rate, steps, sleep, calories, spo2
# - "journals_collection" → journal, mood, dream, food, meditation, reflection
# - "alert_collection" → alert, health warning, abnormal reading
# - "notifications" → notification, reminder, message
# - "tasks" → task, to-do, pending item, checklist
# - "breathing_exercises" → breathing, inhale, exhale, session
# - "conversations_collection" → chat history, past messages
# - "users_collection" → profile, email, username, account info

# Include this as:
# "collection": "<collection_name>"
# 👉 For journal or health data queries that involve a specific **date only**, use this MongoDB date filter to **ignore time**:
# ```json
# {
#   "$expr": {
#     "$eq": [
#       { "$dateToString": { "format": "%Y-%m-%d", "date": "$timestamp" } },
#       "2025-07-23"
#     ]
#   }
# }

# ──────────────────────────────────────────────
# 3. 📓 JOURNAL ENTRY DETECTION:
# If the message logs a real-life activity, classify it as a "journal_entry".
# Further tag the category as one of:
# - "meditation"
# - "food_intake"
# - "sleep"
# - "work_or_study"
# - "personal"
# - "extra_note"

# Examples:
# - "I meditated for 10 minutes" → journal_entry, meditation
# - "Had eggs for breakfast" → journal_entry, food_intake
# - "I studied math today" → journal_entry, work_or_study

# ──────────────────────────────────────────────
# 4. 🔁 ALERT RESPONSE:
# If the message is a simple response like "Yes", "No", "Just did", and the assistant previously asked a reminder or warning — treat it as a response to that prompt.
# Flag it as "alert_response", and connect it to the recent alert.

# ──────────────────────────────────────────────
# 5. 🧠 QUESTION HANDLING:
# If the user asks a question:

# - If it relates to their own data (e.g., "What was my heart rate yesterday?", "What is the summary of my journal today?"), it’s a "mongo_query".
# - If it's general health advice (e.g., "What is a healthy heart rate?"), it’s a "knowledge_query".
# - If the user asks for a **summary of their journal entries**, respond with a formatted and human-friendly summary using the provided journal content.
#   Example format:
#   {
#     "reply": "📝 **Today's Journal Summary**:\n• Food: eggs and toast\n• Sleep: 7.5 hours\n• Mood: relaxed",
#     "intent": "mongo_query",
#     "collection": "journals_collection"
#   }

# ──────────────────────────────────────────────
# 6. 💬 RESPONSE STYLE:
# - Always be conversational, warm, supportive, and helpful.
# - For journal entries or alert responses, confirm the entry was saved.
# - For questions, give helpful and specific answers.
# - For smalltalk, keep it friendly and open-ended.

# ──────────────────────────────────────────────
# 7. 📤 OUTPUT FORMAT:
# Always return a JSON string like:
# {
#   "reply": "Got it! I logged your meditation.",
#   "intent": "journal_entry",
#   "tag": "meditation"
# }

# If intent is "mongo_query", also include:
# "collection": "<target_collection>"
# """

# import re

# YES_TOKENS = {"yes", "yeah", "yup", "ok", "okay", "sure", "please", "yes please", "do it", "go ahead", "confirm", "y", "k"}

# OFFER_PATTERNS = re.compile(
#     r"(would you like me to|should i|do you want me to|want me to|shall i|i can)\s+(check|look|pull|fetch|get)",
#     re.IGNORECASE
# )

# def is_yes_like(text: str) -> bool:
#     t = text.strip().lower()
#     return any(tok in t for tok in YES_TOKENS)

# def assistant_offered_check(text: str) -> bool:
#     return bool(OFFER_PATTERNS.search(text or ""))

# def detect_collection_from_query(query: str) -> str:
#     query = query.lower()

#     collection_keywords = {
#         "health_data_collection": ["steps", "heart rate", "heartrate", "pulse", "sleep", "calories", "spo2", "oxygen", "activity"],
#         "journals_collection": ["journal", "mood", "reflection", "dream", "note", "food", "meal", "meditation", "entry", "log"],
#         "alert_collection": ["alert", "abnormal", "warning", "trigger", "anomaly"],
#         "notifications": ["notification", "reminder", "ping", "unread", "messages"],
#         "tasks": ["task", "todo", "checklist", "pending", "completed", "due"],
#         "breathing_exercises": ["breathing", "inhale", "exhale", "hold", "breathwork", "session"],
#         "conversations_collection": ["chat", "conversation", "message history"],
#         "users_collection": ["profile", "email", "account", "username", "user data"]
#     }

#     for collection, keywords in collection_keywords.items():
#         if any(kw in query for kw in keywords):
#             return collection

#     return "journals_collection"  # default fallback

# def embed(text: str) -> np.ndarray:
#     return np.array(hf_embeddings.embed_query(text))

# def choose_relevant_reply(raw_reply: str, context_reply: str, question: str) -> str:
#     if not context_reply:
#         return raw_reply
#     if not raw_reply:
#         return context_reply

#     try:
#         question_vec = embed(question)
#         raw_vec = embed(raw_reply)
#         context_vec = embed(context_reply)

#         raw_score = cosine_similarity([question_vec], [raw_vec])[0][0]
#         context_score = cosine_similarity([question_vec], [context_vec])[0][0]

#         print(f"🔍 Relevance scores — raw: {raw_score:.4f}, context: {context_score:.4f}")
#         return context_reply if context_score >= raw_score else raw_reply

#     except Exception as e:
#         print(f"⚠️ Fallback to keyword match due to embedding error: {e}")
#         raw_score = sum(1 for word in question.lower().split() if word in raw_reply.lower())
#         context_score = sum(1 for word in question.lower().split() if word in context_reply.lower())
#         return context_reply if context_score >= raw_score else raw_reply


# from functools import lru_cache

# def _has_faiss(dirpath: str) -> bool:
#     return (
#         os.path.exists(os.path.join(dirpath, "index.faiss")) and
#         os.path.exists(os.path.join(dirpath, "index.pkl"))
#     )

# def _safe_read(path: str) -> str:
#     try:
#         with open(path, "r", encoding="utf-8", errors="ignore") as f:
#             return f.read().strip()
#     except Exception:
#         return ""

# @lru_cache(maxsize=1)
# def _index_descriptions() -> dict:
#     """
#     Discover FAISS index folders in FAISS_FOLDER_PATH and get a short description
#     for each (meta.json.description -> description.txt/readme -> folder name).
#     Cached for speed.
#     """
#     descs = {}
#     base = FAISS_FOLDER_PATH
#     for name in os.listdir(base):
#         d = os.path.join(base, name)
#         if not (os.path.isdir(d) and _has_faiss(d)):
#             continue

#         text = ""
#         meta_json = os.path.join(d, "meta.json")
#         if os.path.exists(meta_json):
#             try:
#                 meta = json.loads(_safe_read(meta_json) or "{}")
#                 text = meta.get("description") or ""
#             except Exception:
#                 pass

#         if not text:
#             for fname in ("description.txt", "README.md", "readme.txt"):
#                 fp = os.path.join(d, fname)
#                 if os.path.exists(fp):
#                     text = _safe_read(fp)
#                     if text:
#                         break

#         if not text:
#             # fallback: use a cleaned folder name
#             text = name.replace("__", " ").replace("_", " ").replace("-", " ")

#         descs[name] = text
#     return descs

# @lru_cache(maxsize=1)
# def _desc_embeddings() -> dict:
#     """
#     Embed each index description once (uses your global hf_embeddings).
#     """
#     return {name: tuple(hf_embeddings.embed_query(txt))
#             for name, txt in _index_descriptions().items()}

# def pick_best_index(question: str) -> Optional[str]:
#     """
#     Choose the best index by cosine similarity between the question embedding
#     and the cached description embeddings.
#     """
#     import numpy as np
#     q = np.array(hf_embeddings.embed_query(question), dtype="float32")
#     qn = np.linalg.norm(q) or 1.0

#     best_name, best_sim = None, -1.0
#     for name, vec in _desc_embeddings().items():
#         v = np.array(vec, dtype="float32")
#         sim = float(np.dot(q, v) / ((np.linalg.norm(v) or 1.0) * qn))
#         if sim > best_sim:
#             best_sim, best_name = sim, name
#     return best_name



# def handle_user_message(request: ChatRequest, username: str) -> ChatResponse:
#     start_time_of_api_hit = time.perf_counter()
#     convo_object_id = get_or_create_conversation(request.conversation_id, username)
#     convo_id = str(convo_object_id)
#     history = conversation_store.get(convo_id, [])

#     if convo_id in pending_journal_confirmations:
#         user_reply = request.question.strip().lower()
#         journal = pending_journal_confirmations.pop(convo_id)

#         if user_reply in ["yes", "yeah", "yup", "ok", "sure"]:
#             now = datetime.utcnow()
#             start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
#             end_of_day = start_of_day + timedelta(days=1)

#             new_entry = {
#                 "tag": journal["tag"],
#                 "text": journal["text"],
#                 "timestamp": now
#             }

#             existing_journal = journals_collection.find_one({
#                 "username": username,
#                 "timestamp": {"$gte": start_of_day, "$lt": end_of_day}
#             })

#             if existing_journal:
#                 update_result = journals_collection.update_one(
#                     {"_id": existing_journal["_id"]},
#                     {
#                         "$push": {"entries": new_entry},
#                         "$set": {"last_modified": now}
#                     }
#                 )
#                 if update_result.modified_count > 0:
#                     reply = apply_personality(f"📝 Added a new '{journal['tag']}' entry to today's journal.", "friendly")
#                 else:
#                     reply = apply_personality("⚠️ Tried updating your journal but nothing changed.", "friendly")
#             else:
#                 new_journal = {
#                     "username": username,
#                     "entries": [new_entry],
#                     "timestamp": now,
#                     "last_modified": now,
#                     "conversation_id": convo_id
#                 }

#                 try:
#                     insert_result = journals_collection.insert_one(new_journal)
#                     if insert_result.inserted_id:
#                         print("✅ Inserted journal ID:", insert_result.inserted_id)
#                         reply = apply_personality(f"✅ Got it! Your '{journal['tag']}' entry has been saved to today's journal.", "friendly")
#                     else:
#                         reply = apply_personality("⚠️ Something went wrong while saving your journal.", "friendly")
#                 except Exception as e:
#                     print(f"❌ Error inserting journal: {e}")
#                     reply = apply_personality("⚠️ Something went wrong while saving your journal. Please try again.", "friendly")
#         else:
#             reply = apply_personality("Okay, I won't save it. Let me know if you want to record anything else.", "friendly")

#         save_message(convo_id, "user", request.question)
#         save_message(convo_id, "assistant", reply)

#         history.append({"role": "user", "content": request.question})
#         history.append({"role": "assistant", "content": reply})
#         conversation_store[convo_id] = history[-10:]

#         return ChatResponse(
#             reply=reply,
#             history=conversation_store[convo_id],
#             conversation_id=convo_id,
#             query_type="journal_entry",
#             data_sources=[journal["tag"]] if journal.get("tag") else []
#         )

#     # 💬 Save initial user message
#     save_message(convo_id, "user", request.question)

#     # 🔎 Intercept yes/confirm replies to a previous "shall I check your data?" offer
#     last_assistant_msg = ""
#     for m in reversed(history):
#         if m.get("role") == "assistant":
#             last_assistant_msg = m.get("content", "")
#             break

#     if is_yes_like(request.question) and assistant_offered_check(last_assistant_msg):
#         inferred_collection = detect_collection_from_query(last_assistant_msg + " " + request.question)
#         context_info = {
#             "date": request.date,
#             "start_date": request.start_date,
#             "end_date": request.end_date,
#             "conversation_id": convo_id,
#             "collection": inferred_collection,
#         }

#         if not request.start_date and not request.end_date:
#             context_info.update(extract_date_context(request.question))

#         mongo_query = generate_ai_mongo_query_with_fallback(request.question, username, context_info)
#         personal_data = fetch_personal_data(mongo_query, username)
#         personal_context = build_comprehensive_context(personal_data, username)

#         if personal_context and personal_context.strip() != "No recent data available for your query.":
#             final_reply = apply_personality(personal_context, "friendly")
#         else:
#             final_reply = apply_personality(
#                 "I couldn’t find recent synced data. Want help checking your device/app sync settings?",
#                 "friendly"
#             )

#         history.append({"role": "assistant", "content": final_reply})
#         conversation_store[convo_id] = history[-10:]
#         save_message(convo_id, "assistant", final_reply)

#         return ChatResponse(
#             reply=final_reply,
#             history=conversation_store[convo_id],
#             conversation_id=convo_id,
#             query_type="mongo_query",
#             data_sources=[inferred_collection]
#         )

#     # Step 1: LLM classification
#     response = client.chat.completions.create(
#         model=os.getenv("OPENAI_API_MODEL", "gpt-4"),
#         messages=[
#             {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
#             {"role": "user", "content": request.question}
#         ],
#         temperature=0.5
#     )
#     raw_llm_reply = response.choices[0].message.content.strip()

#     # Step 2: Parse response
#     cleaned_reply = re.sub(r"^```(?:json)?|```$", "", raw_llm_reply.strip(), flags=re.MULTILINE).strip()
#     try:
#         parsed = json.loads(cleaned_reply)
#         raw_reply = parsed.get("reply", raw_llm_reply)
#         intent = parsed.get("intent", "knowledge_query")
#         tag = parsed.get("tag") 
#         collection = parsed.get("collection")
#         print(f"🧾 LLM classified intent: {intent}, tag: {tag}, collection: {collection}")
#     except json.JSONDecodeError:
#         raw_reply = raw_llm_reply
#         intent = "knowledge_query"
#         tag = None
#         collection = None

#     context_reply = None
    
    

#     # ✅ Step 3: Handle journal entry - ask for save confirmation
#     if intent == "journal_entry":
#         if not tag:
#             tag = detect_journal_tag_from_text(request.question)

#         pending_journal_confirmations[convo_id] = {
#             "username": username,
#             "tag": tag or "extra_note",
#             "text": request.question
#         }

#         final_reply = apply_personality(
#             f"Would you like me to save this as a '{tag or 'note'}' journal entry for today? Just say 'yes' or 'no'.",
#             "friendly"
#         )

#         save_message(convo_id, "assistant", final_reply)
#         history.append({"role": "user", "content": request.question})
#         history.append({"role": "assistant", "content": final_reply})
#         conversation_store[convo_id] = history[-10:]

#         return ChatResponse(
#             reply=final_reply,
#             history=conversation_store[convo_id],
#             conversation_id=convo_id,
#             query_type=intent,
#             data_sources=[tag] if tag else []
#         )

#     elif intent == "mongo_query":
#         context_reply = None
#         used_sources = []

#         if request.tags:
#             all_contexts = []

#             for tag_obj in request.tags:
#                 tag = tag_obj.tag
#                 collection = TAG_COLLECTION_MAP.get(tag)
#                 if not collection:
#                     continue

#                 context_info = {
#                     "start_date": tag_obj.start_date,
#                     "end_date": tag_obj.end_date,
#                     "collection": collection,
#                     "conversation_id": convo_id
#                 }

#                 mongo_query = generate_ai_mongo_query_with_fallback(tag, username, context_info)
#                 personal_data = fetch_personal_data(mongo_query, username)
#                 summary = build_comprehensive_context(personal_data, username)

#                 if summary and "No recent data" not in summary:
#                     all_contexts.append(f"🗂️ **{tag.title()} Summary:**\n{summary}")
#                     used_sources.append(tag)

#             if all_contexts:
#                 context_reply = apply_personality("\n\n".join(all_contexts), "friendly")

#         else:
#             context_info = {
#                 "date": request.date,
#                 "start_date": request.start_date,
#                 "end_date": request.end_date,
#                 "conversation_id": convo_id
#             }

#             if not request.start_date and not request.end_date:
#                 context_info.update(extract_date_context(request.question))

#             collection = detect_collection_from_query(request.question)
#             context_info["collection"] = collection

#             print("🔍 Processing MongoDB query with context:", context_info)

#             mongo_query = generate_ai_mongo_query_with_fallback(request.question, username, context_info)
#             print(f"🧠 AI Mongo Query:\n{json.dumps(mongo_query, indent=2)}")

#             personal_data = fetch_personal_data(mongo_query, username)
#             print(f"📦 MongoDB Results:\n{json.dumps(personal_data, indent=2, default=str)}")

#             personal_context = build_comprehensive_context(personal_data, username)
#             print("Personal_Context:", personal_context)

#             if personal_context and personal_context.strip() != "No recent data available for your query.":
#                 context_reply = apply_personality(personal_context, "friendly")
#                 print("context_reply",context_reply)
#             else:
#                 context_reply = apply_personality(
#                     "I couldn't find any recent data for that. Is your device or tracker synced properly? Let me know if you'd like help troubleshooting. 😊",
#                     "friendly"
#                 )

#     elif intent == "knowledge_query":
#         print("🔄 Searching knowledge base...")
#         try:
#             start_time = time.perf_counter()
#             best_index_name = pick_best_index(request.question)  # fast + cached

#             if best_index_name:
#                 index_path = os.path.join(FAISS_FOLDER_PATH, best_index_name)
#                 if (
#                     os.path.exists(os.path.join(index_path, "index.faiss"))
#                     and os.path.exists(os.path.join(index_path, "index.pkl"))
#                 ):
#                     print(f"🔍 Querying FAISS: {best_index_name}")
#                     kb_snippets = query_documents(request.question, index_path)

#                     if kb_snippets:
#                         context_reply = apply_personality("\n\n".join(kb_snippets), "friendly")
#                     else:
#                         context_reply = raw_reply or "I'm here to help!"
#             else:
#                 context_reply = apply_personality(
#                     "I couldn't find any relevant information in the knowledge base.",
#                     "friendly",
#                 )

#             elapsed_time = time.perf_counter() - start_time
#             print(f"⏱️ Main Elapsed time for knowledge base query: {elapsed_time:.4f} seconds")
#         except Exception as e:
#             print(f"❌ Knowledge base error: {e}")
#     if intent == "mongo_query" and context_reply:
#         final = context_reply
#     elif intent == "knowledge_query" and context_reply:
#         # final = choose_relevant_reply(raw_reply, context_reply, request.question)
#         final = context_reply  # prefer context reply for knowledge queries
#     else:
#         final = raw_reply or "I'm here to help!"

#     # Optional LIGHT styling without LLM (fast)
#     def style_light(txt: str) -> str:
#         return txt if txt.startswith(("✅","📝","💡","⚠️")) else f"💡 {txt}"

#     final_reply = style_light(final)  # ⬅️ no apply_personality() here

#     # If you absolutely need LLM styling, gate it via env:
#     STYLE_WITH_LLM = os.getenv("KB_STYLE_WITH_LLM", "false").lower() == "true"

# # later, right before returning:
#     # final_reply = context_reply if (intent in ("mongo_query", "knowledge_query") and context_reply) else (raw_reply or "I'm here to help!")
#     # if STYLE_WITH_LLM:
#         # final_reply = apply_personality(final_reply, "friendly")
#     history.append({"role": "user", "content": request.question})
#     history.append({"role": "assistant", "content": final_reply})
#     conversation_store[convo_id] = history[-10:]

#     print("Context Reply", context_reply )
#     print("🧾 Reply:", raw_reply)


#     save_message(convo_id, "assistant", final_reply)
#     recent = get_recent_history(convo_id)
#     end_time_of_api_hit = time.perf_counter()
#     elapsed_time = end_time_of_api_hit - start_time_of_api_hit
#     print(f"⏱️ API hit duration: {elapsed_time:.4f} seconds")

#     print(f"🏷️ Intent: {intent}, Tag: {tag}, Collection: {collection}")

#     return ChatResponse(
#         reply=final_reply,
#         history=recent['history'],
#         conversation_id=convo_id,
#         query_type=intent,
#         data_sources=[tag] if tag else []
#     )

# def detect_journal_tag_from_text(text: str) -> str:
#     text_lower = text.lower()
#     if any(word in text_lower for word in ["meditat", "mindful", "breathe", "zen"]):
#         return "meditation"
#     elif any(word in text_lower for word in ["ate", "food", "meal", "lunch", "dinner", "breakfast", "snack"]):
#         return "food_intake"
#     elif any(word in text_lower for word in ["sleep", "slept", "nap", "tired", "bed"]):
#         return "sleep"
#     elif any(word in text_lower for word in ["work", "study", "learn", "project", "meeting", "office"]):
#         return "work_or_study"
#     elif any(word in text_lower for word in ["feel", "mood", "happy", "sad", "angry", "anxious", "excited"]):
#         return "mood"
#     elif any(word in text_lower for word in ["reflect", "thought", "realize", "understand", "insight"]):
#         return "reflection"
#     else:
#         return "personal"

# @router.post("/chat", response_model=ChatResponse)
# def chat_endpoint(req: ChatRequest, token: str = Depends(oauth2_scheme)):
#     valid, username = decode_token(token)
#     if not valid:
#         raise HTTPException(status_code=401, detail="Invalid token")

#     return handle_user_message(req, username)

# #####______THE SECONDARY INTENT DETECTION (with follow-up awareness)

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from bson import ObjectId
from datetime import datetime, timedelta
import os, json, re, time
import numpy as np

from groq import Groq as OpenAIClient  # kept for compatibility with your comment
from groq import Groq

# Database imports
from app.db.database import users_collection, conversations_collection
from app.db.health_data_model import alert_collection, health_data_collection
from app.db.journal_model import journals_collection

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.api.auth.auth import decode_token

from app.utils.optimized_code_rag import query_documents, prewarm_indexes
from app.core.advance_chatbot import apply_personality
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# Your v2 advanced chat helpers (assumed present in your codebase)
# Must provide: extract_date_context, generate_ai_mongo_query_with_fallback,
#              fetch_personal_data, build_comprehensive_context,
#              get_or_create_conversation, save_message, get_recent_history
from app.api.v2.adv_chat import *

from functools import lru_cache

load_dotenv()

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

FAISS_FOLDER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "faiss_indexes")
)

client = Groq(
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

# Stores
loaded_indexes: Dict[str, Any] = {}
conversation_store: Dict[str, List[Dict[str, str]]] = {}

# NEW: Per-conversation carry-over for topic/collection/date & last answer hint
conversation_state: Dict[str, Dict[str, Any]] = {}

# In-memory store to track pending journal confirmations
pending_journal_confirmations: Dict[str, Dict[str, Any]] = {}

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
👉 For journal or health data queries that involve a specific **date only**, use this MongoDB date filter to **ignore time**:
```json
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

# -------------------------
# Follow-up detection & context helpers
# -------------------------

YES_TOKENS = {"yes", "yeah", "yup", "ok", "okay", "sure", "please", "yes please", "do it", "go ahead", "confirm", "y", "k"}

OFFER_PATTERNS = re.compile(
    r"(would you like me to|should i|do you want me to|want me to|shall i|i can)\s+(check|look|pull|fetch|get)",
    re.IGNORECASE
)

FOLLOWUP_CLUES = re.compile(
    r"\b(when|where|why|how|which|and then|what about|too high|too low|what next|show me|details|more)\b|^(and|also|then|what about|how about)\b",
    re.IGNORECASE
)

def is_yes_like(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(tok in t for tok in YES_TOKENS)

def assistant_offered_check(text: str) -> bool:
    return bool(OFFER_PATTERNS.search(text or ""))

def is_followup_like(text: str) -> bool:
    t = (text or "").strip()
    return len(t.split()) <= 6 or bool(FOLLOWUP_CLUES.search(t))

def build_context_messages(history: list, current_question: str, max_pairs: int = 3):
    """
    Build a compact prior context block for classification.
    Includes up to `max_pairs` of (user, assistant) turns.
    """
    turns = []
    user_buf, asst_buf = [], []
    count_pairs = 0
    for m in reversed(history):
        role, content = m.get("role"), m.get("content", "")
        if role == "assistant":
            asst_buf.append(content)
        elif role == "user":
            user_buf.append(content)

        if user_buf and asst_buf:
            turns.append((user_buf.pop(0), asst_buf.pop(0)))
            count_pairs += 1
            if count_pairs >= max_pairs:
                break

    turns = list(reversed(turns))

    prior = []
    for i, (u, a) in enumerate(turns, start=1):
        prior.append(f"{i}. User: {u}\n   Assistant: {a}")

    prior_block = "\n".join(prior).strip()
    return [
        {"role": "system", "content": "You can use the following recent conversation for context if the current message is a follow-up."},
        {"role": "user", "content": f"Recent context:\n{prior_block or '(no prior messages)'}\n\nCurrent message: {current_question}\nReturn ONLY the specified JSON per the system instructions."}
    ]

def update_conversation_state(convo_id: str, intent: str, tag: Optional[str], collection: Optional[str], final_reply: str, raw_user_question: str):
    st = conversation_state.get(convo_id, {})
    if intent == "mongo_query":
        topic = tag or collection or detect_collection_from_query(raw_user_question)
        st.update({
            "last_intent": intent,
            "last_collection": collection or detect_collection_from_query(raw_user_question),
            "last_topic": topic,
            "last_answer_hint": (final_reply or "")[:600]
        })
    elif intent == "knowledge_query":
        st.update({
            "last_intent": intent,
            "last_collection": None,
            "last_topic": tag or "general_knowledge",
            "last_answer_hint": (final_reply or "")[:600]
        })
    else:
        st.update({
            "last_intent": intent,
            "last_topic": tag or st.get("last_topic"),
            "last_answer_hint": (final_reply or "")[:600]
        })
    conversation_state[convo_id] = st

def inherit_context_for_followup(convo_id: str, question: str, context_info: dict) -> dict:
    st = conversation_state.get(convo_id, {})
    if not st:
        return context_info

    if is_followup_like(question):
        if not context_info.get("collection") and st.get("last_collection"):
            context_info["collection"] = st["last_collection"]

        context_info["followup_context"] = {
            "last_intent": st.get("last_intent"),
            "last_topic": st.get("last_topic"),
            "last_answer_hint": st.get("last_answer_hint", "")
        }
    return context_info

def nudge_metric_from_last_topic(convo_id: str, question: str) -> Optional[str]:
    st = conversation_state.get(convo_id, {})
    q = (question or "").lower()
    if any(w in q for w in ["too high", "too low", "spike", "drop", "above", "below", "range", "normal"]):
        return st.get("last_collection")  # often "health_data_collection"
    return None

# -------------------------
# Embedding helpers & FAISS index selection (cached)
# -------------------------

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

def _has_faiss(dirpath: str) -> bool:
    return (
        os.path.exists(os.path.join(dirpath, "index.faiss")) and
        os.path.exists(os.path.join(dirpath, "index.pkl"))
    )

def _safe_read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception:
        return ""

@lru_cache(maxsize=1)
def _index_descriptions() -> dict:
    descs = {}
    base = FAISS_FOLDER_PATH
    for name in os.listdir(base):
        d = os.path.join(base, name)
        if not (os.path.isdir(d) and _has_faiss(d)):
            continue

        text = ""
        meta_json = os.path.join(d, "meta.json")
        if os.path.exists(meta_json):
            try:
                meta = json.loads(_safe_read(meta_json) or "{}")
                text = meta.get("description") or ""
            except Exception:
                pass

        if not text:
            for fname in ("description.txt", "README.md", "readme.txt"):
                fp = os.path.join(d, fname)
                if os.path.exists(fp):
                    text = _safe_read(fp)
                    if text:
                        break

        if not text:
            text = name.replace("__", " ").replace("_", " ").replace("-", " ")

        descs[name] = text
    return descs

@lru_cache(maxsize=1)
def _desc_embeddings() -> dict:
    return {name: tuple(hf_embeddings.embed_query(txt))
            for name, txt in _index_descriptions().items()}

def pick_best_index(question: str) -> Optional[str]:
    q = np.array(hf_embeddings.embed_query(question), dtype="float32")
    qn = np.linalg.norm(q) or 1.0

    best_name, best_sim = None, -1.0
    for name, vec in _desc_embeddings().items():
        v = np.array(vec, dtype="float32")
        sim = float(np.dot(q, v) / ((np.linalg.norm(v) or 1.0) * qn))
        if sim > best_sim:
            best_sim, best_name = sim, name
    return best_name

# -------------------------
# Core collection detection
# -------------------------

def detect_collection_from_query(query: str) -> str:
    query = (query or "").lower()

    collection_keywords = {
        "health_data_collection": ["steps", "heart rate", "heartrate", "pulse", "sleep", "calories", "spo2", "oxygen", "activity"],
        "journals_collection": ["journal", "mood", "reflection", "dream", "note", "food", "meal", "meditation", "entry", "log"],
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

# -------------------------
# Journal tag detection
# -------------------------

def detect_journal_tag_from_text(text: str) -> str:
    text_lower = (text or "").lower()
    if any(word in text_lower for word in ["meditat", "mindful", "breathe", "zen"]):
        return "meditation"
    elif any(word in text_lower for word in ["ate", "food", "meal", "lunch", "dinner", "breakfast", "snack"]):
        return "food_intake"
    elif any(word in text_lower for word in ["sleep", "slept", "nap", "tired", "bed"]):
        return "sleep"
    elif any(word in text_lower for word in ["work", "study", "learn", "project", "meeting", "office"]):
        return "work_or_study"
    elif any(word in text_lower for word in ["feel", "mood", "happy", "sad", "angry", "anxious", "excited"]):
        return "mood"
    elif any(word in text_lower for word in ["reflect", "thought", "realize", "understand", "insight"]):
        return "reflection"
    else:
        return "personal"

# -------------------------
# Main handler
# -------------------------

def handle_user_message(request: ChatRequest, username: str) -> ChatResponse:
    start_time_of_api_hit = time.perf_counter()
    convo_object_id = get_or_create_conversation(request.conversation_id, username)
    convo_id = str(convo_object_id)
    history = conversation_store.get(convo_id, [])

    # ✅ Pending journal confirmation path
    if convo_id in pending_journal_confirmations:
        user_reply = (request.question or "").strip().lower()
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

        # Update state for continuity
        update_conversation_state(
            convo_id=convo_id,
            intent="journal_entry",
            tag=journal.get("tag"),
            collection=None,
            final_reply=reply,
            raw_user_question=request.question
        )

        return ChatResponse(
            reply=reply,
            history=conversation_store[convo_id],
            conversation_id=convo_id,
            query_type="journal_entry",
            data_sources=[journal["tag"]] if journal.get("tag") else []
        )

    # 💬 Save initial user message
    save_message(convo_id, "user", request.question)

    # 🔎 Intercept yes/confirm replies to a previous "shall I check your data?" offer
    last_assistant_msg = ""
    for m in reversed(history):
        if m.get("role") == "assistant":
            last_assistant_msg = m.get("content", "")
            break

    if is_yes_like(request.question) and assistant_offered_check(last_assistant_msg):
        inferred_collection = detect_collection_from_query(last_assistant_msg + " " + (request.question or ""))
        context_info = {
            "date": request.date,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "conversation_id": convo_id,
            "collection": inferred_collection,
        }

        if not request.start_date and not request.end_date:
            context_info.update(extract_date_context(request.question))

        # add prior mini context
        mini_ctx = "\n".join([
            f"Prev Q: {m['content']}" if m['role']=='user' else f"Prev A: {m['content']}"
            for m in history[-4:]
        ])
        context_info["prior_qa"] = mini_ctx[-1200:]
        context_info = inherit_context_for_followup(convo_id, request.question, context_info)

        mongo_query = generate_ai_mongo_query_with_fallback(request.question, username, context_info)
        personal_data = fetch_personal_data(mongo_query, username)
        personal_context = build_comprehensive_context(personal_data, username)

        if personal_context and personal_context.strip() != "No recent data available for your query.":
            final_reply = apply_personality(personal_context, "friendly")
        else:
            final_reply = apply_personality(
                "I couldn’t find recent synced data. Want help checking your device/app sync settings?",
                "friendly"
            )

        history.append({"role": "assistant", "content": final_reply})
        conversation_store[convo_id] = history[-10:]
        save_message(convo_id, "assistant", final_reply)

        # Update state
        update_conversation_state(
            convo_id=convo_id,
            intent="mongo_query",
            tag=None,
            collection=inferred_collection,
            final_reply=final_reply,
            raw_user_question=request.question
        )

        return ChatResponse(
            reply=final_reply,
            history=conversation_store[convo_id],
            conversation_id=convo_id,
            query_type="mongo_query",
            data_sources=[inferred_collection]
        )

    # -------------------------
    # Step 1: LLM classification (inject prior context if follow-up-like)
    # -------------------------
    classifier_messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
    if is_followup_like(request.question) and history:
        classifier_messages.extend(build_context_messages(history, request.question))
    else:
        classifier_messages.append({"role": "user", "content": request.question})

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_API_MODEL", "gpt-4"),
        messages=classifier_messages,
        temperature=0.5
    )
    raw_llm_reply = response.choices[0].message.content.strip()

    # Step 2: Parse response
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

    # ✅ Step 3: Handle journal entry - ask for save confirmation
    if intent == "journal_entry":
        if not tag:
            tag = detect_journal_tag_from_text(request.question)

        pending_journal_confirmations[convo_id] = {
            "username": username,
            "tag": tag or "extra_note",
            "text": request.question
        }

        final_reply = apply_personality(
            f"Would you like me to save this as a '{tag or 'note'}' journal entry for today? Just say 'yes' or 'no'.",
            "friendly"
        )

        save_message(convo_id, "assistant", final_reply)
        history.append({"role": "user", "content": request.question})
        history.append({"role": "assistant", "content": final_reply})
        conversation_store[convo_id] = history[-10:]

        # Update state
        update_conversation_state(
            convo_id=convo_id,
            intent=intent,
            tag=tag,
            collection=None,
            final_reply=final_reply,
            raw_user_question=request.question
        )

        return ChatResponse(
            reply=final_reply,
            history=conversation_store[convo_id],
            conversation_id=convo_id,
            query_type=intent,
            data_sources=[tag] if tag else []
        )

    elif intent == "mongo_query":
        used_sources = []
        # shared tiny history block
        mini_ctx = "\n".join([
            f"Prev Q: {m['content']}" if m['role']=='user' else f"Prev A: {m['content']}"
            for m in history[-4:]
        ])

        if request.tags:
            all_contexts = []

            for tag_obj in request.tags:
                a_tag = tag_obj.tag
                coll = TAG_COLLECTION_MAP.get(a_tag)
                if not coll:
                    continue

                context_info = {
                    "start_date": tag_obj.start_date,
                    "end_date": tag_obj.end_date,
                    "collection": coll,
                    "conversation_id": convo_id,
                    "prior_qa": mini_ctx[-1200:]
                }
                context_info = inherit_context_for_followup(convo_id, request.question, context_info)

                mongo_query = generate_ai_mongo_query_with_fallback(a_tag, username, context_info)
                personal_data = fetch_personal_data(mongo_query, username)
                summary = build_comprehensive_context(personal_data, username)

                if summary and "No recent data" not in summary:
                    all_contexts.append(f"🗂️ **{a_tag.title()} Summary:**\n{summary}")
                    used_sources.append(a_tag)

            if all_contexts:
                context_reply = apply_personality("\n\n".join(all_contexts), "friendly")

        else:
            context_info = {
                "date": request.date,
                "start_date": request.start_date,
                "end_date": request.end_date,
                "conversation_id": convo_id
            }

            if not request.start_date and not request.end_date:
                context_info.update(extract_date_context(request.question))

            # Detect or inherit collection
            detected_collection = detect_collection_from_query(request.question)
            nudged = nudge_metric_from_last_topic(convo_id, request.question)
            collection = nudged or detected_collection or collection
            context_info["collection"] = collection

            print("🔍 Processing MongoDB query with context:", context_info)

            # add mini context & inheritance
            context_info["prior_qa"] = mini_ctx[-1200:]
            context_info = inherit_context_for_followup(convo_id, request.question, context_info)

            mongo_query = generate_ai_mongo_query_with_fallback(request.question, username, context_info)
            print(f"🧠 AI Mongo Query:\n{json.dumps(mongo_query, indent=2)}")

            personal_data = fetch_personal_data(mongo_query, username)
            print(f"📦 MongoDB Results:\n{json.dumps(personal_data, indent=2, default=str)}")

            personal_context = build_comprehensive_context(personal_data, username)
            print("Personal_Context:", personal_context)

            if personal_context and personal_context.strip() != "No recent data available for your query.":
                context_reply = apply_personality(personal_context, "friendly")
                print("context_reply", context_reply)
            else:
                context_reply = apply_personality(
                    "I couldn't find any recent data for that. Is your device or tracker synced properly? Let me know if you'd like help troubleshooting. 😊",
                    "friendly"
                )

    elif intent == "knowledge_query":
        print("🔄 Searching knowledge base...")
        try:
            start_time = time.perf_counter()

            # Seed KB search with themed prior mini context
            mini_ctx = "\n".join([
                f"Prev Q: {m['content']}" if m['role']=='user' else f"Prev A: {m['content']}"
                for m in history[-4:]
            ])
            search_prompt = f"{mini_ctx[-800:]}\n\nCurrent: {request.question}".strip()

            best_index_name = pick_best_index(search_prompt)  # fast + cached

            if best_index_name:
                index_path = os.path.join(FAISS_FOLDER_PATH, best_index_name)
                if (
                    os.path.exists(os.path.join(index_path, "index.faiss"))
                    and os.path.exists(os.path.join(index_path, "index.pkl"))
                ):
                    print(f"🔍 Querying FAISS: {best_index_name}")
                    kb_snippets = query_documents(search_prompt, index_path)

                    if kb_snippets:
                        context_reply = apply_personality("\n\n".join(kb_snippets), "friendly")
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

    # -------------------------
    # Final reply selection & persistence
    # -------------------------
    if intent == "mongo_query" and context_reply:
        final = context_reply
    elif intent == "knowledge_query" and context_reply:
        final = context_reply  # prefer KB context for knowledge queries
    else:
        final = raw_reply or "I'm here to help!"

    # Optional LIGHT styling without LLM (fast)
    def style_light(txt: str) -> str:
        return txt if txt.startswith(("✅","📝","💡","⚠️")) else f"💡 {txt}"

    final_reply = style_light(final)

    # Save to rolling history
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

    # Update state for continuity on next turn
    update_conversation_state(
        convo_id=convo_id,
        intent=intent,
        tag=tag,
        collection=collection,
        final_reply=final_reply,
        raw_user_question=request.question
    )

    return ChatResponse(
        reply=final_reply,
        history=recent['history'],
        conversation_id=convo_id,
        query_type=intent,
        data_sources=[tag] if tag else []
    )

# -------------------------
# FastAPI route
# -------------------------

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest, token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid token")
    return handle_user_message(req, username)
