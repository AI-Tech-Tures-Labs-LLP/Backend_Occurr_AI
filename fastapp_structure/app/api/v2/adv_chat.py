

# from fastapi import APIRouter, Depends, HTTPException, Path
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
# from app.utils.optimized_code_rag import load_faiss_index,query_documents
# from app.core.advance_chatbot import normalize, apply_personality
# from app.db.notification_model import notifications_collection
# from dotenv import load_dotenv
# from groq import Groq

# load_dotenv()

# router = APIRouter()
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# # Initialize OpenAI client

# FAISS_FOLDER_PATH = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "faiss_indexes")
# )


# client = Groq(
#     api_key=os.getenv("OPENAI_API_KEY"),
#     # base_url=os.getenv("OPENAI_API_BASE_URL")
# )
# llm = ChatOpenAI(
#     api_key=os.getenv("OPENAI_API_KEY"),
#     base_url=os.getenv("OPENAI_API_BASE_URL"),
#     model=os.getenv("OPENAI_API_MODEL")
# )

# loaded_indexes = {}

# # Pydantic models
# class ChatRequest(BaseModel):
#     question: str
#     tags: Optional[List[str]] = None
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





# from sentence_transformers import SentenceTransformer, util

# topic_model = SentenceTransformer("all-MiniLM-L6-v2")  # local, free, fast

# def load_index_descriptions_from_folders(base_path: str) -> dict:
#     """
#     Use FAISS folder names as natural descriptions (underscores replaced with spaces).
#     """
#     descriptions = {}
#     for folder in os.listdir(base_path):
#         folder_path = os.path.join(base_path, folder)
#         if os.path.isdir(folder_path):
#             descriptions[folder] = folder.replace("_", " ").lower()
#     return descriptions

# def find_best_matching_index(query: str, index_descriptions: dict) -> str:
#     start_time = time.perf_counter()
#     query_embedding = topic_model.encode(query, convert_to_tensor=True)
#     best_index = None
#     best_score = -1

#     for index_name, description in index_descriptions.items():
#         desc_embedding = topic_model.encode(description, convert_to_tensor=True)
#         score = util.pytorch_cos_sim(query_embedding, desc_embedding).item()
#         if score > best_score:
#             best_score = score
#             best_index = index_name
    
#     end_time = time.perf_counter()
#     elapsed_time = end_time - start_time
#     print(f"⏱️ Elapsed time for finding best matching index: {elapsed_time:.4f} seconds")
#     print(f"🔍 Best match: {best_index} (score: {best_score:.4f})")
#     print(f"🔍 Worst match: {index_name} (score: {score:.4f})")
#     return best_index


# # ============================================================================
# # QUERY TYPE DETECTION
# # ============================================================================

# def detect_query_type(question: str, username: str) -> tuple[str, List[str]]:
#     """
#     Detect if the query requires MongoDB data, knowledge base, or both
#     Returns: (query_type, data_sources)
#     """
#     question_lower = question.lower()
    
#     # General health information indicators
#     general_keywords = [
#         'what is', 'how does', 'why does', 'explain', 'tell me about',
#         'definition', 'meaning', 'symptoms', 'causes', 'treatment',
#         'normal range', 'healthy', 'should be', 'recommended',"why i'm"
#     ]
    
#     # Personal data indicators
#     personal_keywords = [
#         'my', 'i', 'me', 'mine', "i'm", 'today', 'yesterday', 'last week', 
#         'this week', 'last month', 'this month', 'journal', 'diary', 'note'
#     ]
    
#     # Health metrics that could be personal or general
#     health_metrics = [
#         'heart rate', 'heartrate', 'pulse', 'bpm', 'steps', 'walking', 'activity',
#         'spo2', 'oxygen', 'blood oxygen', 'sleep', 'calories', 'blood pressure', 
#         'pressure', 'weight', 'bmi'
#     ]
    
#     # Specific data types
#     personal_data_keywords = [
#         'journal', 'diary', 'entries', 'mood', 'feelings', 'alert', 'warning', 
#         'notification', 'reminder', 'summary'
#     ]
    
#     # Check for different types of queries
#     has_personal = any(k in question_lower for k in personal_keywords)
#     has_health_metric = any(k in question_lower for k in health_metrics)
#     has_personal_data = any(k in question_lower for k in personal_data_keywords)
#     has_general = any(k in question_lower for k in general_keywords)
    
#     # Special handling for step-related queries
#     is_step_query = any(k in question_lower for k in ['steps', 'step', 'walked', 'walking'])
    
#     print(f"🔍 Query Analysis - Personal: {has_personal}, Health Metric: {has_health_metric}, "
#           f"Personal Data: {has_personal_data}, General: {has_general}, Steps: {is_step_query}")
    
#     # Determine query type and data sources
#     if has_personal_data or (has_personal and has_health_metric) or is_step_query:
#         query_type = "personal"
#         data_sources = ["mongodb"]
#     elif has_general and not has_personal:
#         query_type = "general"
#         data_sources = ["knowledge_base"]
#     elif has_health_metric and has_general:
#         query_type = "hybrid"
#         data_sources = ["mongodb", "knowledge_base"]
#     elif has_health_metric:
#         query_type = "personal" if has_personal else "general"
#         data_sources = ["mongodb"] if has_personal else ["knowledge_base"]
#     else:
#         query_type = "general"
#         data_sources = ["knowledge_base"]
    
#     print(f"🎯 Detected - Type: {query_type}, Sources: {data_sources}")
#     return query_type, data_sources

# def extract_date_context(question: str) -> dict:
#     """Extract date context from natural language"""
#     question_lower = question.lower()
#     now = datetime.now()
    
#     if "yesterday" in question_lower:
#         yesterday = now - timedelta(days=1)
#         return {
#             "start_date": yesterday.strftime("%Y-%m-%dT00:00:00.000Z"),
#             "end_date": now.strftime("%Y-%m-%dT00:00:00.000Z"),
#             "date": yesterday.strftime("%Y-%m-%d")
#         }
#     elif "today" in question_lower:
#         return {
#             "start_date": now.strftime("%Y-%m-%dT00:00:00.000Z"),
#             "end_date": (now + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00.000Z"),
#             "date": now.strftime("%Y-%m-%d")
#         }
#     elif "last week" in question_lower:
#         week_start = now - timedelta(days=now.weekday() + 7)
#         week_end = week_start + timedelta(days=7)
#         return {
#             "start_date": week_start.strftime("%Y-%m-%dT00:00:00.000Z"),
#             "end_date": week_end.strftime("%Y-%m-%dT00:00:00.000Z")
#         }
    
#     return {}

# # ============================================================================
# # MONGODB QUERY GENERATION (Fixed Version)
# # ============================================================================

# def clean_mongodb_response(generated: str) -> str:
#     """Clean up AI-generated MongoDB query to make it valid JSON"""
    
#     # Remove markdown formatting if present
#     if generated.startswith("```"):
#         generated = generated.strip("`").strip()
#         if generated.lower().startswith("json"):
#             generated = generated[4:].strip()
    
#     # Replace single quotes with double quotes
#     generated = generated.replace("'", '"')
    
#     # Fix ISODate() functions - convert to ISO string format
#     iso_date_pattern = r'ISODate\("([^"]+)"\)'
#     generated = re.sub(iso_date_pattern, r'"\1"', generated)
    
#     # Fix ObjectId() functions if any
#     object_id_pattern = r'ObjectId\("([^"]+)"\)'
#     generated = re.sub(object_id_pattern, r'"\1"', generated)
    
#     # Remove any trailing commas before closing braces/brackets
#     generated = re.sub(r',(\s*[}\]])', r'\1', generated)
    
#     return generated

# def convert_iso_string_to_datetime(value):
#     """Convert ISO string to datetime object if it's a valid ISO date string"""
#     if isinstance(value, str) and re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{3})?Z?$', value):
#         try:
#             # Handle both with and without milliseconds
#             if value.endswith('Z'):
#                 value = value[:-1] + '+00:00'
#             return datetime.fromisoformat(value.replace('Z', '+00:00'))
#         except ValueError:
#             return value
#     return value

# def convert_iso_dates_in_dict(obj: dict) -> dict:
#     """Recursively convert ISO date strings to datetime objects"""
#     if not isinstance(obj, dict):
#         return obj
    
#     result = {}
#     for key, value in obj.items():
#         if isinstance(value, dict):
#             result[key] = convert_iso_dates_in_dict(value)
#         elif isinstance(value, list):
#             result[key] = [convert_iso_dates_in_dict(item) if isinstance(item, dict) else convert_iso_string_to_datetime(item) for item in value]
#         else:
#             result[key] = convert_iso_string_to_datetime(value)
    
#     return result

# def convert_iso_dates_in_query(query_list: list) -> list:
#     """Convert ISO date strings to datetime objects in MongoDB query"""
#     processed_query = []
    
#     for stage in query_list:
#         processed_stage = convert_iso_dates_in_dict(stage)
#         processed_query.append(processed_stage)
    
#     return processed_query

# # def generate_fallback_query(username: str, context: dict) -> Dict[str, Any]:
# #     """Generate a simple fallback query when AI generation fails"""
    
# #     fallback_query = {
# #         "health_data": [],
# #         "journal": [],
# #         "alerts": [],
# #         "user_notifications": []
# #     }
    
# #     # Basic match filter with username
# #     base_match = {"username": username}
    
# #     # Add date range if provided
# #     if context.get('start_date') and context.get('end_date'):
# #         try:
# #             start_date = datetime.fromisoformat(context['start_date'].replace('Z', '+00:00'))
# #             end_date = datetime.fromisoformat(context['end_date'].replace('Z', '+00:00'))
            
# #             date_filter = {
# #                 "timestamp": {
# #                     "$gte": start_date,
# #                     "$lt": end_date
# #                 }
# #             }
# #             base_match.update(date_filter)
# #         except ValueError:
# #             print("⚠️ Invalid date format in context, using basic query")
    
# #     # Generate basic queries for each collection
# #     fallback_query["health_data"] = [{"$match": base_match}]
# #     fallback_query["journal"] = [{"$match": base_match}]
# #     fallback_query["alerts"] = [{"$match": base_match}]
    
# #     return fallback_query


# def generate_fallback_query(username: str, context: dict) -> Dict[str, Any]:
#     """Generate a simple fallback query when AI generation fails"""

#     fallback_query = {
#         "health_data": [],
#         "journal": [],
#         "alerts": [],
#         "user_notifications": []
#     }

#     # Basic match filter with username
#     base_match = {"username": username}

#     # ✅ 1. Prioritize date-only match
#     if context.get("date"):
#         date_expr_match = {
#             "username": username,
#             "$expr": {
#                 "$eq": [
#                     { "$dateToString": { "format": "%Y-%m-%d", "date": "$timestamp" } },
#                     context["date"]
#                 ]
#             }
#         }

#         fallback_query["health_data"] = [{"$match": date_expr_match}]
#         fallback_query["journal"] = [{"$match": date_expr_match}]
#         fallback_query["alerts"] = [{"$match": date_expr_match}]
#         return fallback_query

#     # 2. Range filter (used only if no specific `date` is present)
#     if context.get('start_date') and context.get('end_date'):
#         try:
#             start_date = datetime.fromisoformat(context['start_date'].replace('Z', '+00:00'))
#             end_date = datetime.fromisoformat(context['end_date'].replace('Z', '+00:00'))

#             date_filter = {
#                 "timestamp": {
#                     "$gte": start_date,
#                     "$lt": end_date
#                 }
#             }
#             base_match.update(date_filter)
#         except ValueError:
#             print("⚠️ Invalid date format in context, using basic query")

#     fallback_query["health_data"] = [{"$match": base_match}]
#     fallback_query["journal"] = [{"$match": base_match}]
#     fallback_query["alerts"] = [{"$match": base_match}]
    
#     return fallback_query


# def generate_ai_mongo_query(question: str, username: str, context: dict) -> Dict[str, Any]:
#     """Generate MongoDB queries using AI with improved error handling"""
    
#     system_prompt = (
#         """You are an assistant that writes MongoDB queries. Return a valid JSON object where keys are 'health_data', 'journal', 'alerts', or 'user_notifications'.
# Each value must be a list (aggregation pipeline). Use [{ "$match": {...} }] format even for simple filters.

# IMPORTANT: Use only valid JSON syntax. For dates, use ISO string format like "2025-06-16T00:00:00.000Z" - DO NOT use ISODate() function.

# Collections schema:
# - 'health_data': username, metric, value, timestamp, created_at
# - 'alerts': username, metric, value, timestamp, message, responded, created_at  
# - 'journal': username, timestamp, mood, food_intake, sleep, personal, work_or_study, extra_note
# - 'users': has embedded notifications array with fields: type, message, timestamp, read

# Always filter by username. Use timestamp with "$gte" and "$lt" for date ranges. For aggregation queries, add "$group" stage.

# Health metrics mapping:
# - "steps" → metric: "steps"
# - "heart rate", "heartrate", "pulse", "bpm" → metric: "heartrate" 
# - "spo2", "oxygen" → metric: "spo2"
# - "sleep" → metric: "sleep"
# - "calories" → metric: "calories"

# Aggregation types:
# - "total", "sum" → {"$group": {"_id": null, "total_value": {"$sum": "$value"}}}
# - "average", "avg", "mean" → {"$group": {"_id": null, "average_value": {"$avg": "$value"}}}
# - "minimum", "min", "lowest" → {"$group": {"_id": null, "min_value": {"$min": "$value"}}}
# - "maximum", "max", "highest" → {"$group": {"_id": null, "max_value": {"$max": "$value"}}}

# Examples:

# 1. Total steps yesterday:
# {
#   "health_data": [
#     {"$match": {"username": "user123", "metric": "steps", "timestamp": {"$gte": "2025-06-19T00:00:00.000Z", "$lt": "2025-06-20T00:00:00.000Z"}}},
#     {"$group": {"_id": null, "total_steps": {"$sum": "$value"}}}
#   ]
# }

# 2. Average heart rate:
# {
#   "health_data": [
#     {"$match": {"username": "user123", "metric": "heartrate"}},
#     {"$group": {"_id": null, "average_heartrate": {"$avg": "$value"}}}
#   ]
# }

# 3. Maximum heart rate today:
# {
#   "health_data": [
#     {"$match": {"username": "user123", "metric": "heartrate", "timestamp": {"$gte": "2025-06-20T00:00:00.000Z", "$lt": "2025-06-21T00:00:00.000Z"}}},
#     {"$group": {"_id": null, "max_heartrate": {"$max": "$value"}}}
#   ]
# }"""
#     )

#     user_input = (
#         f"User question: {question}\n"
#         f"Username: {username}\n"
#         f"Date: {context.get('date', '')}\n"
#         f"Start Date: {context.get('start_date', '')}\n"
#         f"End Date: {context.get('end_date', '')}\n\n"
#         f"For questions about 'yesterday', use date range from yesterday 00:00:00 to today 00:00:00.\n"
#         f"Current date context: Today is {datetime.now().strftime('%Y-%m-%d')}, so yesterday was {(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')}."
#     )

#     try:
#         response = client.chat.completions.create(
#             model=os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo"),
#             temperature=0.3,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_input}
#             ]
#         )
#         generated = response.choices[0].message.content.strip()

#         if not generated:
#             print("⚠️ Empty response from GPT for query generation.")
#             return {}

#         # Clean up the response
#         generated = clean_mongodb_response(generated)
        
#         print(f"🔍 Generated JSON: {generated}")
        
#         try:
#             return json.loads(generated)
#         except json.JSONDecodeError as err:
#             print(f"❌ JSON decode failed: {err}")
#             print(f"Raw output: {generated}")
#             return {}

#     except Exception as e:
#         print(f"❌ AI Mongo query generation failed: {e}")
#         return {}

# def generate_ai_mongo_query_with_fallback(question: str, username: str, context: dict) -> Dict[str, Any]:
#     """Main function with fallback support"""
    
#     # Try AI generation first
#     result = generate_ai_mongo_query(question, username, context)
    
#     # If AI generation fails, use fallback
#     if not result:
#         print("🔄 Using fallback query generation")
#         result = generate_fallback_query(username, context)
    
#     return result

# # ============================================================================
# # DATA FETCHING
# # ============================================================================

# def fetch_personal_data(queries: Dict, username: str) -> Dict[str, List]:
#     """Enhanced version with better error handling and date conversion"""
#     results = {"health_data": [], "journal": [], "alerts": [], "user_notifications": []}
#     collections = {
#         "health_data": health_data_collection,
#         "journal": journals_collection,
#         "alerts": alert_collection
#     }

#     for key, query in queries.items():
#         try:
#             print(f"🔍 Processing {key} query: {query}")
            
#             # if key == "user_notifications":
#             #     user = users_collection.find_one({"username": username})
#             #     if user and "notifications" in user:
#             #         results[key] = user["notifications"]
#             #         print(f"✅ Found {len(results[key])} notifications")
#             #     else:
#             #         print("❌ No notifications found")
#             if key == "user_notifications":
            
#                 results[key] = list(notifications_collection.find(
#                     {"username": username},
#                     {"_id": 0}
#                 ))
#                 # print(f"✅ Found {len(results[key])} user notifications")

                    
#             elif isinstance(query, list):
#                 # Convert ISO date strings to datetime objects in the query
#                 processed_query = convert_iso_dates_in_query(query)
#                 print(f"🔄 Processed aggregation query: {processed_query}")
                
#                 collection = collections[key]
#                 cursor = collection.aggregate(processed_query)
#                 results[key] = list(cursor)
#                 print(f"✅ Aggregation returned {len(results[key])} results")
                
#                 # Debug: Print first few results
#                 if results[key]:
#                     print(f"📋 Sample result: {results[key][0]}")
#                 else:
#                     print("⚠️ No results from aggregation - checking if data exists...")
#                     # Test query without aggregation
#                     if processed_query and processed_query[0].get("$match"):
#                         test_query = processed_query[0]["$match"]
#                         test_results = list(collection.find(test_query).limit(5))
#                         print(f"🔍 Test query found {len(test_results)} raw documents")
#                         if test_results:
#                             print(f"📋 Sample raw document: {test_results[0]}")
                
#             elif isinstance(query, dict):
#                 collection = collections[key]
#                 if "$match" in query:
#                     # Wrap single $match into aggregation pipeline
#                     processed_query = convert_iso_dates_in_query([query])
#                     results[key] = list(collection.aggregate(processed_query))
#                 else:
#                     processed_query = convert_iso_dates_in_dict(query)
#                     results[key] = list(collection.find(processed_query))
                
#                 print(f"✅ Query returned {len(results[key])} results")
                
#         except Exception as e:
#             print(f"❌ Error in {key}: {e}")
#             import traceback
#             traceback.print_exc()
#             results[key] = []
    
#     return results

# # ============================================================================
# # CONTEXT BUILDING
# # ============================================================================


# from datetime import datetime
# from typing import Dict, List, Any, Optional
# from enum import Enum

# # ---------- Metric normalization ----------
# class Metric(str, Enum):
#     STEPS = "steps"
#     HEART_RATE = "heart_rate"
#     SPO2 = "spo2"
#     SLEEP = "sleep"
#     CALORIES = "calories"
#     UNKNOWN = "unknown"

# _METRIC_ALIASES = {
#     # steps
#     "steps": Metric.STEPS, "step": Metric.STEPS,
#     # heart rate
#     "heart rate": Metric.HEART_RATE, "heart_rate": Metric.HEART_RATE,
#     "heartrate": Metric.HEART_RATE, "hr": Metric.HEART_RATE,
#     "bpm": Metric.HEART_RATE, "heartrates": Metric.HEART_RATE,
#     "heartratebpm": Metric.HEART_RATE, "heartrate_bpm": Metric.HEART_RATE,
#     "heartrate (bpm)": Metric.HEART_RATE, "heartrateb p m": Metric.HEART_RATE,
#     "heartratevalue": Metric.HEART_RATE, "heartrate_value": Metric.HEART_RATE,
#     "heartrateavg": Metric.HEART_RATE, "heartrate_avg": Metric.HEART_RATE,
#     "heartrateaverage": Metric.HEART_RATE, "heartrate_average": Metric.HEART_RATE,
#     "heartrate_max": Metric.HEART_RATE, "heartrate_min": Metric.HEART_RATE,
#     "heartratepeak": Metric.HEART_RATE, "heartRate": Metric.HEART_RATE,  # your key
#     # spo2
#     "spo2": Metric.SPO2, "sp o2": Metric.SPO2, "spO2".lower(): Metric.SPO2,
#     "oxygen": Metric.SPO2, "blood oxygen": Metric.SPO2, "blood_oxygen": Metric.SPO2,
#     # sleep
#     "sleep": Metric.SLEEP, "sleep_hours": Metric.SLEEP,
#     "sleepminutes": Metric.SLEEP, "sleep_minutes": Metric.SLEEP,
#     # calories
#     "calories": Metric.CALORIES, "kcal": Metric.CALORIES, "energy": Metric.CALORIES,
#     "active_calories": Metric.CALORIES, "resting_calories": Metric.CALORIES,
# }

# def normalize_metric(m: Any) -> Metric:
#     if not m:
#         return Metric.UNKNOWN
#     key = str(m).strip().lower().replace("-", " ").replace("_", " ")
#     return _METRIC_ALIASES.get(key, Metric.UNKNOWN)


# # ---------- Builder ----------
# def build_comprehensive_context(data: Dict[str, List], username: str) -> str:
#     """
#     Build human-friendly context from fetched data.
#     Supports aggregation documents and raw record lists for:
#       - steps, heart_rate, spo2, sleep, calories
#     Also summarizes alerts, journal, and notifications.
#     """
#     context: List[str] = []
#     print(f"🔍 Building context for user: {username} , collection: {data.keys()}")

#     # ---------------- Health data analysis ----------------
#     health = data.get("health_data") or []
#     if health:
#         first_item = health[0]

#         # ---- Aggregation-style payloads (totals/averages/min/max) ----
#         # ---- Aggregation-style payloads (totals/averages/min/max) ----
#         if isinstance(first_item, dict) and any(
#             k in first_item
#             for k in [
#                 # existing keys you already had...
#                 "total_steps","total_value","average_value",
#                 "average_heartrate","min_value","max_value",
#                 "min_heartrate","max_heartrate","highest_heartrate","lowest_heartrate",
#                 # sleep (hours)
#                 "total_sleep_hours","average_sleep_hours","min_sleep_hours","max_sleep_hours",
#                 # NEW: sleep (minutes) like your result
#                 "total_sleep","average_sleep","min_sleep","max_sleep",
#                 # calories
#                 "total_calories","average_calories","min_calories","max_calories",
#                 # spo2
#                 "average_spo2","min_spo2","max_spo2",
#             ]
#         ):
#             # ---------------- STEPS ----------------
#             if "total_steps" in first_item:
#                 total = int(round(first_item["total_steps"]))
#                 context.append(f"📊 **Total Steps**: You walked **{total:,} steps** during the requested period! 🚶‍♀️")

#             # ---------------- HEART RATE ----------------
#             elif "average_heartrate" in first_item:
#                 avg = round(float(first_item["average_heartrate"]), 1)
#                 context.append(f"💓 **Average Heart Rate**: **{avg} bpm** during the requested period.")
#                 if avg < 60:
#                     context.append("📘 This falls in bradycardia (<60 bpm). If you notice symptoms, consider consulting a professional.")
#                 elif avg > 100:
#                     context.append("📘 This falls in tachycardia (>100 bpm). If you notice symptoms, consider consulting a professional.")
#                 else:
#                     context.append("✅ Within typical resting range (60–100 bpm).")
#             elif "max_heartrate" in first_item or "highest_heartrate" in first_item:
#                 context.append(f"📈 **Maximum Heart Rate**: **{float(first_item['max_heartrate']):.0f} bpm**.")
#             elif "min_heartrate" in first_item or "lowest_heartrate" in first_item:
#                 context.append(f"📉 **Minimum Heart Rate**: **{float(first_item['min_heartrate']):.0f} bpm**.")

#             # ---------------- SLEEP (HOURS) ----------------
#             elif "total_sleep_hours" in first_item:
#                 total_sleep = round(float(first_item["total_sleep_hours"]), 2)
#                 context.append(f"😴 **Total Sleep**: **{total_sleep} hours**.")
#             elif "average_sleep_hours" in first_item:
#                 avg_sleep = round(float(first_item["average_sleep_hours"]), 2)
#                 tip = " (below 7–9h guideline)." if avg_sleep < 7 else (" (above typical 7–9h guideline)." if avg_sleep > 9 else "")
#                 context.append(f"😴 **Average Sleep**: **{avg_sleep} h/night**{tip}")
#             elif "min_sleep_hours" in first_item:
#                 context.append(f"😴 **Shortest Sleep**: **{float(first_item['min_sleep_hours']):.2f} h**.")
#             elif "max_sleep_hours" in first_item:
#                 context.append(f"😴 **Longest Sleep**: **{float(first_item['max_sleep_hours']):.2f} h**.")

#             # ---------------- SLEEP (MINUTES) — YOUR CASE ----------------
#             elif "average_sleep" in first_item or "total_sleep" in first_item or "min_sleep" in first_item or "max_sleep" in first_item:
#                 total_min = first_item.get("total_sleep")
#                 avg_min = first_item.get("average_sleep")
#                 min_min = first_item.get("min_sleep")
#                 max_min = first_item.get("max_sleep")

#                 def to_hours(m): 
#                     return None if m is None else float(m)/60.0

#                 # convert when present
#                 total_h = to_hours(total_min)
#                 avg_h = to_hours(avg_min)
#                 min_h = to_hours(min_min)
#                 max_h = to_hours(max_min)

#                 # Build sentences only for provided stats
#                 if total_h is not None:
#                     context.append(f"😴 **Total Sleep**: **{total_h:.2f} h** (converted from minutes).")
#                 if avg_h is not None:
#                     tip = " (below 7–9h guideline)." if avg_h < 7 else (" (above typical 7–9h guideline)." if avg_h > 9 else "")
#                     context.append(f"😴 **Average Sleep**: **{avg_h:.2f} h/night**{tip} (converted).")
#                 if min_h is not None:
#                     context.append(f"😴 **Shortest Sleep**: **{min_h:.2f} h** (converted).")
#                 if max_h is not None:
#                     context.append(f"😴 **Longest Sleep**: **{max_h:.2f} h** (converted).")

#             # ---------------- CALORIES ----------------
#             elif "total_calories" in first_item:
#                 context.append(f"🔥 **Total Calories**: **{int(round(float(first_item['total_calories']))):,} kcal**.")
#             elif "average_calories" in first_item:
#                 context.append(f"🔥 **Average Daily Calories**: **{int(round(float(first_item['average_calories']))):,} kcal/day**.")
#             elif "min_calories" in first_item:
#                 context.append(f"🔥 **Lowest Daily Calories**: **{int(round(float(first_item['min_calories']))):,} kcal**.")
#             elif "max_calories" in first_item:
#                 context.append(f"🔥 **Highest Daily Calories**: **{int(round(float(first_item['max_calories']))):,} kcal**.")

#             # ---------------- SpO₂ ----------------
#             elif "average_spo2" in first_item:
#                 avg_spo2 = float(first_item["average_spo2"])
#                 flag = " ⚠️ Low average." if avg_spo2 < 95 else ""
#                 context.append(f"🫁 **Average SpO₂**: **{avg_spo2:.1f}%**.{flag}")
#             elif "min_spo2" in first_item:
#                 context.append(f"🫁 **Lowest SpO₂**: **{float(first_item['min_spo2']):.1f}%**")
#             elif "max_spo2" in first_item:
#                 context.append(f"🫁 **Highest SpO₂**: **{float(first_item['max_spo2']):.1f}%**")

#             # ---------------- Generic fallback ----------------
#             elif "total_value" in first_item:
#                 context.append(f"📊 **Total**: {first_item['total_value']:,}")
#             elif "average_value" in first_item:
#                 context.append(f"📊 **Average**: {round(float(first_item['average_value']), 1)}")
#             elif "min_value" in first_item:
#                 context.append(f"📊 **Minimum**: {first_item['min_value']}")
#             elif "max_value" in first_item:
#                 context.append(f"📊 **Maximum**: {first_item['max_value']}")


#         # ---- Raw record list (per-sample values) ----
#         else:
#             # Extract numeric values
#             values = [d.get("value") for d in health if isinstance(d.get("value"), (int, float))]
#             raw_metric = health[0].get("metric") or health[0].get("type") or "unknown"
#             metric = normalize_metric(raw_metric)

#             if values:
#                 total = float(sum(values))
#                 avg = total / len(values)
#                 min_val = float(min(values))
#                 max_val = float(max(values))

#                 # Group by day for simple 3-day trend
#                 date_groups: Dict[str, List[float]] = {}
#                 for d in health:
#                     ts = d.get("timestamp")
#                     if isinstance(ts, datetime):
#                         day = ts.strftime("%Y-%m-%d")
#                     else:
#                         day = str(ts)[:10]
#                     if isinstance(d.get("value"), (int, float)):
#                         date_groups.setdefault(day, []).append(float(d["value"]))

#                 trend = ""
#                 if len(date_groups) >= 3:
#                     sorted_days = sorted(date_groups.items())[-3:]
#                     day_avgs = [sum(vals) / max(len(vals), 1) for _, vals in sorted_days]
#                     if day_avgs[0] < day_avgs[1] < day_avgs[2]:
#                         trend = "📈 You've shown a 3-day improvement trend!"
#                     elif day_avgs[0] > day_avgs[1] > day_avgs[2]:
#                         trend = "📉 Your recent data suggests a slight decline."

#                 # Metric-specific summaries
#                 if metric == Metric.STEPS:
#                     summary = (
#                         f"You recorded a total of **{int(round(total)):,} steps** with {len(values)} readings. "
#                         f"Average: {avg:.0f} steps per reading. {trend}"
#                     )
#                     if total >= 10000:
#                         summary += " 🎉 Excellent! You hit the 10,000+ steps goal!"
#                     elif total >= 7000:
#                         summary += " 👍 Great job staying active!"
#                     else:
#                         summary += " 💪 Every step counts — keep it up!"
#                     context.append(f"📊 **Steps Summary**: {summary}")

#                 elif metric == Metric.HEART_RATE:
#                     summary = f"Average **{avg:.0f} bpm**, range **{min_val:.0f}-{max_val:.0f} bpm**. {trend}"
#                     context.append(f"💓 **Heart Rate Summary**: {summary}")

#                 elif metric == Metric.SPO2:
#                     flag = ""
#                     if avg < 95 or min_val < 90:
#                         flag = " ⚠️ Consider following up if low values persist."
#                     summary = f"Average **{avg:.1f}%**, range **{min_val:.1f}–{max_val:.1f}%**.{flag} {trend}"
#                     context.append(f"🫁 **SpO₂ Summary**: {summary}")

#                 elif metric == Metric.SLEEP:
#                     # Assume hours; if max suggests minutes (e.g., >24), convert
#                     converted = False
#                     if max_val > 24:
#                         values_h = [v / 60.0 for v in values]
#                         total = sum(values_h)
#                         avg = total / len(values_h)
#                         min_val = min(values_h)
#                         max_val = max(values_h)
#                         converted = True

#                     tip = ""
#                     if avg < 7:
#                         tip = " (below 7–9h guideline)."
#                     elif avg > 9:
#                         tip = " (above typical 7–9h guideline)."

#                     pretty = (
#                         f"Total **{total:.2f} h**, avg **{avg:.2f} h**, "
#                         f"range **{min_val:.2f}–{max_val:.2f} h**. {trend}{tip}"
#                     )
#                     if converted:
#                         pretty += " ⤴️ Converted from minutes → hours."
#                     context.append(f"😴 **Sleep Summary**: {pretty}")

#                 elif metric == Metric.CALORIES:
#                     summary = (
#                         f"Total **{int(round(total)):,} kcal**, avg **{int(round(avg)):,} kcal/read**. "
#                         f"Range **{int(round(min_val)):,}–{int(round(max_val)):,} kcal**. {trend}"
#                     )
#                     context.append(f"🔥 **Calories Summary**: {summary}")

#                 else:
#                     summary = f"Total **{total:.1f}**, avg **{avg:.1f}**, range **{min_val:.1f}-{max_val:.1f}**. {trend}"
#                     title = (raw_metric or "unknown").replace("_", " ").title()
#                     context.append(f"📊 **{title} Summary**: {summary}")

#     # ---------------- Alerts analysis ----------------
#     if data.get("alerts"):
#         alerts = []
#         for d in data["alerts"]:
#             ts = d.get("timestamp")
#             ts_str = ts.strftime("%Y-%m-%d") if isinstance(ts, datetime) else str(ts)[:10]
#             msg = d.get("message", "")
#             alerts.append(f"{ts_str}: {msg}")
#         if alerts:
#             context.append("🚨 **Recent Alerts**:\n" + "\n".join(alerts[:5]))

#     # ---------------- Journal entries analysis ----------------
#     if data.get("journal"):
#         tag_map = {
#             "food_intake": "Food",
#             "sleep": "Sleep",
#             "mood": "Mood",
#             "meditation": "Meditation",
#             "personal": "Personal Note",
#             "work_or_study": "Work/Study",
#             "extra_note": "Note",
#         }

#         tag_summary: Dict[str, List[str]] = {}
#         flattened: List[Dict[str, Any]] = []

#         for doc in data["journal"]:
#             if isinstance(doc, dict) and isinstance(doc.get("entries"), list):
#                 flattened.extend(doc["entries"])
#             else:
#                 flattened.append(doc)

#         for entry in flattened:
#             if isinstance(entry, dict):
#                 if "tag" in entry and "text" in entry:
#                     tag = entry.get("tag", "unknown")
#                     text = entry.get("text", "")
#                 else:
#                     possible_tag = next((key for key in entry if key in tag_map), None)
#                     if not possible_tag:
#                         continue
#                     tag = possible_tag
#                     text = entry.get(tag, "")
#             else:
#                 continue

#             readable = tag_map.get(tag, tag.replace("_", " ").title())
#             tag_summary.setdefault(readable, []).append(text)

#         if tag_summary:
#             lines = ["📝 **Journal Summary**:"]
#             for tag, notes in tag_summary.items():
#                 lines.append(f"• {tag}: {' | '.join(notes)}")
#             context.append("\n".join(lines))

#     # ---------------- Notifications analysis ----------------
#     if isinstance(data.get("user_notifications"), list):
#         unread = [n for n in data["user_notifications"] if not n.get("read", True)]
#         if unread:
#             lines = []
#             for n in unread[:5]:
#                 ts = n.get("timestamp")
#                 ts_str = ts.strftime("%Y-%m-%d") if isinstance(ts, datetime) else str(ts)[:10]
#                 body = n.get("body", "[No message]")
#                 lines.append(f"{ts_str} - {body}")
#             context.append("🔔 **Unread Notifications**:\n" + "\n".join(lines))

#     return "\n\n".join(context) if context else "No recent data available for your query."


# # def build_comprehensive_context(data: Dict[str, List], username: str) -> str:
# #     """Build comprehensive context from fetched data"""
# #     context = []
    
# #     # Health data analysis
# #     if data["health_data"]:
# #         # Check if it's aggregation result (like total steps)
# #         first_item = data["health_data"][0]
        
# #         if isinstance(first_item, dict) and any(key in first_item for key in ['total_steps', 'total_value', 'average_value', 'average_heartrate', 'min_value', 'max_value', 'min_heartrate', 'max_heartrate']):
# #             # Handle aggregation results
# #             if 'total_steps' in first_item:
# #                 total = first_item['total_steps']
# #                 context.append(f"📊 **Total Steps**: You walked **{total:,} steps** during the requested period! 🚶‍♀️")
                
# #             elif 'average_heartrate' in first_item:
# #                 avg = round(first_item['average_heartrate'], 1)
# #                 context.append(f"💓 **Average Heart Rate**: Your average heart rate was **{avg} bpm** during the requested period!")
# #                 if avg < 60:
# #                     context.append("📘 This is considered bradycardia (slow heart rate). Consider consulting a healthcare professional if you experience symptoms.")
# #                 elif avg > 100:
# #                     context.append("📘 This is considered tachycardia (fast heart rate). Consider consulting a healthcare professional if you experience symptoms.")
# #                 else:
# #                     context.append("✅ This is within the normal resting heart rate range (60-100 bpm).")
                    
# #             elif 'max_heartrate' in first_item:
# #                 max_hr = first_item['max_heartrate','highest','high']
# #                 context.append(f"📈 **Maximum Heart Rate**: Your peak heart rate was **{max_hr} bpm** during the requested period!")
                
# #             elif 'min_heartrate' in first_item:
# #                 min_hr = first_item['min_heartrate','lowest','low']
# #                 context.append(f"📉 **Minimum Heart Rate**: Your lowest heart rate was **{min_hr} bpm** during the requested period!")
                
# #             elif 'total_value' in first_item:
# #                 total = first_item['total_value']
# #                 context.append(f"📊 **Total**: {total:,}")
                
# #             elif 'average_value' in first_item:
# #                 avg = round(first_item['average_value'], 1)
# #                 context.append(f"📊 **Average**: {avg}")
                
# #             elif 'min_value' in first_item:
# #                 min_val = first_item['min_value']
# #                 context.append(f"📊 **Minimum**: {min_val}")
                
# #             elif 'max_value' in first_item:
# #                 max_val = first_item['max_value']
# #                 context.append(f"📊 **Maximum**: {max_val}")
                
# #         else:
# #             # Handle individual records
# #             values = [d["value"] for d in data["health_data"] if "value" in d and isinstance(d["value"], (int, float))]
# #             metric = data["health_data"][0].get("metric", "unknown")
            
# #             if values:
# #                 total = sum(values)
# #                 avg = total / len(values)
# #                 min_val = min(values)
# #                 max_val = max(values)
                
# #                 # Group by date for trend analysis
# #                 date_groups = {}
# #                 for d in data["health_data"]:
# #                     ts = d.get("timestamp")
# #                     if isinstance(ts, datetime):
# #                         day = ts.strftime("%Y-%m-%d")
# #                         date_groups.setdefault(day, []).append(d["value"])
                
# #                 # Trend analysis
# #                 trend = ""
# #                 if len(date_groups) >= 3:
# #                     sorted_days = sorted(date_groups.items())[-3:]
# #                     day_avgs = [sum(vals) / len(vals) for _, vals in sorted_days]
# #                     if day_avgs[0] < day_avgs[1] < day_avgs[2]:
# #                         trend = "📈 You've shown a 3-day improvement trend!"
# #                     elif day_avgs[0] > day_avgs[1] > day_avgs[2]:
# #                         trend = "📉 Your recent data suggests a slight decline."
                
# #                 # Build summary based on metric type
# #                 if metric.lower() == "steps":
# #                     summary = f"You recorded a total of **{total:,} steps** with {len(values)} readings. Average: {avg:.0f} steps per reading. {trend}"
# #                     if total >= 10000:
# #                         summary += " 🎉 Excellent! You hit the 10,000+ steps goal!"
# #                     elif total >= 7000:
# #                         summary += " 👍 Great job staying active!"
# #                     else:
# #                         summary += " 💪 Every step counts - keep it up!"
# #                 elif metric.lower() in ["heartrate", "heart_rate"]:
# #                     summary = f"Heart rate readings: Average {avg:.0f} bpm, Range: {min_val}-{max_val} bpm. {trend}"
# #                 else:
# #                     summary = f"{metric.title()}: Total {total}, Average {avg:.1f}, Range: {min_val}-{max_val}. {trend}"
                
# #                 context.append(f"📊 **{metric.title()} Summary**: {summary}")
    
# #     # Alerts analysis
# #     if data["alerts"]:
# #         alerts = [
# #             f"{d.get('timestamp').strftime('%Y-%m-%d') if isinstance(d.get('timestamp'), datetime) else str(d.get('timestamp'))[:10]}: {d.get('message', '')}"
# #             for d in data["alerts"]
# #         ]
# #         context.append("🚨 **Recent Alerts**:\n" + "\n".join(alerts[:5]))
    
# #     # Journal entries analysis
# #     # if data["journal"]:
# #     #     entries = [
# #     #         f"{d.get('timestamp').strftime('%Y-%m-%d') if isinstance(d.get('timestamp'), datetime) else str(d.get('timestamp'))[:10]} | Sleep: {d.get('sleep', 'N/A')} | Note: {d.get('extra_note', 'N/A')}"
# #     #         for d in data["journal"]
# #     #     ]
# #     #     context.append("📝 **Recent Journal Entries**:\n" + "\n".join(entries[:3]))

# #     # if data["journal"]:
# #     #     tag_map = {
# #     #         "food_intake": "Food",
# #     #         "sleep": "Sleep",
# #     #         "mood": "Mood",
# #     #         "meditation": "Meditation",
# #     #         "personal": "Personal Note",
# #     #         "work_or_study": "Work/Study",
# #     #         "extra_note": "Note"
# #     #     }

# #     #     tag_summary = {}

# #     #     for entry in data["journal"]:
# #     #         tag = entry.get("tag", "unknown")
# #     #         readable = tag_map.get(tag, tag.replace("_", " ").title())
# #     #         text = entry.get("text", "")
# #     #         tag_summary.setdefault(readable, []).append(text)

# #     #     summary_lines = ["📝 **Today's Journal Summary**:"]
# #     #     for tag, notes in tag_summary.items():
# #     #         summary_lines.append(f"• {tag}: {' | '.join(notes)}")

# #     #     context.append("\n".join(summary_lines))
    
# #     # 
# #     if data.get("journal"):
# #         tag_map = {
# #             "food_intake": "Food",
# #             "sleep": "Sleep",
# #             "mood": "Mood",
# #             "meditation": "Meditation",
# #             "personal": "Personal Note",
# #             "work_or_study": "Work/Study",
# #             "extra_note": "Note"
# #         }

# #         tag_summary = {}

# #         # ✅ Flatten entries from each document
# #         flattened_entries = []
# #         for doc in data["journal"]:
# #             if "entries" in doc and isinstance(doc["entries"], list):
# #                 flattened_entries.extend(doc["entries"])
# #             else:
# #                 flattened_entries.append(doc)

# #         for entry in flattened_entries:
# #             if "tag" in entry and "text" in entry:
# #                 tag = entry.get("tag", "unknown")
# #                 text = entry.get("text", "")
# #             else:
# #                 possible_tag = next((key for key in entry if key in tag_map), None)
# #                 if not possible_tag:
# #                     continue
# #                 tag = possible_tag
# #                 text = entry.get(tag, "")

# #             readable = tag_map.get(tag, tag.replace("_", " ").title())
# #             tag_summary.setdefault(readable, []).append(text)

# #         if tag_summary:
# #             summary_lines = ["📝 **Journal Summary**:"]
# #             for tag, notes in tag_summary.items():
# #                 summary_lines.append(f"• {tag}: {' | '.join(notes)}")
# #             context.append("\n".join(summary_lines))



# #     # Notifications analysis

# #     # if data.get("user_notifications"):
# #     if "user_notifications" in data and isinstance(data["user_notifications"], list):

# #         unread = [n for n in data["user_notifications"] if not n.get("read", True)]
# #         if unread:
# #             summary = [
# #                 f"{n.get('timestamp').strftime('%Y-%m-%d') if isinstance(n.get('timestamp'), datetime) else str(n.get('timestamp'))[:10]} - {n.get('body', '[No message]')}"
# #                 for n in unread[:5]
# #             ]
# #             context.append("🔔 **Unread Notifications**:\n" + "\n".join(summary))
# #     # if data.get("user_notifications"):
# #     #     unread = [n for n in data["user_notifications"] if not n.get("read", True)]
# #     #     if unread:
# #     #         summary = [
# #     #             f"{n['timestamp'].strftime('%Y-%m-%d') if isinstance(n['timestamp'], datetime) else str(n['timestamp'])[:10]} - {n['message']}"
# #     #             for n in unread[:5]
# #     #         ]
# #     #         context.append("🔔 **Unread Notifications**:\n" + "\n".join(summary))
    
# #     return "\n\n".join(context) if context else "No recent data available for your query."




# # ============================================================================
# # MONGODB QUERY GENERATION (Fixed Version)
# # ============================================================================
# def is_greeting(text: str) -> bool:
#     greetings = {"hi", "hello", "hey", "good morning", "good night", "good evening", "good afternoon"}
#     return text.strip().lower() in greetings

# def get_time_based_greeting() -> str:
#     hour = datetime.now().hour
#     if 5 <= hour < 12:
#         return "🌞 Good morning! How can I help you today?"
#     elif 12 <= hour < 17:
#         return "🌤️ Good afternoon! How can I assist you?"
#     elif 17 <= hour < 21:
#         return "🌙 Good evening! Need help with any health questions?"
#     else:
#         return "🌜 Good night! If you have any health concerns, feel free to ask!"

# # ============================================================================
# # KNOWLEDGE BASE INTEGRATION
# # ============================================================================

# def gather_kb_context(query: str) -> List[str]:
#     """Gather context from knowledge base with improved error handling"""
#     context = []

#     if not os.path.exists(FAISS_FOLDER_PATH):
#         print(f"⚠️ FAISS folder not found: {FAISS_FOLDER_PATH}")
#         return context

#     for index_name in os.listdir(FAISS_FOLDER_PATH):
#         path = os.path.join(FAISS_FOLDER_PATH, index_name)
#         if os.path.isdir(path):
#             try:
#                 if index_name not in loaded_indexes:
#                     loaded_indexes[index_name] = load_faiss_index(path)
#                     print(f"✅ Successfully loaded FAISS index from {path}")

#                 retriever = loaded_indexes.get(index_name)
#                 if retriever:
#                     results = retriever.as_retriever(search_kwargs={"k": 3}).invoke(query)
#                     context += [doc.page_content.strip() for doc in results]
#                     print(f"📄 Retrieved {len(results)} documents from: {index_name}")
#                 else:
#                     print(f"⚠️ No retriever found for: {index_name}")
#             except Exception as e:
#                 print(f"⚠️ KB error loading [{index_name}]: {str(e)}")

#     return context

# # ============================================================================
# # RESPONSE GENERATION
# # ============================================================================

# def generate_intelligent_response(question: str, personal_context: str, kb_context: List[str], 
#                                   query_type: str, username: str) -> str:
#     """Generate intelligent response using both personal and knowledge base context."""
    
#     kb_text = "\n".join(kb_context[:3]) if kb_context else ""

#     if query_type == "personal":
#         system_prompt = (
#             f"You are a personal health assistant for {username}. Use their health data to answer. "
#             f"Be supportive and specific. Reference real data points and trends."
#         )
#         context_content = f"Personal Data:\n{personal_context}" if personal_context else "No personal data found."

#     elif query_type == "general":
#         system_prompt = (
#             "You are a knowledgeable health assistant. Answer factually based on health knowledge. "
#             "Also, if the user says 'hi', respond with an appropriate greeting based on the time of day."
#         )
#         context_content = f"Knowledge Base Information:\n{kb_text}" if kb_text else "Limited information available."

#     elif query_type == "hybrid":
#         system_prompt = (
#             f"You are a smart health assistant for {username}. Combine their health data with general knowledge. "
#             f"Compare their stats to normal ranges and give helpful insights."
#         )
#         parts = []
#         if personal_context:
#             parts.append(f"Personal Data:\n{personal_context}")
#         if kb_text:
#             parts.append(f"General Health Information:\n{kb_text}")
#         context_content = "\n\n".join(parts) if parts else "Limited data available."

#     else:
#         system_prompt = "You are a helpful assistant. Answer the user's question to the best of your ability."
#         context_content = personal_context or kb_text or "No data available."

#     try:
#         start = time.time()
#         response = client.chat.completions.create(
#             model="deepseek-chat",
#             temperature=0.4,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": f"Question: {question}\n\nContext:\n{context_content}"}
#             ]
#         )
#         end = time.time()
#         print(f"⏱️ DeepSeek LLM response time: {end - start:.2f} seconds")

#         return response.choices[0].message.content.strip()

#     except Exception as e:
#         print(f"❌ LLM generation failed: {e}")
#         return (
#             f"Sorry, I couldn't generate a response at this time. "
#             f"Based on your question about '{question}', I suggest checking with a healthcare provider."
#         )
# # ============================================================================
# # CONVERSATION MANAGEMENT
# # ============================================================================

# def get_or_create_conversation(convo_id: Optional[str], username: str):
#     """Get existing conversation or create new one"""
#     if convo_id:
#         try:
#             obj_id = ObjectId(convo_id)
#             convo = conversations_collection.find_one({"_id": obj_id, "username": username})
#             if convo:
#                 return obj_id
#         except:
#             raise HTTPException(status_code=400, detail="Invalid conversation ID")
    
#     result = conversations_collection.insert_one({
#         "username": username, 
#         "history": [], 
#         "created_at": datetime.utcnow()
#     })
#     return result.inserted_id

# def save_message(convo_id, role: str, content: str):
#     """Save message to conversation history"""
#     conversations_collection.update_one(
#         {"_id": ObjectId(convo_id)},
#         {
#             "$push": {
#                 "history": {
#                     "role": role, 
#                     "content": content,
#                     "timestamp": datetime.utcnow()
#                 }
#             }
#         }
#     )

# # def get_recent_history(convo_id, limit: int = 6):
# #     """Get recent conversation history"""
# #     convo = conversations_collection.find_one({"_id": convo_id})
# #     return {
# #         "_id": str(convo_id), 
# #         "history": convo.get("history", [])[-limit:] if convo else []
# #     }
# def get_recent_history(convo_id: str, limit: int = 6):
#     """Safely fetch recent conversation history by ObjectId"""
#     try:
#         obj_id = ObjectId(convo_id)
#     except Exception:
#         return {"_id": convo_id, "history": []}

#     convo = conversations_collection.find_one({"_id": obj_id})
#     if not convo:
#         return {"_id": convo_id, "history": []}

#     history = convo.get("history", [])
#     return {
#         "_id": str(obj_id),
#         "history": history[-limit:]  # last N entries
#     }
# # ============================================================================
# # MAIN API ENDPOINT
# # ============================================================================

# @router.post("/chat/ask", response_model=ChatResponse)
# def ask_chatbot(req: ChatRequest, token: str = Depends(oauth2_scheme)):
#     """
#     Main chatbot endpoint with intelligent query routing
#     """
#     # Authenticate user
#     valid, username = decode_token(token)
#     if not valid:
#         raise HTTPException(status_code=401, detail="Invalid token")

#     # Normalize and process query
#     query = normalize(req.question)
#     print(f"🤖 Processing query: {query} for user: {username}")


#     if is_greeting(query):
#         conversation_id = get_or_create_conversation(req.conversation_id, username)
#         save_message(conversation_id, "user", query)
#         greeting_response = get_time_based_greeting()
#         save_message(conversation_id, "assistant", greeting_response)
#         final_response = apply_personality(greeting_response, "friendly")
#         recent = get_recent_history(conversation_id)
#         return ChatResponse(
#             reply=final_response,
#             history=recent["history"],
#             conversation_id=str(conversation_id),
#             query_type="general",
#             data_sources=[]
#         )

    
#     # Get or create conversation
#     conversation_id = get_or_create_conversation(req.conversation_id, username)
#     save_message(conversation_id, "user", query)

#     # STEP 1: Detect query type and data sources needed
#     query_type, data_sources = detect_query_type(query, username)

#     # STEP 2: Prepare context for MongoDB queries (enhanced date detection)
#     context_info = {
#         "date": req.date,
#         "start_date": req.start_date,
#         "end_date": req.end_date,
#         "conversation_id": str(conversation_id)
#     }
    
#     # Auto-detect date context if not provided
#     if not req.start_date and not req.end_date:
#         auto_date_context = extract_date_context(query)
#         context_info.update(auto_date_context)
#         print(f"📅 Auto-detected date context: {auto_date_context}")

#     # STEP 3: Fetch personal data if needed
#     personal_context = ""
#     mongo_queries = {}
#     personal_data = {}
    
#     if "mongodb" in data_sources:
#         print("📊 Fetching personal data from MongoDB...")
#         mongo_queries = generate_ai_mongo_query_with_fallback(query, username, context_info)
#         print(f"🧠 AI MongoDB Query: {json.dumps(mongo_queries, indent=2)}")
        
#         personal_data = fetch_personal_data(mongo_queries, username)
#         print(f"📦 MongoDB Query Results: {json.dumps(personal_data, indent=2, default=str)}")
        
#         # Build context for response generation only
#         personal_context = build_comprehensive_context(personal_data, username)

#     # STEP 4: Gather knowledge base context if needed
#     kb_context = []
#     if "knowledge_base" in data_sources:
#         print("🔄 Searching knowledge base...")
#         try:
#             index_descriptions = load_index_descriptions_from_folders(FAISS_FOLDER_PATH)
#             best_index_name = find_best_matching_index(query, index_descriptions)

#             if best_index_name:
#                 index_path = os.path.join(FAISS_FOLDER_PATH, best_index_name)

#                 if not (
#                     os.path.exists(os.path.join(index_path, "index.faiss")) and
#                     os.path.exists(os.path.join(index_path, "index.pkl"))
#                 ):
#                     print(f"⚠️ Index files missing for: {best_index_name}")
#                 else:
#                     print(f"🔍 Querying FAISS index: {best_index_name}")
#                     kb_snippets = query_documents(query, index_path)  # uses DeepSeek in chain
#                     if kb_snippets:
#                         kb_context.extend(kb_snippets)
#                         print(f"✅ Retrieved {len(kb_snippets)} snippets from {best_index_name}")
#             else:
#                 print("⚠️ No good match found for the query.")

#         except Exception as e:
#             print(f"❌ Knowledge base search failed: {e}")


#     # STEP 5: Generate intelligent response
#     if personal_context or kb_context:
#         print("🤖 Generating intelligent response...")
#         response_text = generate_intelligent_response(query, personal_context, kb_context, query_type, username)
#     else:
#         response_text = "I couldn't find relevant information to answer your question. Could you please rephrase or provide more details?"

#     # STEP 6: Save assistant response and apply personality
#     save_message(conversation_id, "assistant", response_text)
#     final_response = apply_personality(response_text, "friendly")
    
#     # STEP 7: Get recent history and return response
#     recent = get_recent_history(conversation_id)
    
#     print(f"✅ Generated {query_type} response using {data_sources}")
    
#     return ChatResponse(
#         reply=final_response,
#         history=recent["history"],
#         conversation_id=str(conversation_id),
#         query_type=query_type,
#         data_sources=data_sources
#     )

# # ============================================================================
# # ADDITIONAL ENDPOINTS
# # ============================================================================

# @router.post("/chat/debug")
# def debug_chat_query(req: ChatRequest, token: str = Depends(oauth2_scheme)):
#     """
#     Debug endpoint that returns raw MongoDB queries and results
#     """
#     valid, username = decode_token(token)
#     if not valid:
#         raise HTTPException(status_code=401, detail="Invalid token")

#     query = normalize(req.question)
#     print(f"🔍 DEBUG: Processing query: {query} for user: {username}")
    
#     # Detect query type
#     query_type, data_sources = detect_query_type(query, username)
    
#     # Prepare context
#     context_info = {
#         "date": req.date,
#         "start_date": req.start_date,
#         "end_date": req.end_date
#     }
    
#     # Auto-detect date context if not provided
#     if not req.start_date and not req.end_date:
#         auto_date_context = extract_date_context(query)
#         context_info.update(auto_date_context)
    
#     # Generate and execute queries
#     mongo_queries = {}
#     personal_data = {}
    
#     if "mongodb" in data_sources:
#         mongo_queries = generate_ai_mongo_query_with_fallback(query, username, context_info)
#         personal_data = fetch_personal_data(mongo_queries, username)
    
#     return {
#         "query": query,
#         "username": username,
#         "query_type": query_type,
#         "data_sources": data_sources,
#         "context_info": context_info,
#         "generated_queries": mongo_queries,
#         "raw_results": personal_data
#     }

# # @router.get("/chat/history/", response_model=Dict[str, Any])
# # def get_previous_conversation(token: str = Depends(oauth2_scheme), limit: int = 6,conversation_id=str) -> Dict[str, Any]:
# #     """Fetch the last 'limit' messages and the conversation ID from the user's history."""
# #     valid, username = decode_token(token)
# #     if not valid or not username:
# #         raise HTTPException(status_code=401, detail="Invalid token or user not found")

# #     conversation = conversations_collection.find_one({"username": username ,"_id":conversation_id})
# #     print(conversation)
# #     if conversation and "history" in conversation:
# #         return {
# #             "_id": str(conversation["_id"]),
# #             "history": conversation["history"][-limit:]
# #         }

# #     return {
# #         "_id": None,
# #         "history": []
# #     }

# @router.get("/chat/history/{conversation_id}", response_model=Dict[str, Any])
# def get_previous_conversation(
#     token: str = Depends(oauth2_scheme), 
#     limit: int = 6, 
#     conversation_id: str = Path(..., title="Conversation ID", description="The ID of the conversation to retrieve")
# ) -> Dict[str, Any]:
#     """Fetch the last 'limit' messages and the conversation ID from the user's history."""
    
#     # Decode the token
#     valid, username = decode_token(token)
#     if not valid or not username:
#         raise HTTPException(status_code=401, detail="Invalid token or user not found")
    
#     try:
#         conversation_id_obj = ObjectId(conversation_id)
#     except Exception:
#         raise HTTPException(status_code=400, detail="Invalid conversation ID format")

#     # Fetch conversation data from the collection
#     conversation = conversations_collection.find_one({"username": username, "_id": conversation_id_obj})
#     if conversation and "history" in conversation:
#         return {
#             "_id": str(conversation["_id"]),
#             "history": conversation["history"][-limit:]
#         }

#     # Return empty result if no conversation is found
#     return {
#         "_id": None,
#         "history": []
#     }

# @router.get("/chat/conversations")
# def list_conversations(token: str = Depends(oauth2_scheme)):
#     """List all conversations for the user"""
#     valid, username = decode_token(token)
#     if not valid:
#         raise HTTPException(status_code=401, detail="Invalid token")
    
#     conversations = list(conversations_collection.find(
#         {"username": username},
#         {"_id": 1, "created_at": 1, "history": {"$slice": -1}}
#     ).sort("created_at", -1).limit(10))
    
#     for conv in conversations:
#         conv["_id"] = str(conv["_id"])
#         conv["last_message"] = conv["history"][0]["content"] if conv["history"] else "No messages"
#         del conv["history"]
    
#     return {"conversations": conversations}





from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from fastapi import APIRouter, Depends, HTTPException, Path
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from bson import ObjectId
from datetime import datetime, timedelta
import os, json, re
from groq import Groq as OpenAIClient
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re
from dataclasses import dataclass

# Database imports
from app.db.database import users_collection, conversations_collection
from app.db.health_data_model import alert_collection, health_data_collection
from app.db.journal_model import journals_collection
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.api.auth.auth import decode_token
import time
from app.utils.optimized_code_rag import load_faiss_index,query_documents
from app.core.advance_chatbot import normalize, apply_personality
from app.db.notification_model import notifications_collection
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize OpenAI client

FAISS_FOLDER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "faiss_indexes")
)


client = Groq(
    api_key=os.getenv("OPENAI_API_KEY"),
    # base_url=os.getenv("OPENAI_API_BASE_URL")
)
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL"),
    model=os.getenv("OPENAI_API_MODEL")
)

loaded_indexes = {}

# Pydantic models
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





from sentence_transformers import SentenceTransformer, util

topic_model = SentenceTransformer("all-MiniLM-L6-v2")  # local, free, fast

def load_index_descriptions_from_folders(base_path: str) -> dict:
    """
    Use FAISS folder names as natural descriptions (underscores replaced with spaces).
    """
    descriptions = {}
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            descriptions[folder] = folder.replace("_", " ").lower()
    return descriptions

def find_best_matching_index(query: str, index_descriptions: dict) -> str:
    start_time = time.perf_counter()
    query_embedding = topic_model.encode(query, convert_to_tensor=True)
    best_index = None
    best_score = -1

    for index_name, description in index_descriptions.items():
        desc_embedding = topic_model.encode(description, convert_to_tensor=True)
        score = util.pytorch_cos_sim(query_embedding, desc_embedding).item()
        if score > best_score:
            best_score = score
            best_index = index_name
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"⏱️ Elapsed time for finding best matching index: {elapsed_time:.4f} seconds")
    print(f"🔍 Best match: {best_index} (score: {best_score:.4f})")
    print(f"🔍 Worst match: {index_name} (score: {score:.4f})")
    return best_index


# ============================================================================
# QUERY TYPE DETECTION
# ============================================================================

def detect_query_type(question: str, username: str) -> tuple[str, List[str]]:
    """
    Detect if the query requires MongoDB data, knowledge base, or both
    Returns: (query_type, data_sources)
    """
    question_lower = question.lower()
    
    # General health information indicators
    general_keywords = [
        'what is', 'how does', 'why does', 'explain', 'tell me about',
        'definition', 'meaning', 'symptoms', 'causes', 'treatment',
        'normal range', 'healthy', 'should be', 'recommended',"why i'm"
    ]
    
    # Personal data indicators
    personal_keywords = [
        'my', 'i', 'me', 'mine', "i'm", 'today', 'yesterday', 'last week', 
        'this week', 'last month', 'this month', 'journal', 'diary', 'note'
    ]
    
    # Health metrics that could be personal or general
    health_metrics = [
        'heart rate', 'heartrate', 'pulse', 'bpm', 'steps', 'walking', 'activity',
        'spo2', 'oxygen', 'blood oxygen', 'sleep', 'calories', 'blood pressure', 
        'pressure', 'weight', 'bmi'
    ]
    
    # Specific data types
    personal_data_keywords = [
        'journal', 'diary', 'entries', 'mood', 'feelings', 'alert', 'warning', 
        'notification', 'reminder', 'summary'
    ]
    
    # Check for different types of queries
    has_personal = any(k in question_lower for k in personal_keywords)
    has_health_metric = any(k in question_lower for k in health_metrics)
    has_personal_data = any(k in question_lower for k in personal_data_keywords)
    has_general = any(k in question_lower for k in general_keywords)
    
    # Special handling for step-related queries
    is_step_query = any(k in question_lower for k in ['steps', 'step', 'walked', 'walking'])
    
    print(f"🔍 Query Analysis - Personal: {has_personal}, Health Metric: {has_health_metric}, "
          f"Personal Data: {has_personal_data}, General: {has_general}, Steps: {is_step_query}")
    
    # Determine query type and data sources
    if has_personal_data or (has_personal and has_health_metric) or is_step_query:
        query_type = "personal"
        data_sources = ["mongodb"]
    elif has_general and not has_personal:
        query_type = "general"
        data_sources = ["knowledge_base"]
    elif has_health_metric and has_general:
        query_type = "hybrid"
        data_sources = ["mongodb", "knowledge_base"]
    elif has_health_metric:
        query_type = "personal" if has_personal else "general"
        data_sources = ["mongodb"] if has_personal else ["knowledge_base"]
    else:
        query_type = "general"
        data_sources = ["knowledge_base"]
    
    print(f"🎯 Detected - Type: {query_type}, Sources: {data_sources}")
    return query_type, data_sources

def extract_date_context(question: str) -> dict:
    """Extract date context from natural language"""
    question_lower = question.lower()
    now = datetime.now()
    
    if "yesterday" in question_lower:
        yesterday = now - timedelta(days=1)
        return {
            "start_date": yesterday.strftime("%Y-%m-%dT00:00:00.000Z"),
            "end_date": now.strftime("%Y-%m-%dT00:00:00.000Z"),
            "date": yesterday.strftime("%Y-%m-%d")
        }
    elif "today" in question_lower:
        return {
            "start_date": now.strftime("%Y-%m-%dT00:00:00.000Z"),
            "end_date": (now + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00.000Z"),
            "date": now.strftime("%Y-%m-%d")
        }
    elif "last week" in question_lower:
        week_start = now - timedelta(days=now.weekday() + 7)
        week_end = week_start + timedelta(days=7)
        return {
            "start_date": week_start.strftime("%Y-%m-%dT00:00:00.000Z"),
            "end_date": week_end.strftime("%Y-%m-%dT00:00:00.000Z")
        }
    
    return {}

# ============================================================================
# MONGODB QUERY GENERATION (Fixed Version)
# ============================================================================

def clean_mongodb_response(generated: str) -> str:
    """Clean up AI-generated MongoDB query to make it valid JSON"""
    
    # Remove markdown formatting if present
    if generated.startswith("```"):
        generated = generated.strip("`").strip()
        if generated.lower().startswith("json"):
            generated = generated[4:].strip()
    
    # Replace single quotes with double quotes
    generated = generated.replace("'", '"')
    
    # Fix ISODate() functions - convert to ISO string format
    iso_date_pattern = r'ISODate\("([^"]+)"\)'
    generated = re.sub(iso_date_pattern, r'"\1"', generated)
    
    # Fix ObjectId() functions if any
    object_id_pattern = r'ObjectId\("([^"]+)"\)'
    generated = re.sub(object_id_pattern, r'"\1"', generated)
    
    # Remove any trailing commas before closing braces/brackets
    generated = re.sub(r',(\s*[}\]])', r'\1', generated)
    
    return generated

def convert_iso_string_to_datetime(value):
    """Convert ISO string to datetime object if it's a valid ISO date string"""
    if isinstance(value, str) and re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{3})?Z?$', value):
        try:
            # Handle both with and without milliseconds
            if value.endswith('Z'):
                value = value[:-1] + '+00:00'
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            return value
    return value

def convert_iso_dates_in_dict(obj: dict) -> dict:
    """Recursively convert ISO date strings to datetime objects"""
    if not isinstance(obj, dict):
        return obj
    
    result = {}
    for key, value in obj.items():
        if isinstance(value, dict):
            result[key] = convert_iso_dates_in_dict(value)
        elif isinstance(value, list):
            result[key] = [convert_iso_dates_in_dict(item) if isinstance(item, dict) else convert_iso_string_to_datetime(item) for item in value]
        else:
            result[key] = convert_iso_string_to_datetime(value)
    
    return result

def convert_iso_dates_in_query(query_list: list) -> list:
    """Convert ISO date strings to datetime objects in MongoDB query"""
    processed_query = []
    
    for stage in query_list:
        processed_stage = convert_iso_dates_in_dict(stage)
        processed_query.append(processed_stage)
    
    return processed_query

# def generate_fallback_query(username: str, context: dict) -> Dict[str, Any]:
#     """Generate a simple fallback query when AI generation fails"""
    
#     fallback_query = {
#         "health_data": [],
#         "journal": [],
#         "alerts": [],
#         "user_notifications": []
#     }
    
#     # Basic match filter with username
#     base_match = {"username": username}
    
#     # Add date range if provided
#     if context.get('start_date') and context.get('end_date'):
#         try:
#             start_date = datetime.fromisoformat(context['start_date'].replace('Z', '+00:00'))
#             end_date = datetime.fromisoformat(context['end_date'].replace('Z', '+00:00'))
            
#             date_filter = {
#                 "timestamp": {
#                     "$gte": start_date,
#                     "$lt": end_date
#                 }
#             }
#             base_match.update(date_filter)
#         except ValueError:
#             print("⚠️ Invalid date format in context, using basic query")
    
#     # Generate basic queries for each collection
#     fallback_query["health_data"] = [{"$match": base_match}]
#     fallback_query["journal"] = [{"$match": base_match}]
#     fallback_query["alerts"] = [{"$match": base_match}]
    
#     return fallback_query


def generate_fallback_query(username: str, context: dict) -> Dict[str, Any]:
    """Generate a simple fallback query when AI generation fails"""

    fallback_query = {
        "health_data": [],
        "journal": [],
        "alerts": [],
        "user_notifications": []
    }

    # Basic match filter with username
    base_match = {"username": username}

    # ✅ 1. Prioritize date-only match
    if context.get("date"):
        date_expr_match = {
            "username": username,
            "$expr": {
                "$eq": [
                    { "$dateToString": { "format": "%Y-%m-%d", "date": "$timestamp" } },
                    context["date"]
                ]
            }
        }

        fallback_query["health_data"] = [{"$match": date_expr_match}]
        fallback_query["journal"] = [{"$match": date_expr_match}]
        fallback_query["alerts"] = [{"$match": date_expr_match}]
        return fallback_query

    # 2. Range filter (used only if no specific `date` is present)
    if context.get('start_date') and context.get('end_date'):
        try:
            start_date = datetime.fromisoformat(context['start_date'].replace('Z', '+00:00'))
            end_date = datetime.fromisoformat(context['end_date'].replace('Z', '+00:00'))

            date_filter = {
                "timestamp": {
                    "$gte": start_date,
                    "$lt": end_date
                }
            }
            base_match.update(date_filter)
        except ValueError:
            print("⚠️ Invalid date format in context, using basic query")

    fallback_query["health_data"] = [{"$match": base_match}]
    fallback_query["journal"] = [{"$match": base_match}]
    fallback_query["alerts"] = [{"$match": base_match}]
    
    return fallback_query


def generate_ai_mongo_query(question: str, username: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a Mongo aggregation plan JSON for our collections, guided by the question + context.

    Returns a dict like:
    {
      "health_data": [ ...pipeline... ],
      "journal":     [ ...pipeline... ],
      "alerts":      [ ...pipeline... ],
      "user_notifications": [ ...pipeline... ],  # optional hint of the primary target
    }

    NOTE: This template encodes rules for WHEN/extremes/thresholds so the LLM
    (or your rule engine) always includes a timestamp with the result.
    Replace the 'llm_call' with your actual completion call if needed.
    """
    # ---- normalize/resolve date window from context ----
    start = context.get("start")
    end   = context.get("end")
    date_filter = {}
    if start or end:
        date_filter["timestamp"] = {}
        if start: date_filter["timestamp"]["$gte"] = start
        if end:   date_filter["timestamp"]["$lt"]  = end

    # ---- lightweight intent detection for WHEN ----
    ql = (question or "").strip().lower()
    is_when = ql.startswith("when")
    extreme_match = re.search(r"(highest|max|peak|lowest|min)\s+([a-z_ ]+)", ql)
    threshold_match = re.search(r"(hit|reach|cross(?:ed)?)\s+(\d+(?:\.\d+)?)\s*([a-z_]+)", ql)

    # Metric aliases used by builder:
    alias = {
        "heartrate":"heartrate", "heart rate":"heartrate", "hr":"heartrate", "bpm":"heartrate",
        "spo2":"spo2", "oxygen":"spo2", "o2":"spo2",
        "sleep":"sleep", "sleep minutes":"sleep", "sleep hours":"sleep",
        "steps":"steps", "step":"steps",
        "calorie":"calories", "calories":"calories", "kcal":"calories",
    }

    def norm_metric(name: str) -> str:
        n = name.strip().lower().replace("  ", " ").replace(" ", "_")
        return alias.get(n.replace("_", " "), alias.get(n, n))

    # ---------------- WHEN: extremes ----------------
    if is_when and extreme_match:
        extreme_word = extreme_match.group(1)
        metric_word  = norm_metric(extreme_match.group(2))
        # Build a simple sort→limit→project to get value + timestamp
        match_stage = {"username": username, "metric": metric_word}
        match_stage.update(date_filter)

        key = f"max_{metric_word}" if extreme_word in {"highest", "max", "peak"} else f"min_{metric_word}"
        sort = {"value": -1, "timestamp": -1} if "max" in key else {"value": 1, "timestamp": -1}

        return {
            "health_data": [
                {"$match": match_stage},
                {"$sort": sort},
                {"$limit": 1},
                {"$project": {"_id": 0, key: "$value", f"{key}_ts": "$timestamp"}}
            ],
            # "meta": {"collection": "health_data"}
        }

    # ---------------- WHEN: thresholds ----------------
    if is_when and threshold_match:
        threshold = float(threshold_match.group(2))
        metric_word = norm_metric(threshold_match.group(3))
        match_stage = {"username": username, "metric": metric_word, "value": {"$gte": threshold}}
        match_stage.update(date_filter)
        return {
            "health_data": [
                {"$match": match_stage},
                {"$sort": {"timestamp": 1}},
                {"$limit": 1},
                {"$project": {"_id": 0, "threshold_value": "$value", "threshold_ts": "$timestamp"}}
            ],
            # "meta": {"collection": "health_data"}
        }

    # ---------------- Journal WHEN: last tag ----------------
    # For performance you can narrow by date window here. We keep it simple: fetch whole docs and let the builder resolve.
    if is_when and re.search(r"when did i (?:last )?(meditat|eat|meal|breakfast|lunch|dinner|exercise|work|study|breath)", ql):
        return {
            "journal": [
                {"$match": {"username": username, **date_filter}},
                # We keep documents intact; builder will inspect entries[]
            ],
            # "meta": {"collection": "journal"}
        }

    # ---------------- Generic examples (avg/total/min/max) ----------------
    # Heuristic fallbacks: (Adjust to your use-cases or keep using your existing LLM prompting)
    if "average" in ql or "avg" in ql or "mean" in ql:
        m = re.search(r"(average|avg|mean)\s+([a-z_ ]+)", ql)
        metric_word = norm_metric(m.group(2) if m else "heartrate")
        return {
            "health_data": [
                {"$match": {"username": username, "metric": metric_word, **date_filter}},
                {"$group": {"_id": None, f"average_{metric_word}": {"$avg": "$value"}}},
                {"$project": {"_id": 0, f"average_{metric_word}": 1}}
            ],
            # "meta": {"collection": "health_data"}
        }

    if "total" in ql or "sum" in ql:
        m = re.search(r"(total|sum)\s+([a-z_ ]+)", ql)
        metric_word = norm_metric(m.group(2) if m else "steps")
        if metric_word == "steps":
            # Special case, many devices store steps in "value"
            return {
                "health_data": [
                    {"$match": {"username": username, "metric": "steps", **date_filter}},
                    {"$group": {"_id": None, "total_steps": {"$sum": "$value"}}},
                    {"$project": {"_id": 0, "total_steps": 1}}
                ],
                # "meta": {"collection": "health_data"}
            }
        return {
            "health_data": [
                {"$match": {"username": username, "metric": metric_word, **date_filter}},
                {"$group": {"_id": None, "total_value": {"$sum": "$value"}}},
                {"$project": {"_id": 0, "total_value": 1}}
            ],
            # "meta": {"collection": "health_data"}
        }

    if "highest" in ql or "max" in ql or "peak" in ql:
        m = re.search(r"(highest|max|peak)\s+([a-z_ ]+)", ql)
        metric_word = norm_metric(m.group(2) if m else "heartrate")
        return {
            "health_data": [
                {"$match": {"username": username, "metric": metric_word, **date_filter}},
                {"$sort": {"value": -1, "timestamp": -1}},
                {"$limit": 1},
                {"$project": {"_id": 0, f"max_{metric_word}": "$value", f"max_{metric_word}_ts": "$timestamp"}}
            ],
            # "meta": {"collection": "health_data"}
        }

    if "lowest" in ql or re.search(r"\bmin\b", ql):
        m = re.search(r"(lowest|min)\s+([a-z_ ]+)", ql)
        metric_word = norm_metric(m.group(2) if m else "heartrate")
        return {
            "health_data": [
                {"$match": {"username": username, "metric": metric_word, **date_filter}},
                {"$sort": {"value": 1, "timestamp": -1}},
                {"$limit": 1},
                {"$project": {"_id": 0, f"min_{metric_word}": "$value", f"min_{metric_word}_ts": "$timestamp"}}
            ],
            # "meta": {"collection": "health_data"}
        }

    # Fallback: no-op
    return {"meta": {"collection": None}}

def generate_ai_mongo_query_with_fallback(question: str, username: str, context: dict) -> Dict[str, Any]:
    """Main function with fallback support"""
    
    # Try AI generation first
    result = generate_ai_mongo_query(question, username, context)
    
    # If AI generation fails, use fallback
    if not result:
        print("🔄 Using fallback query generation")
        result = generate_fallback_query(username, context)
    
    return result

# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_personal_data(queries: Dict, username: str) -> Dict[str, List]:
    """Enhanced version with better error handling and date conversion"""
    results = {"health_data": [], "journal": [], "alerts": [], "user_notifications": []}
    collections = {
        "health_data": health_data_collection,
        "journal": journals_collection,
        "alerts": alert_collection
    }

    for key, query in queries.items():
        try:
            print(f"🔍 Processing {key} query: {query}")
            
            # if key == "user_notifications":
            #     user = users_collection.find_one({"username": username})
            #     if user and "notifications" in user:
            #         results[key] = user["notifications"]
            #         print(f"✅ Found {len(results[key])} notifications")
            #     else:
            #         print("❌ No notifications found")
            if key == "user_notifications":
            
                results[key] = list(notifications_collection.find(
                    {"username": username},
                    {"_id": 0}
                ))
                # print(f"✅ Found {len(results[key])} user notifications")

                    
            elif isinstance(query, list):
                # Convert ISO date strings to datetime objects in the query
                processed_query = convert_iso_dates_in_query(query)
                print(f"🔄 Processed aggregation query: {processed_query}")
                
                collection = collections[key]
                cursor = collection.aggregate(processed_query)
                results[key] = list(cursor)
                print(f"✅ Aggregation returned {len(results[key])} results")
                
                # Debug: Print first few results
                if results[key]:
                    print(f"📋 Sample result: {results[key][0]}")
                else:
                    print("⚠️ No results from aggregation - checking if data exists...")
                    # Test query without aggregation
                    if processed_query and processed_query[0].get("$match"):
                        test_query = processed_query[0]["$match"]
                        test_results = list(collection.find(test_query).limit(5))
                        print(f"🔍 Test query found {len(test_results)} raw documents")
                        if test_results:
                            print(f"📋 Sample raw document: {test_results[0]}")
                
            elif isinstance(query, dict):
                collection = collections[key]
                if "$match" in query:
                    # Wrap single $match into aggregation pipeline
                    processed_query = convert_iso_dates_in_query([query])
                    results[key] = list(collection.aggregate(processed_query))
                else:
                    processed_query = convert_iso_dates_in_dict(query)
                    results[key] = list(collection.find(processed_query))
                
                print(f"✅ Query returned {len(results[key])} results")
                
        except Exception as e:
            print(f"❌ Error in {key}: {e}")
            import traceback
            traceback.print_exc()
            results[key] = []
    
    return results


# ============================================================
#  Mongo Query Generation (LLM prompt + thin wrapper)
# ============================================================



# ============================================================
#  Context Builder (health + journal + alerts + notifications)
#  - concise for single-fact + all “when …” asks
#  - supports conversation-style journal you shared
# ============================================================

class Metric:
    STEPS = "steps"
    heartrate = "heartrate"
    SPO2 = "spo2"
    SLEEP = "sleep"
    CALORIES = "calories"

def normalize_metric(name: str) -> str:
    if not name:
        return "unknown"
    n = name.strip().lower().replace(" ", "_")
    aliases = {
        "hr": Metric.heartrate,
        "heartrate": Metric.HEART_RATE,
        "heart_rate": Metric.HEART_RATE,
        "bpm": Metric.HEART_RATE,
        "o2": Metric.SPO2,
        "spo2": Metric.SPO2,
        "oxygen": Metric.SPO2,
        "sleep_minutes": Metric.SLEEP,
        "sleep_hours": Metric.SLEEP,
        "sleep": Metric.SLEEP,
        "step": Metric.STEPS,
        "steps": Metric.STEPS,
        "calorie": Metric.CALORIES,
        "calories": Metric.CALORIES,
        "kcal": Metric.CALORIES,
    }
    return aliases.get(n, n)

def _safe_ts_to_str(ts: Any) -> str:
    if isinstance(ts, datetime):
        return ts.strftime("%Y-%m-%d %H:%M")
    s = str(ts)
    return s[:16] if len(s) > 16 else s

def _asks_single_fact(q: str) -> bool:
    if not q:
        return False
    ql = q.strip().lower()
    if ql.startswith("when"):
        return True
    return bool(re.search(r"\b(highest|max|lowest|min|average|avg|mean|total|sum)\b", ql))

@dataclass
class WhenQuery:
    kind: str                      # "extreme", "last_tag", "threshold"
    metric: Optional[str] = None
    extreme: Optional[str] = None  # "max" | "min"
    tag: Optional[str] = None
    threshold: Optional[float] = None

def _parse_when_question(q: str) -> Optional[WhenQuery]:
    if not q:
        return None
    ql = q.strip().lower()
    if not ql.startswith("when"):
        return None

    m = re.search(r"(highest|max|peak|lowest|min)\s+([a-z_ ]+)", ql)
    if m:
        extreme_word = m.group(1)
        metric_word = normalize_metric(m.group(2))
        extreme = "max" if extreme_word in {"highest", "max", "peak"} else "min"
        return WhenQuery(kind="extreme", metric=metric_word, extreme=extreme)

    tag_aliases = {
        "meditate": "meditation",
        "meditation": "meditation",
        "eat": "food_intake",
        "meal": "food_intake",
        "food": "food_intake",
        "sleep": "sleep",
        "work": "work_or_study",
        "study": "work_or_study",
        "mood": "mood",
        "note": "personal",
        "personal": "personal",
        "breathing": "breathing",
        "breathe": "breathing",
    }
    m = re.search(r"when did i (?:last )?([a-z_]+)", ql)
    if m:
        verb = m.group(1)
        tag = tag_aliases.get(verb)
        if tag:
            return WhenQuery(kind="last_tag", tag=tag)

    m = re.search(r"(hit|reach|cross(?:ed)?)\s+(\d+(?:\.\d+)?)\s*([a-z_]+)", ql)
    if m:
        threshold = float(m.group(2))
        metric = normalize_metric(m.group(3))
        return WhenQuery(kind="threshold", metric=metric, threshold=threshold)

    if re.search(r"when\s+(was|is)\s+it\??$", ql):
        return WhenQuery(kind="extreme")
    return None

# --------- Conversation-style journal helpers (your schema) ----------

TAG_PATTERNS: Dict[str, List[re.Pattern]] = {
    "meditation": [
        re.compile(r"\bdo your daily meditation\b", re.I),
        re.compile(r"\bmeditat(e|ion)\b", re.I),
    ],
    "breathing": [
        re.compile(r"\bdo your breathing exercises?\b", re.I),
        re.compile(r"\bbreathing exercise(s)?\b", re.I),
    ],
    "breakfast": [
        re.compile(r"\blog your breakfast\b", re.I),
        re.compile(r"\bbreakfast\b", re.I),
    ],
    "lunch": [
        re.compile(r"\blog your lunch\b", re.I),
        re.compile(r"\blunch\b", re.I),
    ],
    "dinner": [
        re.compile(r"\blog your dinner\b", re.I),
        re.compile(r"\bdinner\b", re.I),
    ],
    "exercise": [
        re.compile(r"\blog your exercise\b", re.I),
        re.compile(r"\bexercise\b", re.I),
        re.compile(r"\bworkout\b", re.I),
        re.compile(r"\bcycling|run(ning)?|walk(ing)?|gym\b", re.I),
    ],
    "work_or_study": [
        re.compile(r"\breflect on your workday\b", re.I),
        re.compile(r"\bwork(?:day)?\b", re.I),
        re.compile(r"\bstudy\b", re.I),
    ],
    "food_intake": [  # generic fallback
        re.compile(r"\b(log your )?meal(s)?\b", re.I),
        re.compile(r"\bfood\b", re.I),
    ],
    "sleep": [
        re.compile(r"\bsleep\b", re.I)
    ],
}

def _as_dt(ts: Any) -> Optional[datetime]:
    if isinstance(ts, datetime):
        return ts
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None

def _iter_entries_from_conversation_docs(journal_docs: List[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for doc in journal_docs:
        entries = doc.get("entries") if isinstance(doc, dict) else None
        if not isinstance(entries, list):
            continue
        for e in entries:
            if not isinstance(e, dict):
                continue
            yield {
                "role": e.get("role"),
                "content": (e.get("content") or "").strip(),
                "timestamp": e.get("timestamp") or doc.get("timestamp"),
                "task_id": e.get("task_id"),
            }

def _detect_tag_from_content(text: str) -> Optional[str]:
    if not text:
        return None
    for tag, patterns in TAG_PATTERNS.items():
        for p in patterns:
            if p.search(text):
                return tag
    return None

def _find_last_tag_timestamp_conversation(journal_docs: List[Dict[str, Any]], wanted_tag: str) -> Optional[datetime]:
    last_ts: Optional[datetime] = None
    by_task: Dict[str, Dict[str, Any]] = {}

    for e in _iter_entries_from_conversation_docs(journal_docs):
        task_id = e.get("task_id")
        ts = _as_dt(e.get("timestamp"))
        if not task_id or not ts:
            continue
        node = by_task.setdefault(task_id, {"assistant": None, "user": None})
        if e.get("role") == "assistant":
            node["assistant"] = max(node["assistant"], ts) if node["assistant"] else ts
        elif e.get("role") == "user":
            node["user"] = max(node["user"], ts) if node["user"] else ts

    candidate_ts: List[datetime] = []
    for doc in journal_docs:
        entries = doc.get("entries") if isinstance(doc, dict) else None
        if not isinstance(entries, list):
            continue
        for e in entries:
            if e.get("role") != "assistant":
                continue
            tag = _detect_tag_from_content(str(e.get("content", "")))
            if tag != wanted_tag:
                continue
            ts_assistant = _as_dt(e.get("timestamp") or doc.get("timestamp"))
            task_id = e.get("task_id")
            if task_id and task_id in by_task and by_task[task_id].get("user"):
                candidate_ts.append(by_task[task_id]["user"])
            elif ts_assistant:
                candidate_ts.append(ts_assistant)

    if not candidate_ts:
        for e in _iter_entries_from_conversation_docs(journal_docs):
            tag_any = _detect_tag_from_content(e.get("content", ""))
            if tag_any == wanted_tag:
                ts_any = _as_dt(e.get("timestamp"))
                if ts_any:
                    candidate_ts.append(ts_any)

    if candidate_ts:
        last_ts = max(candidate_ts)
    return last_ts

# --------- Health helpers for raw-record WHEN ----------

def _find_extreme_event(records: List[Dict[str, Any]], metric: str, extreme: str) -> Optional[Tuple[float, Any, Dict[str, Any]]]:
    cand = None
    for r in records:
        if normalize_metric(r.get("metric") or r.get("type") or "") != metric:
            continue
        v = r.get("value")
        if isinstance(v, (int, float)):
            if cand is None:
                cand = r
            else:
                if extreme == "max" and v > cand.get("value", float("-inf")):
                    cand = r
                elif extreme == "min" and v < cand.get("value", float("inf")):
                    cand = r
    if cand is None:
        return None
    return float(cand["value"]), cand.get("timestamp"), cand

def _find_threshold_event(records: List[Dict[str, Any]], metric: str, threshold: float) -> Optional[Any]:
    events = []
    for r in records:
        if normalize_metric(r.get("metric") or r.get("type") or "") != metric:
            continue
        v = r.get("value")
        ts = r.get("timestamp")
        if isinstance(v, (int, float)):
            key = ts if isinstance(ts, datetime) else str(ts)
            events.append((key, ts, float(v)))
    events.sort(key=lambda t: t[0])
    for _, ts, v in events:
        if v >= threshold:
            return ts
    return None

# --------------------- MAIN BUILDER ---------------------

def build_comprehensive_context(
    data: Dict[str, List],
    username: str,
    question: Optional[str] = None
) -> str:
    """
    Build context. Single-fact (max/min/avg/total) or any 'when ...' → one-liners.
    Broader asks → friendly summaries.
    """
    context: List[str] = []
    suppress_extras = _asks_single_fact(question or "")
    when_q = _parse_when_question(question or "")

    # ---- HEALTH ----
    health = data.get("health_data") or []
    if health:
        first_item = health[0]

        # Raw-record WHEN
        if when_q and isinstance(first_item, dict) and ("value" in first_item or "metric" in first_item or "type" in first_item):
            if when_q.kind == "extreme" and when_q.metric:
                found = _find_extreme_event(health, when_q.metric, when_q.extreme or "max")
                if found:
                    val, ts, _ = found
                    unit = ""
                    if when_q.metric == Metric.SLEEP and val > 24:
                        val, unit = val/60.0, " h (converted)"
                    elif when_q.metric == Metric.HEART_RATE:
                        unit = " bpm"
                    elif when_q.metric == Metric.SPO2:
                        unit = " %"
                    label = "highest" if (when_q.extreme or "max") == "max" else "lowest"
                    return f"🕒 {label.title()} {when_q.metric.replace('_',' ')}: {val:.2f}{unit} on {_safe_ts_to_str(ts)}"
            if when_q.kind == "threshold" and when_q.metric and when_q.threshold is not None:
                ts = _find_threshold_event(health, when_q.metric, when_q.threshold)
                if ts is not None:
                    return f"🕒 First reached {int(when_q.threshold):,} {when_q.metric.replace('_',' ')} on {_safe_ts_to_str(ts)}"

        # Aggregation-style
        if isinstance(first_item, dict) and any(
            k in first_item for k in [
                "total_steps","total_value","average_value",
                "average_heartrate","min_value","max_value",
                "min_heartrate","max_heartrate","highest_heartrate","lowest_heartrate",
                "total_sleep_hours","average_sleep_hours","min_sleep_hours","max_sleep_hours",
                "total_sleep","average_sleep","min_sleep","max_sleep",
                "total_calories","average_calories","min_calories","max_calories",
                "average_spo2","min_spo2","max_spo2",
            ]
        ):
            def _return(line: str):
                if suppress_extras:
                    return line
                context.append(line)
                return None

            if "total_steps" in first_item:
                out = _return(f"📊 Total steps: {int(round(first_item['total_steps'])):,}")
                if out: return out

            elif "average_heartrate" in first_item:
                out = _return(f"💓 Average heart rate: {float(first_item['average_heartrate']):.1f} bpm")
                if out: return out
            elif "max_heartrate" in first_item or "highest_heartrate" in first_item:
                val = float(first_item.get("max_heartrate") or first_item.get("highest_heartrate"))
                ts = first_item.get("max_heartrate_ts") or first_item.get("timestamp")
                line = f"💓 Maximum heart rate: {val:.0f} bpm" + (f" on {_safe_ts_to_str(ts)}" if ts else "")
                out = _return(line);  # noqa
                if out: return out
            elif "min_heartrate" in first_item or "lowest_heartrate" in first_item:
                val = float(first_item.get("min_heartrate") or first_item.get("lowest_heartrate"))
                ts = first_item.get("min_heartrate_ts") or first_item.get("timestamp")
                line = f"💓 Minimum heart rate: {val:.0f} bpm" + (f" on {_safe_ts_to_str(ts)}" if ts else "")
                out = _return(line);  # noqa
                if out: return out

            elif "total_sleep_hours" in first_item:
                out = _return(f"😴 Total sleep: {float(first_item['total_sleep_hours']):.2f} h")
                if out: return out
            elif "average_sleep_hours" in first_item:
                out = _return(f"😴 Average sleep: {float(first_item['average_sleep_hours']):.2f} h/night")
                if out: return out
            elif "min_sleep_hours" in first_item:
                out = _return(f"😴 Shortest sleep: {float(first_item['min_sleep_hours']):.2f} h")
                if out: return out
            elif "max_sleep_hours" in first_item:
                out = _return(f"😴 Longest sleep: {float(first_item['max_sleep_hours']):.2f} h")
                if out: return out

            elif any(k in first_item for k in ["average_sleep", "total_sleep", "min_sleep", "max_sleep"]):
                def to_h(m): return None if m is None else float(m)/60.0
                total_h = to_h(first_item.get("total_sleep"))
                avg_h   = to_h(first_item.get("average_sleep"))
                min_h   = to_h(first_item.get("min_sleep"))
                max_h   = to_h(first_item.get("max_sleep"))
                if total_h is not None:
                    out = _return(f"😴 Total sleep: {total_h:.2f} h")
                    if out: return out
                if avg_h is not None:
                    out = _return(f"😴 Average sleep: {avg_h:.2f} h/night")
                    if out: return out
                if min_h is not None:
                    out = _return(f"😴 Shortest sleep: {min_h:.2f} h")
                    if out: return out
                if max_h is not None:
                    out = _return(f"😴 Longest sleep: {max_h:.2f} h")
                    if out: return out

            elif "total_calories" in first_item:
                out = _return(f"🔥 Total calories: {int(round(float(first_item['total_calories']))):,} kcal")
                if out: return out
            elif "average_calories" in first_item:
                out = _return(f"🔥 Average calories: {int(round(float(first_item['average_calories']))):,} kcal/day")
                if out: return out
            elif "min_calories" in first_item:
                out = _return(f"🔥 Lowest daily calories: {int(round(float(first_item['min_calories']))):,} kcal")
                if out: return out
            elif "max_calories" in first_item:
                out = _return(f"🔥 Highest daily calories: {int(round(float(first_item['max_calories']))):,} kcal")
                if out: return out

            elif "average_spo2" in first_item:
                out = _return(f"🫁 Average SpO₂: {float(first_item['average_spo2']):.1f}%")
                if out: return out
            elif "min_spo2" in first_item:
                out = _return(f"🫁 Lowest SpO₂: {float(first_item['min_spo2']):.1f}%")
                if out: return out
            elif "max_spo2" in first_item:
                out = _return(f"🫁 Highest SpO₂: {float(first_item['max_spo2']):.1f}%")
                if out: return out

            elif "total_value" in first_item:
                out = _return(f"📊 Total: {first_item['total_value']:,}")
                if out: return out
            elif "average_value" in first_item:
                out = _return(f"📊 Average: {round(float(first_item['average_value']), 1)}")
                if out: return out
            elif "min_value" in first_item:
                out = _return(f"📊 Minimum: {first_item['min_value']}")
                if out: return out
            elif "max_value" in first_item:
                out = _return(f"📊 Maximum: {first_item['max_value']}")
                if out: return out

        # Raw summaries (non-when)
        else:
            values = [d.get("value") for d in health if isinstance(d.get("value"), (int, float))]
            if values:
                total = float(sum(values))
                avg = total / len(values)
                min_val = float(min(values))
                max_val = float(max(values))
                raw_metric = health[0].get("metric") or health[0].get("type") or "unknown"
                metric = normalize_metric(raw_metric)

                date_groups: Dict[str, List[float]] = {}
                for d in health:
                    ts = d.get("timestamp")
                    day = ts.strftime("%Y-%m-%d") if isinstance(ts, datetime) else str(ts)[:10]
                    if isinstance(d.get("value"), (int, float)):
                        date_groups.setdefault(day, []).append(float(d["value"]))
                trend = ""
                if len(date_groups) >= 3:
                    sorted_days = sorted(date_groups.items())[-3:]
                    avgs = [sum(vals)/max(len(vals),1) for _, vals in sorted_days]
                    if avgs[0] < avgs[1] < avgs[2]:
                        trend = " 📈 3-day improvement."
                    elif avgs[0] > avgs[1] > avgs[2]:
                        trend = " 📉 Slight decline."

                if metric == Metric.STEPS:
                    context.append(
                        f"📊 Steps Summary: total {int(round(total)):,}, avg {avg:.0f} per reading, "
                        f"range {int(min_val):,}–{int(max_val):,}.{trend}"
                    )
                elif metric == Metric.HEART_RATE:
                    context.append(f"💓 Heart Rate Summary: average {avg:.0f} bpm, range {min_val:.0f}-{max_val:.0f} bpm.{trend}")
                elif metric == Metric.SPO2:
                    context.append(f"🫁 SpO₂ Summary: average {avg:.1f}%, range {min_val:.1f}–{max_val:.1f}%.{trend}")
                elif metric == Metric.SLEEP:
                    converted = False
                    if max_val > 24:
                        values_h = [v/60.0 for v in values]
                        total, avg = sum(values_h), sum(values_h)/len(values_h)
                        min_val, max_val, converted = min(values_h), max(values_h), True
                    pretty = f"😴 Sleep Summary: total {total:.2f} h, avg {avg:.2f} h, range {min_val:.2f}–{max_val:.2f} h.{trend}"
                    if converted:
                        pretty += " (converted from minutes)"
                    context.append(pretty)
                elif metric == Metric.CALORIES:
                    context.append(
                        f"🔥 Calories Summary: total {int(round(total)):,} kcal, "
                        f"avg {int(round(avg)):,} kcal/read, range {int(round(min_val)):,}–{int(round(max_val)):,} kcal.{trend}"
                    )
                else:
                    title = (raw_metric or "unknown").replace("_", " ").title()
                    context.append(f"📊 {title} Summary: total {total:.1f}, avg {avg:.1f}, range {min_val:.1f}-{max_val:.1f}.{trend}")

    # If single-fact so far, return early (no extras)
    if _asks_single_fact(question or "") and context:
        return "\n\n".join(context)

    # ---- JOURNAL WHEN ----
    journal = data.get("journal") or []
    if when_q and when_q.kind == "last_tag":
        last_ts = _find_last_tag_timestamp_conversation(journal, when_q.tag or "")
        if last_ts is not None:
            return f"🕒 You last logged {when_q.tag.replace('_',' ')} on {_safe_ts_to_str(last_ts)}"

    # ---- JOURNAL SUMMARY (conversation-style) ----
    if journal and not _asks_single_fact(question or ""):
        counts: Dict[str, int] = {k: 0 for k in TAG_PATTERNS.keys()}
        latest: Dict[str, Optional[datetime]] = {k: None for k in TAG_PATTERNS.keys()}
        for e in _iter_entries_from_conversation_docs(journal):
            tag = _detect_tag_from_content(e.get("content", ""))
            if not tag:
                continue
            counts[tag] += 1
            ts = _as_dt(e.get("timestamp"))
            if ts and (latest[tag] is None or ts > latest[tag]):
                latest[tag] = ts

        lines = ["📝 Journal Summary:"]
        shown = False
        for tag in ["meditation", "breathing", "breakfast", "lunch", "dinner", "exercise", "work_or_study"]:
            if counts[tag] or latest[tag]:
                when = _safe_ts_to_str(latest[tag]) if latest[tag] else "—"
                lines.append(f"• {tag.replace('_',' ').title()}: {counts[tag]} entries, last on {when}")
                shown = True
        if shown:
            context.append("\n".join(lines))

    # ---- ALERTS ----
    if data.get("alerts"):
        alerts = []
        for d in data["alerts"]:
            ts = d.get("timestamp")
            ts_str = ts.strftime("%Y-%m-%d") if isinstance(ts, datetime) else str(ts)[:10]
            msg = d.get("message", "")
            alerts.append(f"{ts_str}: {msg}")
        if alerts:
            context.append("🚨 Recent Alerts:\n" + "\n".join(alerts[:5]))

    # ---- NOTIFICATIONS ----
    if isinstance(data.get("user_notifications"), list):
        unread = [n for n in data["user_notifications"] if not n.get("read", True)]
        if unread:
            lines = []
            for n in unread[:5]:
                ts = n.get("timestamp")
                ts_str = ts.strftime("%Y-%m-%d") if isinstance(ts, datetime) else str(ts)[:10]
                body = n.get("body", "[No message]")
                lines.append(f"{ts_str} - {body}")
            context.append("🔔 Unread Notifications:\n" + "\n".join(lines))

    return "\n\n".join(context) if context else "No recent data available for your query."






# ============================================================================
# MONGODB QUERY GENERATION (Fixed Version)
# ============================================================================
def is_greeting(text: str) -> bool:
    greetings = {"hi", "hello", "hey", "good morning", "good night", "good evening", "good afternoon"}
    return text.strip().lower() in greetings

def get_time_based_greeting() -> str:
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "🌞 Good morning! How can I help you today?"
    elif 12 <= hour < 17:
        return "🌤️ Good afternoon! How can I assist you?"
    elif 17 <= hour < 21:
        return "🌙 Good evening! Need help with any health questions?"
    else:
        return "🌜 Good night! If you have any health concerns, feel free to ask!"

# ============================================================================
# KNOWLEDGE BASE INTEGRATION
# ============================================================================

def gather_kb_context(query: str) -> List[str]:
    """Gather context from knowledge base with improved error handling"""
    context = []

    if not os.path.exists(FAISS_FOLDER_PATH):
        print(f"⚠️ FAISS folder not found: {FAISS_FOLDER_PATH}")
        return context

    for index_name in os.listdir(FAISS_FOLDER_PATH):
        path = os.path.join(FAISS_FOLDER_PATH, index_name)
        if os.path.isdir(path):
            try:
                if index_name not in loaded_indexes:
                    loaded_indexes[index_name] = load_faiss_index(path)
                    print(f"✅ Successfully loaded FAISS index from {path}")

                retriever = loaded_indexes.get(index_name)
                if retriever:
                    results = retriever.as_retriever(search_kwargs={"k": 3}).invoke(query)
                    context += [doc.page_content.strip() for doc in results]
                    print(f"📄 Retrieved {len(results)} documents from: {index_name}")
                else:
                    print(f"⚠️ No retriever found for: {index_name}")
            except Exception as e:
                print(f"⚠️ KB error loading [{index_name}]: {str(e)}")

    return context

# ============================================================================
# RESPONSE GENERATION
# ============================================================================

def generate_intelligent_response(question: str, personal_context: str, kb_context: List[str], 
                                  query_type: str, username: str) -> str:
    """Generate intelligent response using both personal and knowledge base context."""
    
    kb_text = "\n".join(kb_context[:3]) if kb_context else ""

    if query_type == "personal":
        system_prompt = (
            f"You are a personal health assistant for {username}. Use their health data to answer. "
            f"Be supportive and specific. Reference real data points and trends."
        )
        context_content = f"Personal Data:\n{personal_context}" if personal_context else "No personal data found."

    elif query_type == "general":
        system_prompt = (
            "You are a knowledgeable health assistant. Answer factually based on health knowledge. "
            "Also, if the user says 'hi', respond with an appropriate greeting based on the time of day."
        )
        context_content = f"Knowledge Base Information:\n{kb_text}" if kb_text else "Limited information available."

    elif query_type == "hybrid":
        system_prompt = (
            f"You are a smart health assistant for {username}. Combine their health data with general knowledge. "
            f"Compare their stats to normal ranges and give helpful insights."
        )
        parts = []
        if personal_context:
            parts.append(f"Personal Data:\n{personal_context}")
        if kb_text:
            parts.append(f"General Health Information:\n{kb_text}")
        context_content = "\n\n".join(parts) if parts else "Limited data available."

    else:
        system_prompt = "You are a helpful assistant. Answer the user's question to the best of your ability."
        context_content = personal_context or kb_text or "No data available."

    try:
        start = time.time()
        response = client.chat.completions.create(
            model="deepseek-chat",
            temperature=0.4,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}\n\nContext:\n{context_content}"}
            ]
        )
        end = time.time()
        print(f"⏱️ DeepSeek LLM response time: {end - start:.2f} seconds")

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"❌ LLM generation failed: {e}")
        return (
            f"Sorry, I couldn't generate a response at this time. "
            f"Based on your question about '{question}', I suggest checking with a healthcare provider."
        )
# ============================================================================
# CONVERSATION MANAGEMENT
# ============================================================================

def get_or_create_conversation(convo_id: Optional[str], username: str):
    """Get existing conversation or create new one"""
    if convo_id:
        try:
            obj_id = ObjectId(convo_id)
            convo = conversations_collection.find_one({"_id": obj_id, "username": username})
            if convo:
                return obj_id
        except:
            raise HTTPException(status_code=400, detail="Invalid conversation ID")
    
    result = conversations_collection.insert_one({
        "username": username, 
        "history": [], 
        "created_at": datetime.utcnow()
    })
    return result.inserted_id

def save_message(convo_id, role: str, content: str):
    """Save message to conversation history"""
    conversations_collection.update_one(
        {"_id": ObjectId(convo_id)},
        {
            "$push": {
                "history": {
                    "role": role, 
                    "content": content,
                    "timestamp": datetime.utcnow()
                }
            }
        }
    )

# def get_recent_history(convo_id, limit: int = 6):
#     """Get recent conversation history"""
#     convo = conversations_collection.find_one({"_id": convo_id})
#     return {
#         "_id": str(convo_id), 
#         "history": convo.get("history", [])[-limit:] if convo else []
#     }
def get_recent_history(convo_id: str, limit: int = 6):
    """Safely fetch recent conversation history by ObjectId"""
    try:
        obj_id = ObjectId(convo_id)
    except Exception:
        return {"_id": convo_id, "history": []}

    convo = conversations_collection.find_one({"_id": obj_id})
    if not convo:
        return {"_id": convo_id, "history": []}

    history = convo.get("history", [])
    return {
        "_id": str(obj_id),
        "history": history[-limit:]  # last N entries
    }
# ============================================================================
# MAIN API ENDPOINT
# ============================================================================

@router.post("/chat/ask", response_model=ChatResponse)
def ask_chatbot(req: ChatRequest, token: str = Depends(oauth2_scheme)):
    """
    Main chatbot endpoint with intelligent query routing
    """
    # Authenticate user
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid token")

    # Normalize and process query
    query = normalize(req.question)
    print(f"🤖 Processing query: {query} for user: {username}")


    if is_greeting(query):
        conversation_id = get_or_create_conversation(req.conversation_id, username)
        save_message(conversation_id, "user", query)
        greeting_response = get_time_based_greeting()
        save_message(conversation_id, "assistant", greeting_response)
        final_response = apply_personality(greeting_response, "friendly")
        recent = get_recent_history(conversation_id)
        return ChatResponse(
            reply=final_response,
            history=recent["history"],
            conversation_id=str(conversation_id),
            query_type="general",
            data_sources=[]
        )

    
    # Get or create conversation
    conversation_id = get_or_create_conversation(req.conversation_id, username)
    save_message(conversation_id, "user", query)

    # STEP 1: Detect query type and data sources needed
    query_type, data_sources = detect_query_type(query, username)

    # STEP 2: Prepare context for MongoDB queries (enhanced date detection)
    context_info = {
        "date": req.date,
        "start_date": req.start_date,
        "end_date": req.end_date,
        "conversation_id": str(conversation_id)
    }
    
    # Auto-detect date context if not provided
    if not req.start_date and not req.end_date:
        auto_date_context = extract_date_context(query)
        context_info.update(auto_date_context)
        print(f"📅 Auto-detected date context: {auto_date_context}")

    # STEP 3: Fetch personal data if needed
    personal_context = ""
    mongo_queries = {}
    personal_data = {}
    
    if "mongodb" in data_sources:
        print("📊 Fetching personal data from MongoDB...")
        mongo_queries = generate_ai_mongo_query_with_fallback(query, username, context_info)
        print(f"🧠 AI MongoDB Query: {json.dumps(mongo_queries, indent=2)}")
        
        personal_data = fetch_personal_data(mongo_queries, username)
        print(f"📦 MongoDB Query Results: {json.dumps(personal_data, indent=2, default=str)}")
        
        # Build context for response generation only
        personal_context = build_comprehensive_context(personal_data, username)

    # STEP 4: Gather knowledge base context if needed
    kb_context = []
    if "knowledge_base" in data_sources:
        print("🔄 Searching knowledge base...")
        try:
            index_descriptions = load_index_descriptions_from_folders(FAISS_FOLDER_PATH)
            best_index_name = find_best_matching_index(query, index_descriptions)

            if best_index_name:
                index_path = os.path.join(FAISS_FOLDER_PATH, best_index_name)

                if not (
                    os.path.exists(os.path.join(index_path, "index.faiss")) and
                    os.path.exists(os.path.join(index_path, "index.pkl"))
                ):
                    print(f"⚠️ Index files missing for: {best_index_name}")
                else:
                    print(f"🔍 Querying FAISS index: {best_index_name}")
                    kb_snippets = query_documents(query, index_path)  # uses DeepSeek in chain
                    if kb_snippets:
                        kb_context.extend(kb_snippets)
                        print(f"✅ Retrieved {len(kb_snippets)} snippets from {best_index_name}")
            else:
                print("⚠️ No good match found for the query.")

        except Exception as e:
            print(f"❌ Knowledge base search failed: {e}")


    # STEP 5: Generate intelligent response
    if personal_context or kb_context:
        print("🤖 Generating intelligent response...")
        response_text = generate_intelligent_response(query, personal_context, kb_context, query_type, username)
    else:
        response_text = "I couldn't find relevant information to answer your question. Could you please rephrase or provide more details?"

    # STEP 6: Save assistant response and apply personality
    save_message(conversation_id, "assistant", response_text)
    final_response = apply_personality(response_text, "friendly")
    
    # STEP 7: Get recent history and return response
    recent = get_recent_history(conversation_id)
    
    print(f"✅ Generated {query_type} response using {data_sources}")
    
    return ChatResponse(
        reply=final_response,
        history=recent["history"],
        conversation_id=str(conversation_id),
        query_type=query_type,
        data_sources=data_sources
    )

# ============================================================================
# ADDITIONAL ENDPOINTS
# ============================================================================

@router.post("/chat/debug")
def debug_chat_query(req: ChatRequest, token: str = Depends(oauth2_scheme)):
    """
    Debug endpoint that returns raw MongoDB queries and results
    """
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid token")

    query = normalize(req.question)
    print(f"🔍 DEBUG: Processing query: {query} for user: {username}")
    
    # Detect query type
    query_type, data_sources = detect_query_type(query, username)
    
    # Prepare context
    context_info = {
        "date": req.date,
        "start_date": req.start_date,
        "end_date": req.end_date
    }
    
    # Auto-detect date context if not provided
    if not req.start_date and not req.end_date:
        auto_date_context = extract_date_context(query)
        context_info.update(auto_date_context)
    
    # Generate and execute queries
    mongo_queries = {}
    personal_data = {}
    
    if "mongodb" in data_sources:
        mongo_queries = generate_ai_mongo_query_with_fallback(query, username, context_info)
        personal_data = fetch_personal_data(mongo_queries, username)
    
    return {
        "query": query,
        "username": username,
        "query_type": query_type,
        "data_sources": data_sources,
        "context_info": context_info,
        "generated_queries": mongo_queries,
        "raw_results": personal_data
    }

# @router.get("/chat/history/", response_model=Dict[str, Any])
# def get_previous_conversation(token: str = Depends(oauth2_scheme), limit: int = 6,conversation_id=str) -> Dict[str, Any]:
#     """Fetch the last 'limit' messages and the conversation ID from the user's history."""
#     valid, username = decode_token(token)
#     if not valid or not username:
#         raise HTTPException(status_code=401, detail="Invalid token or user not found")

#     conversation = conversations_collection.find_one({"username": username ,"_id":conversation_id})
#     print(conversation)
#     if conversation and "history" in conversation:
#         return {
#             "_id": str(conversation["_id"]),
#             "history": conversation["history"][-limit:]
#         }

#     return {
#         "_id": None,
#         "history": []
#     }

@router.get("/chat/history/{conversation_id}", response_model=Dict[str, Any])
def get_previous_conversation(
    token: str = Depends(oauth2_scheme), 
    limit: int = 6, 
    conversation_id: str = Path(..., title="Conversation ID", description="The ID of the conversation to retrieve")
) -> Dict[str, Any]:
    """Fetch the last 'limit' messages and the conversation ID from the user's history."""
    
    # Decode the token
    valid, username = decode_token(token)
    if not valid or not username:
        raise HTTPException(status_code=401, detail="Invalid token or user not found")
    
    try:
        conversation_id_obj = ObjectId(conversation_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid conversation ID format")

    # Fetch conversation data from the collection
    conversation = conversations_collection.find_one({"username": username, "_id": conversation_id_obj})
    if conversation and "history" in conversation:
        return {
            "_id": str(conversation["_id"]),
            "history": conversation["history"][-limit:]
        }

    # Return empty result if no conversation is found
    return {
        "_id": None,
        "history": []
    }

@router.get("/chat/conversations")
def list_conversations(token: str = Depends(oauth2_scheme)):
    """List all conversations for the user"""
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    conversations = list(conversations_collection.find(
        {"username": username},
        {"_id": 1, "created_at": 1, "history": {"$slice": -1}}
    ).sort("created_at", -1).limit(10))
    
    for conv in conversations:
        conv["_id"] = str(conv["_id"])
        conv["last_message"] = conv["history"][0]["content"] if conv["history"] else "No messages"
        del conv["history"]
    
    return {"conversations": conversations}