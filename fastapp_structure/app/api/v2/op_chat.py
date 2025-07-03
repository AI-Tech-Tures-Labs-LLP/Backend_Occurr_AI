

from fastapi import APIRouter, Depends, HTTPException,WebSocket,WebSocketDisconnect
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from bson import ObjectId
from datetime import datetime, timedelta
import re, json, os, asyncio
from openai import OpenAI as OpenAIClient

# Database imports
from app.db.database import users_collection, conversations_collection
from app.db.health_data_model import alert_collection, health_data_collection
from app.db.journal_model import journals_collection
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.api.auth.auth import decode_token
import time
from app.utils.optimized_code_rag import load_faiss_index,query_documents
from app.core.advance_chatbot import normalize, apply_personality
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize OpenAI client

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
incomplete_profile_sessions = {}


# WebSocket manager for active connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, username: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[username] = websocket

    def disconnect(self, username: str):
        self.active_connections.pop(username, None)

    async def send_personal_message(self, username: str, message: str):
        if username in self.active_connections:
            await self.active_connections[username].send_text(message)

manager = ConnectionManager()

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
    query_embedding = topic_model.encode(query, convert_to_tensor=True)
    best_index = None
    best_score = -1

    for index_name, description in index_descriptions.items():
        desc_embedding = topic_model.encode(description, convert_to_tensor=True)
        score = util.pytorch_cos_sim(query_embedding, desc_embedding).item()
        if score > best_score:
            best_score = score
            best_index = index_name

    print(f"🔍 Best match: {best_index} (score: {best_score:.4f})")
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
    
    # Add date range if provided
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
    
    # Generate basic queries for each collection
    fallback_query["health_data"] = [{"$match": base_match}]
    fallback_query["journal"] = [{"$match": base_match}]
    fallback_query["alerts"] = [{"$match": base_match}]
    
    return fallback_query

def generate_ai_mongo_query(question: str, username: str, context: dict) -> Dict[str, Any]:
    """Generate MongoDB queries using AI with improved error handling"""
    
    system_prompt = (
        """You are an assistant that writes MongoDB queries. Return a valid JSON object where keys are 'health_data', 'journal', 'alerts', or 'user_notifications'.
Each value must be a list (aggregation pipeline). Use [{ "$match": {...} }] format even for simple filters.

IMPORTANT: Use only valid JSON syntax. For dates, use ISO string format like "2025-06-16T00:00:00.000Z" - DO NOT use ISODate() function.

Collections schema:
- 'health_data': username, metric, value, timestamp, created_at
- 'alerts': username, metric, value, timestamp, message, responded, created_at  
- 'journal': username, timestamp, mood, food_intake, sleep, personal, work_or_study, extra_note
- 'users': has embedded notifications array with fields: type, message, timestamp, read

Always filter by username. Use timestamp with "$gte" and "$lt" for date ranges. For aggregation queries, add "$group" stage.

Health metrics mapping:
- "steps" → metric: "steps"
- "heart rate", "heartrate", "pulse", "bpm" → metric: "heartRate" 
- "spo2", "oxygen" → metric: "spo2"
- "sleep" → metric: "sleep"
- "calories" → metric: "calories"

Aggregation types:
- "total", "sum" → {"$group": {"_id": null, "total_value": {"$sum": "$value"}}}
- "average", "avg", "mean" → {"$group": {"_id": null, "average_value": {"$avg": "$value"}}}
- "minimum", "min", "lowest" → {"$group": {"_id": null, "min_value": {"$min": "$value"}}}
- "maximum", "max", "highest" → {"$group": {"_id": null, "max_value": {"$max": "$value"}}}

Examples:

1. Total steps yesterday:
{
  "health_data": [
    {"$match": {"username": "user123", "metric": "steps", "timestamp": {"$gte": "2025-06-19T00:00:00.000Z", "$lt": "2025-06-20T00:00:00.000Z"}}},
    {"$group": {"_id": null, "total_steps": {"$sum": "$value"}}}
  ]
}

2. Average heart rate:
{
  "health_data": [
    {"$match": {"username": "user123", "metric": "heartRate"}},
    {"$group": {"_id": null, "average_heartrate": {"$avg": "$value"}}}
  ]
}

3. Maximum heart rate today:
{
  "health_data": [
    {"$match": {"username": "user123", "metric": "heartRate", "timestamp": {"$gte": "2025-06-20T00:00:00.000Z", "$lt": "2025-06-21T00:00:00.000Z"}}},
    {"$group": {"_id": null, "max_heartrate": {"$max": "$value"}}}
  ]
}"""
    )

    user_input = (
        f"User question: {question}\n"
        f"Username: {username}\n"
        f"Date: {context.get('date', '')}\n"
        f"Start Date: {context.get('start_date', '')}\n"
        f"End Date: {context.get('end_date', '')}\n\n"
        f"For questions about 'yesterday', use date range from yesterday 00:00:00 to today 00:00:00.\n"
        f"Current date context: Today is {datetime.now().strftime('%Y-%m-%d')}, so yesterday was {(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')}."
    )

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo"),
            temperature=0.3,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        )
        generated = response.choices[0].message.content.strip()

        if not generated:
            print("⚠️ Empty response from GPT for query generation.")
            return {}

        # Clean up the response
        generated = clean_mongodb_response(generated)
        
        print(f"🔍 Generated JSON: {generated}")
        
        try:
            return json.loads(generated)
        except json.JSONDecodeError as err:
            print(f"❌ JSON decode failed: {err}")
            print(f"Raw output: {generated}")
            return {}

    except Exception as e:
        print(f"❌ AI Mongo query generation failed: {e}")
        return {}

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
            
            if key == "user_notifications":
                user = users_collection.find_one({"username": username})
                if user and "notifications" in user:
                    results[key] = user["notifications"]
                    print(f"✅ Found {len(results[key])} notifications")
                else:
                    print("❌ No notifications found")
                    
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

# ============================================================================
# CONTEXT BUILDING
# ============================================================================

def build_comprehensive_context(data: Dict[str, List], username: str) -> str:
    """Build comprehensive context from fetched data"""
    context = []
    
    # Health data analysis
    if data["health_data"]:
        # Check if it's aggregation result (like total steps)
        first_item = data["health_data"][0]
        
        if isinstance(first_item, dict) and any(key in first_item for key in ['total_steps', 'total_value', 'average_value', 'average_heartrate', 'min_value', 'max_value', 'min_heartrate', 'max_heartrate']):
            # Handle aggregation results
            if 'total_steps' in first_item:
                total = first_item['total_steps']
                context.append(f"📊 **Total Steps**: You walked **{total:,} steps** during the requested period! 🚶‍♀️")
                
            elif 'average_heartrate' in first_item:
                avg = round(first_item['average_heartrate'], 1)
                context.append(f"💓 **Average Heart Rate**: Your average heart rate was **{avg} bpm** during the requested period!")
                if avg < 60:
                    context.append("📘 This is considered bradycardia (slow heart rate). Consider consulting a healthcare professional if you experience symptoms.")
                elif avg > 100:
                    context.append("📘 This is considered tachycardia (fast heart rate). Consider consulting a healthcare professional if you experience symptoms.")
                else:
                    context.append("✅ This is within the normal resting heart rate range (60-100 bpm).")
                    
            elif 'max_heartrate' in first_item:
                max_hr = first_item['max_heartrate']
                context.append(f"📈 **Maximum Heart Rate**: Your peak heart rate was **{max_hr} bpm** during the requested period!")
                
            elif 'min_heartrate' in first_item:
                min_hr = first_item['min_heartrate']
                context.append(f"📉 **Minimum Heart Rate**: Your lowest heart rate was **{min_hr} bpm** during the requested period!")
                
            elif 'total_value' in first_item:
                total = first_item['total_value']
                context.append(f"📊 **Total**: {total:,}")
                
            elif 'average_value' in first_item:
                avg = round(first_item['average_value'], 1)
                context.append(f"📊 **Average**: {avg}")
                
            elif 'min_value' in first_item:
                min_val = first_item['min_value']
                context.append(f"📊 **Minimum**: {min_val}")
                
            elif 'max_value' in first_item:
                max_val = first_item['max_value']
                context.append(f"📊 **Maximum**: {max_val}")
                
        else:
            # Handle individual records
            values = [d["value"] for d in data["health_data"] if "value" in d and isinstance(d["value"], (int, float))]
            metric = data["health_data"][0].get("metric", "unknown")
            
            if values:
                total = sum(values)
                avg = total / len(values)
                min_val = min(values)
                max_val = max(values)
                
                # Group by date for trend analysis
                date_groups = {}
                for d in data["health_data"]:
                    ts = d.get("timestamp")
                    if isinstance(ts, datetime):
                        day = ts.strftime("%Y-%m-%d")
                        date_groups.setdefault(day, []).append(d["value"])
                
                # Trend analysis
                trend = ""
                if len(date_groups) >= 3:
                    sorted_days = sorted(date_groups.items())[-3:]
                    day_avgs = [sum(vals) / len(vals) for _, vals in sorted_days]
                    if day_avgs[0] < day_avgs[1] < day_avgs[2]:
                        trend = "📈 You've shown a 3-day improvement trend!"
                    elif day_avgs[0] > day_avgs[1] > day_avgs[2]:
                        trend = "📉 Your recent data suggests a slight decline."
                
                # Build summary based on metric type
                if metric.lower() == "steps":
                    summary = f"You recorded a total of **{total:,} steps** with {len(values)} readings. Average: {avg:.0f} steps per reading. {trend}"
                    if total >= 10000:
                        summary += " 🎉 Excellent! You hit the 10,000+ steps goal!"
                    elif total >= 7000:
                        summary += " 👍 Great job staying active!"
                    else:
                        summary += " 💪 Every step counts - keep it up!"
                elif metric.lower() in ["heartrate", "heart_rate"]:
                    summary = f"Heart rate readings: Average {avg:.0f} bpm, Range: {min_val}-{max_val} bpm. {trend}"
                else:
                    summary = f"{metric.title()}: Total {total}, Average {avg:.1f}, Range: {min_val}-{max_val}. {trend}"
                
                context.append(f"📊 **{metric.title()} Summary**: {summary}")
    
    # Alerts analysis
    if data["alerts"]:
        alerts = [
            f"{d.get('timestamp').strftime('%Y-%m-%d') if isinstance(d.get('timestamp'), datetime) else str(d.get('timestamp'))[:10]}: {d.get('message', '')}"
            for d in data["alerts"]
        ]
        context.append("🚨 **Recent Alerts**:\n" + "\n".join(alerts[:5]))
    
    # Journal entries analysis
    if data["journal"]:
        entries = [
            f"{d.get('timestamp').strftime('%Y-%m-%d') if isinstance(d.get('timestamp'), datetime) else str(d.get('timestamp'))[:10]} | Sleep: {d.get('sleep', 'N/A')} | Note: {d.get('extra_note', 'N/A')}"
            for d in data["journal"]
        ]
        context.append("📝 **Recent Journal Entries**:\n" + "\n".join(entries[:3]))
    
    # Notifications analysis
    if data.get("user_notifications"):
        unread = [n for n in data["user_notifications"] if not n.get("read", True)]
        if unread:
            summary = [
                f"{n['timestamp'].strftime('%Y-%m-%d') if isinstance(n['timestamp'], datetime) else str(n['timestamp'])[:10]} - {n['message']}"
                for n in unread[:5]
            ]
            context.append("🔔 **Unread Notifications**:\n" + "\n".join(summary))
    
    return "\n\n".join(context) if context else "No recent data available for your query."




# ============================================================================
# MONGODB QUERY GENERATION (Fixed Version)
# ============================================================================
def is_greeting(text: str) -> bool:
    greetings = {"hi", "hii","hello", "hey", "good morning", "good night", "good evening", "good afternoon"}
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
        {"_id": convo_id},
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

def get_recent_history(convo_id, limit: int = 6):
    """Get recent conversation history"""
    convo = conversations_collection.find_one({"_id": convo_id})
    return {
        "_id": str(convo_id), 
        "history": convo.get("history", [])[-limit:] if convo else []
    }



@router.websocket("/ws/{username}")
async def websocket_endpoint(websocket: WebSocket, username: str):
    await manager.connect(username, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_personal_message(username, f"Echo: {data}")
    except WebSocketDisconnect:
        manager.disconnect(username)

def save_message(convo_id, role: str, content: str):
    conversations_collection.update_one({"_id": convo_id}, {"$push": {"history": {"role": role, "content": content, "timestamp": datetime.utcnow()}}})

def get_recent_history(convo_id, limit: int = 6):
    convo = conversations_collection.find_one({"_id": convo_id})
    return {"_id": str(convo_id), "history": convo.get("history", [])[-limit:] if convo else []}

def get_or_create_conversation(convo_id: Optional[str], username: str):
    if convo_id:
        try:
            obj_id = ObjectId(convo_id)
            convo = conversations_collection.find_one({"_id": obj_id, "username": username})
            if convo:
                return obj_id
        except:
            raise HTTPException(status_code=400, detail="Invalid conversation ID")
    result = conversations_collection.insert_one({"username": username, "history": [], "created_at": datetime.utcnow()})
    return result.inserted_id


async def check_abnormal_health_metrics(username: str) -> List[str]:
    now = datetime.utcnow()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)

    metrics = {
        "heartRate": {"$gt": 120},
        "spo2": {"$lt": 90},
        "sleep": {"$lt": 300}  # minutes
    }

    alerts = []
    for metric, condition in metrics.items():
        query = {
            "username": username,
            "metric": metric,
            "timestamp": {"$gte": start, "$lt": end},
            "value": condition
        }
        if health_data_collection.find_one(query):
            msg = {
                "heartRate": "💓 High heart rate detected today.",
                "spo2": "🫁 Low SpO₂ detected.",
                "sleep": "😴 Low sleep duration detected."
            }[metric]
            alerts.append(msg)

            alert_collection.insert_one({
                "username": username,
                "metric": metric,
                "value": condition,
                "timestamp": now,
                "message": msg,
                "responded": False,
                "created_at": now
            })

    return alerts

# ✅ Background task to trigger alerts/reminders async
async def health_alert_reminder_background():
    users = users_collection.find({})
    now = datetime.utcnow()
    for user in users:
        username = user["username"]
        food_schedule = user.get("food_schedule", {})
        today_journal = journals_collection.find_one({
            "username": username,
            "timestamp": {"$gte": now.replace(hour=0, minute=0, second=0, microsecond=0)}
        }) or {}

        def should_prompt(activity, time_str):
            if not time_str:
                return False
            try:
                hour, minute = map(int, time_str.split(":"))
                scheduled = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                return now >= scheduled and not today_journal.get(activity)
            except:
                return False

        reminders = []
        for meal in ["breakfast", "lunch", "dinner"]:
            if should_prompt("food_intake", food_schedule.get(meal)):
                reminders.append(f"🍽️ {username}, have you eaten {meal}?")

        if should_prompt("personal", user.get("meditation_time")):
            reminders.append(f"🧘 {username}, did you meditate today?")
        if should_prompt("work_or_study", user.get("exercise_time")):
            reminders.append(f"🏋️ {username}, did you work out today?")

        manual_alerts = []
        if user.get("weight_kg", 0) > 100:
            manual_alerts.append("⚠️ Your weight is above 100kg. Consider a health check or routine exercise.")

        abnormal_alerts = await check_abnormal_health_metrics(username)

        all_alerts = reminders + manual_alerts + abnormal_alerts
        if all_alerts:
            convo_id = get_or_create_conversation(None, username)
            message = "\n".join(all_alerts)
            save_message(convo_id, "assistant", message)
            await manager.send_personal_message(username, message)

@router.on_event("startup")
async def run_health_background():
    asyncio.create_task(health_alert_reminder_background())

@router.get("/background/reminders")
async def manual_health_alert_trigger(token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid token")
    await health_alert_reminder_background()
    return {"message": "✅ Health reminders triggered successfully."}
# ============================================================================
# MAIN API ENDPOINT
# ============================================================================

def prompt_and_track(field: str, prompt: str, username: str, conversation_id):
    global incomplete_profile_sessions
    incomplete_profile_sessions[username] = field
    save_message(conversation_id, "assistant", prompt)
    return ChatResponse(
        reply=prompt,
        history=get_recent_history(conversation_id)["history"],
        conversation_id=str(conversation_id),
        query_type="profile_completion",
        data_sources=[]
    )

@router.post("/chat/advanced", response_model=ChatResponse)
def ask_chatbot(req: ChatRequest, token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid token")

    query = normalize(req.question)
    print(f"🤖 Processing query: {query} for user: {username}")

    conversation_id = get_or_create_conversation(req.conversation_id, username)
    save_message(conversation_id, "user", query)

    user = users_collection.find_one({"username": username}) or {}
    food_schedule = user.get("food_schedule", {})


    if value in {"no", "nope", "not now", "skip", "maybe later","nah", "not interested"}:
            reply = f"👍 Got it! We’ll skip setting your {field.replace('_', ' ')} for now."
            save_message(conversation_id, "assistant", reply)
            return ChatResponse(
                reply=reply,
                history=get_recent_history(conversation_id)["history"],
                conversation_id=str(conversation_id),
                query_type="profile_completion",
                data_sources=[]
            )
    

    if any(kw in query.lower() for kw in ["i ate", "i had", "i have eaten", "today i ate", "breakfast", "lunch", "dinner"]):
        journals_collection.update_one(
            {
                "username": username,
                "timestamp": {"$gte": datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)}
            },
            {
                "$set": {"food_intake": query.strip(), "updated_at": datetime.utcnow()},
                "$setOnInsert": {"username": username, "timestamp": datetime.utcnow(), "entry_type": "freeform", "tags": ["food_intake"], "response": query.strip()}
            },
            upsert=True
        )
        confirmation = "🍽️ Got it! I've added that to your journal for today."
        save_message(conversation_id, "assistant", confirmation)
        return ChatResponse(
            reply=confirmation,
            history=get_recent_history(conversation_id)["history"],
            conversation_id=str(conversation_id),
            query_type="journal_entry",
            data_sources=["journal"]
        )

    if any(kw in query.lower() for kw in ["worked on", "studied", "did work", "assignment", "project"]):
        journals_collection.update_one(
            {
                "username": username,
                "timestamp": {"$gte": datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)}
            },
            {
                "$set": {"work_or_study": query.strip(), "updated_at": datetime.utcnow()},
                "$setOnInsert": {"username": username, "timestamp": datetime.utcnow(), "entry_type": "freeform", "tags": ["work_or_study"], "response": query.strip()}
            },
            upsert=True
        )
        confirmation = "📚 Logged your work/study activity for today."
        save_message(conversation_id, "assistant", confirmation)
        return ChatResponse(
            reply=confirmation,
            history=get_recent_history(conversation_id)["history"],
            conversation_id=str(conversation_id),
            query_type="journal_entry",
            data_sources=["journal"]
        )

    if any(kw in query.lower() for kw in ["slept", "sleep", "nap", "went to bed", "woke up"]):
        journals_collection.update_one(
            {
                "username": username,
                "timestamp": {"$gte": datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)}
            },
            {
                "$set": {"sleep": query.strip(), "updated_at": datetime.utcnow()},
                "$setOnInsert": {"username": username, "timestamp": datetime.utcnow(), "entry_type": "freeform", "tags": ["sleep"], "response": query.strip()}
            },
            upsert=True
        )
        confirmation = "😴 Got it. Sleep info logged for today."
        save_message(conversation_id, "assistant", confirmation)
        return ChatResponse(
            reply=confirmation,
            history=get_recent_history(conversation_id)["history"],
            conversation_id=str(conversation_id),
            query_type="journal_entry",
            data_sources=["journal"]
        )

    if any(kw in query.lower() for kw in ["felt", "emotion", "mood", "personal", "mental", "anxious", "happy", "sad"]):
        journals_collection.update_one(
            {
                "username": username,
                "timestamp": {"$gte": datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)}
            },
            {
                "$set": {"personal": query.strip(), "updated_at": datetime.utcnow()},
                "$setOnInsert": {"username": username, "timestamp": datetime.utcnow(), "entry_type": "freeform", "tags": ["personal"], "response": query.strip()}
            },
            upsert=True
        )
        confirmation = "🧠 Logged your personal notes for today."
        save_message(conversation_id, "assistant", confirmation)
        return ChatResponse(
            reply=confirmation,
            history=get_recent_history(conversation_id)["history"],
            conversation_id=str(conversation_id),
            query_type="journal_entry",
            data_sources=["journal"]
        )
                
    if is_greeting(query):
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

    query_type, data_sources = detect_query_type(query, username)
    context_info = {
        "date": req.date,
        "start_date": req.start_date,
        "end_date": req.end_date,
        "conversation_id": str(conversation_id)
    }
    if not req.start_date and not req.end_date:
        auto_date_context = extract_date_context(query)
        context_info.update(auto_date_context)

    # 🔐 Only prompt for profile if query_type is personal
    if query_type == "personal" and any(x in query.lower() for x in ["height", "weight", "diet", "job", "profession", "meditat", "exercis", "breakfast", "lunch", "dinner"]):
        if username in incomplete_profile_sessions:
            field = incomplete_profile_sessions.pop(username)
            value = query.strip()
            if re.match(r"^\d{2}:\d{2}$", value) and field.endswith("_time"):
                users_collection.update_one({"username": username}, {"$set": {field: value}})
                reply = f"✅ Your {field.replace('_', ' ')} has been set to {value}."
            elif field in ["height_cm", "weight_kg"] and value.isdigit():
                users_collection.update_one({"username": username}, {"$set": {field: int(value)}})
                reply = f"✅ Your {field.replace('_', ' ')} has been set to {value} cm/kg."
            elif field == "profession":
                users_collection.update_one({"username": username}, {"$set": {field: value}})
                reply = f"✅ Your profession has been set to {value}."
            elif field == "preferences.diet":
                users_collection.update_one({"username": username}, {"$set": {"preferences.diet": value}})
                reply = f"✅ Your diet preference has been set to {value}."
            else:
                reply = "⚠️ I couldn't understand your input. Please try again."
            save_message(conversation_id, "assistant", reply)
            return ChatResponse(
                reply=reply,
                history=get_recent_history(conversation_id)["history"],
                conversation_id=str(conversation_id),
                query_type="profile_completion",
                data_sources=[]
            )

        required_fields = {
            "meditation_time": "🧘 Would you like to set your daily meditation time? (e.g., 07:30)",
            "exercise_time": "🏋️ What time do you usually exercise? (e.g., 18:30)",
            "height_cm": "📏 What is your height in centimeters? (e.g., 175)",
            "weight_kg": "⚖️ What is your weight in kilograms? (e.g., 70)",
            "profession": "💼 What’s your profession?",
            "preferences.diet": "🥗 What’s your diet type? (vegetarian, vegan, non-vegetarian)"
        }
        for field, prompt in required_fields.items():
            if "." in field:
                k, sub = field.split(".")
                if not user.get(k, {}).get(sub):
                    return prompt_and_track(field, prompt, username, conversation_id)
            else:
                if not user.get(field):
                    return prompt_and_track(field, prompt, username, conversation_id)
        for meal in ["breakfast", "lunch", "dinner"]:
            if not food_schedule.get(meal):
                return prompt_and_track(f"food_schedule.{meal}", f"🍽️ What time do you usually eat {meal}? (e.g., 08:00)", username, conversation_id)

    personal_context, mongo_queries, personal_data = "", {}, {}
    if "mongodb" in data_sources:
        mongo_queries = generate_ai_mongo_query_with_fallback(query, username, context_info)
        personal_data = fetch_personal_data(mongo_queries, username)
        personal_context = build_comprehensive_context(personal_data, username)

    kb_context = []
    if "knowledge_base" in data_sources:
        try:
            index_descriptions = load_index_descriptions_from_folders(FAISS_FOLDER_PATH)
            best_index_name = find_best_matching_index(query, index_descriptions)
            if best_index_name:
                index_path = os.path.join(FAISS_FOLDER_PATH, best_index_name)
                if os.path.exists(os.path.join(index_path, "index.faiss")) and os.path.exists(os.path.join(index_path, "index.pkl")):
                    kb_snippets = query_documents(query, index_path)
                    if kb_snippets:
                        kb_context.extend(kb_snippets)
        except Exception as e:
            print(f"❌ Knowledge base search failed: {e}")

    if personal_context or kb_context:
        response_text = generate_intelligent_response(query, personal_context, kb_context, query_type, username)
    else:
        # response_text = "I couldn't find relevant information to answer your question. Could you please rephrase or provide more details?"
        response_text = "🤖 I couldn't find any relevant data to answer your question right now, but I'm always learning. Please try rephrasing or ask about your health, journal, or general health knowledge!""I couldn't find relevant information to answer your question. Could you please rephrase or provide more details?"


    save_message(conversation_id, "assistant", response_text)
    final_response = apply_personality(response_text, "friendly")
    recent = get_recent_history(conversation_id)

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

@router.get("/chat/history/", response_model=Dict[str, Any])
def get_previous_conversation(token: str = Depends(oauth2_scheme), limit: int = 6) -> Dict[str, Any]:
    """Fetch the last 'limit' messages and the conversation ID from the user's history."""
    valid, username = decode_token(token)
    if not valid or not username:
        raise HTTPException(status_code=401, detail="Invalid token or user not found")

    conversation = conversations_collection.find_one({"username": username})
    if conversation and "history" in conversation:
        return {
            "_id": str(conversation["_id"]),
            "history": conversation["history"][-limit:]
        }

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