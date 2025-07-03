
from pymongo import MongoClient
from datetime import datetime, timezone
from app.utils.prompt_rule import PROMPT_TRIGGER_RULES
from app.db.database import MONGO_URL, users_collection


client = MongoClient(MONGO_URL)
db = client["techjewel"]
health_data_collection = db["health_data_entries"]
alert_collection = db["health_alerts"]



def estimate_calories(steps, duration_min, weight_kg=70):
    steps_per_min = steps / duration_min if duration_min else 0
    if steps_per_min > 100:
        met = 6.0
    elif steps_per_min > 60:
        met = 3.5
    else:
        met = 1.5
    return round(met * weight_kg * (duration_min / 60), 2)


def should_trigger_prompt(metric, value, timestamp: datetime):
    rule = PROMPT_TRIGGER_RULES.get(metric)
    if not rule:
        return False

    if not rule["condition"](value):
        return False

    if rule.get("time_sensitive"):
        if timestamp.hour not in rule["valid_hours"]:
            return False

    return True


def generate_ai_prompt(metric, value):
    import random
    if metric == "heartRate":
        prompts = [
            "Your heart rate spiked. Were you exercising or feeling stressed?",
            "We noticed an unusually high heart rate. Did something trigger it?",
            "Heart rate was elevated recently. How’s your energy and mood today?"
        ]
        return random.choice(prompts)
    if metric == "spo2":
        prompts = [
            "Your oxygen level dropped. Are you feeling short of breath?",
            "SpO₂ was low. Make sure you're in a well-ventilated space and stay calm.",
            "Low oxygen detected. Have you experienced any fatigue or dizziness?"
        ]
        return random.choice(prompts)
    if metric == "sleep":
        prompts = [
            "You had less than 5 hours of sleep. Are you feeling well-rested?",
            "Sleep duration was short last night. How are your focus and mood today?",
            "Not much sleep detected. Would you like to reflect on your day?"
        ]
        return random.choice(prompts)
    return None


def create_health_alert(username, metric, value, timestamp):
    prompt = generate_ai_prompt(metric, value)
    if not prompt:
        return
    alert_collection.insert_one({
        "username": username,
        "metric": metric,
        "value": value,
        "timestamp": timestamp,
        "message": prompt,
        "responded": False,
        "created_at": datetime.utcnow().replace(tzinfo=timezone.utc)
    })

    users_collection.update_one(
        {"username": username},
        {"$push": {"notifications": {
            "type": "health_alert",
            "message": prompt,
            "timestamp": datetime.utcnow().replace(tzinfo=timezone.utc),
            "read": False
        }}}
    )





def save_health_data(username: str, payload: dict):
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    docs = []

    steps_data = payload.get("steps", {})
    calories_data = {}

    # Auto-calculate calories from step deltas
    sorted_steps = sorted(steps_data.items())
    for i in range(1, len(sorted_steps)):
        ts_prev, val_prev = sorted_steps[i - 1]
        ts_now, val_now = sorted_steps[i]

        t1 = datetime.fromisoformat(ts_prev)
        t2 = datetime.fromisoformat(ts_now)
        duration = (t2 - t1).seconds / 60
        steps_diff = max(0, val_now - val_prev)
        kcal = estimate_calories(steps_diff, duration)
        calories_data[ts_now] = kcal

    for metric in ["steps", "heartRate", "spo2", "sleep"]:
        metric_data = payload.get(metric, {})
        for ts, val in metric_data.items():
            ts_dt = datetime.fromisoformat(ts)


            if health_data_collection.find_one({
                "username": username,
                "timestamp": ts_dt
            }):
                continue
            doc = {
                "username": username,
                "metric": metric,
                "timestamp": ts_dt,
                "value": val,
                "created_at": now
            }
            if should_trigger_prompt(metric, val, ts_dt):
                doc["abnormal"] = True
                create_health_alert(username, metric, val, ts_dt)
            docs.append(doc)

    for ts, val in calories_data.items():
        docs.append({
            "username": username,
            "metric": "calories",
            "timestamp": datetime.fromisoformat(ts),
            "value": val,
            "created_at": now
        })

    if docs:
        health_data_collection.insert_many(docs)



def get_health_data_by_range(username: str, start: datetime, end: datetime):
    return list(health_data_collection.find({
        "username": username,
        "timestamp": {"$gte": start, "$lte": end}
    }).sort("timestamp", 1))


def get_metric_summary(username, metric, start: datetime, end: datetime):
    pipeline = [
        {"$match": {
            "username": username,
            "timestamp": {"$gte": start, "$lte": end},
            "metric": metric
        }},
        {"$group": {
            "_id": None,
            "min": {"$min": "$value"},
            "max": {"$max": "$value"},
            "avg": {"$avg": "$value"},
            "total": {"$sum": "$value"},
            "count": {"$sum": 1}
        }}
    ]
    result = list(health_data_collection.aggregate(pipeline))
    return result[0] if result else None

def extract_metric_value(entry, metric):
    try:
        if metric == "steps":
            return entry["value"] if entry["metric"] == "steps" else 0
        elif metric == "heartRate":
            return entry["value"] if entry["metric"] == "heartRate" else 0
        elif metric == "spo2":
            return entry["value"]  if entry["metric"] == "spo2" else 0
        elif metric == "sleep":
            return entry["value"] if entry["metric"] == "sleep" else 0
        elif metric == "calories":
            return entry["value"] if entry["metric"] == "calories" else 0
    except:
        return None
    
    
