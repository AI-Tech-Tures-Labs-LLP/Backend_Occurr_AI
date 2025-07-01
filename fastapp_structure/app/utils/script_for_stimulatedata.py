import pandas as pd
import numpy as np
import requests
import random
from datetime import datetime ,timedelta
import time

AUTH_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJBbnRpbWFAMTIzIiwiZXhwIjoxNzQ5NTM5NDgxfQ.f0UnTNhtxZCZhDXUzs5vavRnX2lJ9l8gmAJ51IFEn44"

API_URL = "http://localhost:8000/api/v1/health_data/health/save"

headers = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}


def simulate_and_send():
    now = datetime.utcnow().replace(second=0, microsecond=0)
    ts = now.isoformat()

    steps = {ts: random.randint(10, 100)}
    heartRate = {ts: random.randint(60, 150)}
    spo2 = {ts: round(random.uniform(95.0, 98.5), 1)}
    
    # Optional: simulate short nap (20–40 min sleep)
    sleep = {}
    if random.random() < 0.05:  # 5% chance
        sleep[ts] = random.randint(20, 40)

    payload = {
        "steps": steps,
        "heartRate": heartRate,
        "spo2": spo2,
        "sleep": sleep
    }

    res = requests.post(API_URL, headers=headers, json=payload)
    print(f"[{ts}] Sent → Status: {res.status_code}")

# === Loop: send data every 30 seconds ===
while True:
    simulate_and_send()
    time.sleep(30)  # wait 30 seconds


# version = "1.0.0"
# now = datetime.utcnow().replace(second=0, microsecond=0)
# timestamps = [now - timedelta(minutes=i * 5) for i in range(12)]  # 1 hour window

# # Steps (cumulative increasing)
# step_counts = [random.randint(20, 80) for _ in timestamps]
# steps = {}
# total = 0
# for ts, count in zip(reversed(timestamps), step_counts):
#     total += count
#     steps[ts.isoformat()] = total

# # Heart rate
# heartRate = {
#     ts.isoformat(): random.randint(60, 100)
#     for ts in timestamps
# }

# # SpO2 (as percentage)
# spo2 = {
#     ts.isoformat(): round(random.uniform(95.0, 98.5), 1)
#     for ts in timestamps
# }

# # Sleep (one 8-hour block at night)
# sleep_start = (now - timedelta(hours=8)).replace(minute=0)
# sleep = {
#     sleep_start.isoformat(): 480  # 8 hours = 480 minutes
# }

# # Combine payload
# payload = {
#     "steps": steps,
#     "heartRate": heartRate,
#     "spo2": spo2,
#     "sleep": sleep
# }

# # === POST TO API ===
# response = requests.post(API_URL, headers=headers, json=payload)

# print("Status Code:", response.status_code)
# print("Response:", response.json())