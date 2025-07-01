import pandas as pd
import numpy as np
import random
import requests
from datetime import datetime

AUTH_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJBbnRpbWFAMTIzIiwiZXhwIjoxNzQ5MTIxMTkwfQ.kX_Rw7VBePXyDLzexbnbc8osJRcnLGJ63AToUqf6Yog"
API_URL = "http://localhost:8000/api/v1/health_data/health/save"    

headers = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}



# === Load Kaggle CSVs ===
hr_df = pd.read_csv(r"C:\Users\phefa\OneDrive\Desktop\Teach Jewellery\Knowledge_Base_chatbot\fastapp_structure\data\heartrate_seconds_merged.csv", nrows=3000)
step_df = pd.read_csv(r"C:\Users\phefa\OneDrive\Desktop\Teach Jewellery\Knowledge_Base_chatbot\fastapp_structure\data\minuteStepsNarrow_merged.csv", nrows=3000)
sleep_df = pd.read_csv(r"C:\Users\phefa\OneDrive\Desktop\Teach Jewellery\Knowledge_Base_chatbot\fastapp_structure\data\minuteSleep_merged.csv", nrows=500)


# === Format timestamps ===
hr_df["Time"] = pd.to_datetime(hr_df["Time"])
step_df["ActivityMinute"] = pd.to_datetime(step_df["ActivityMinute"])
sleep_df["date"] = pd.to_datetime(sleep_df["date"])

# === Pick data for 1 user + 1 day ===
filtered_hr = hr_df.head(50)
filtered_steps = step_df.head(50)
filtered_sleep = sleep_df.head(1)

# === Build payload ===
payload = {
    "heartRate": {
        row["Time"].isoformat(): int(row["Value"])
        for _, row in filtered_hr.iterrows()
    },
    "steps": {
        row["ActivityMinute"].isoformat(): int(row["Steps"])
        for _, row in filtered_steps.iterrows()
    },
    "spo2": {  # Simulated from heart rate
        row["Time"].isoformat(): round(random.uniform(95, 98), 1)
        for _, row in filtered_hr.iterrows()
    },
    "sleep": {
        filtered_sleep.iloc[0]["date"].isoformat(): int(filtered_sleep.iloc[0]["value"])
    }
}

# === Send to API ===
res = requests.post(API_URL, json=payload, headers=headers)
print("Status:", res.status_code)
print("Response:", res.json())