PROMPT_TRIGGER_RULES = {
    "heartRate": {
        "condition": lambda value: value > 110,
        "time_sensitive": False
    },
    "spo2": {
        "condition": lambda value: value < 94,
        "time_sensitive": False
    },
    "sleep": {
        "condition": lambda value: value < 300,  # Less than 5 hours
        "time_sensitive": True,
        "valid_hours": range(6, 12)  # Only trigger in morning (6 AM to 11:59 AM)
    }
}
