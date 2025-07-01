

###Version 2: Improved error handling and prompt generation logic
from openai import OpenAI
import os
from fastapi import HTTPException
from datetime import datetime
import random
import asyncio

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),base_url=os.getenv("OPENAI_API_BASE_URL"))

recent_journal_prompts = set()


FALLBACK_QUESTIONS = {
    "food": [
        "What did you eat today?",
        "Did you try anything new for your meals today?",
        "How did you feel about your food choices today?"
    ],
    "personal": [
        "Did you do anything for your wellbeing today?",
        "How did you take care of yourself today?",
        "Was there a moment you felt proud of yourself today?"
    ],
    "work_study": [
        "How was your work or study today?",
        "Did you accomplish something at work or school?",
        "What was the most interesting part of your work/study today?"
    ],
    "sleep": [
        "How did you sleep last night?",
        "Did you feel rested when you woke up?",
        "Was your sleep different from usual?"
    ],
    "extra": [
        "Thank you. Is there anything else you'd like to add about your day?",
        "Any other thoughts or feelings you'd like to journal?",
        "Would you like to note anything else?",
        "Is there something you wish you’d done differently today?",
        "Anything you'd like to let go of before tomorrow starts?"
    ]
}

# FALLBACK_QUESTIONS = {
#     "food": [
#         "What did you eat today?",
#         "How satisfied were you with your meals today?"
#     ],
#     "personal": [
#         "Did anything make you smile today?",
#         "How did you take care of yourself today?"
#     ],
#     "work_study": [
#         "What was the most meaningful thing you did today?",
#         "How did your work or studies go?"
#     ],
#     "sleep": [
#         "How did you sleep last night?",
#         "Did you feel rested this morning?"
#     ],
#     "extra": [
#         "Is there anything else you’d like to share?",
#         "Would you like to reflect on anything else today?"
#     ],
#     "scheduled": [
#         "It’s time for your daily journal. How are you feeling?",
#         "What’s one thing you'd like to reflect on right now?"
#     ]
# }

def get_time_of_day():
    hour = datetime.now().hour
    if hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    else:
        return "evening"

async def generate_journal_prompt(category: str, context: str = "", username: str = "", time_of_day: str = ""):
    attempt = 0
    max_attempts = 3
    while attempt < max_attempts:
        try:
            system_msg = (
                "You are a journaling assistant.\n"
                "Rules:\n"
                "1. Do NOT greet the user.\n"
                "2. Ask short, thoughtful journaling questions.\n"
                "3. Tone should match the time of day.\n"
                "4. Ask one clear question only — no polite closings or follow-ups.\n"
                "5. Avoid repeating questions.\n"
                "6. One clear sentence only."
            )

            user_msg = f"User: {username}\nTime: {time_of_day or get_time_of_day()}\nCategory: {category}\nContext: {context or 'none'}"

            response = client.chat.completions.create(
                model=os.getenv("OPENAI_API_MODEL"),
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=70,
                temperature=0.4
            )

            prompt = response.choices[0].message.content.strip()
            if prompt not in recent_journal_prompts:
                recent_journal_prompts.add(prompt)
                if len(recent_journal_prompts) > 100:
                    recent_journal_prompts.pop()
                return prompt
            else:
                attempt += 1
        except Exception as e:
            print("⚠️ GPT prompt generation error:", e)
            break  # Exit loop on quota/network/etc

    # ❗Fallback if all attempts fail
    return random.choice(FALLBACK_QUESTIONS.get(category, ["How are you feeling today?"]))
