# Backend_Occurr_AI
A fastapi Backend with the functionality of health care fits and the chatbot

📄 README.md
# 🧠 Health & Wellness Assistant API

A FastAPI-powered backend that manages user tasks, health data, journal entries, and AI-enhanced insights. Built with MongoDB, JWT auth, email verification, and push notification support.

---

## 🚀 Features

- 🧑 User registration & login with JWT authentication
- ✅ Email verification via link
- 📅 Auto-generates personalized daily tasks (meals, meditation, exercise, breathing, reflection)
- 🔔 Push notifications for pending tasks
- 📓 Journal entry logging with AI summary
- 📊 Health data support: steps, heart rate, calories, sleep
- 💬 AI-powered insights via DeepSeek or OpenAI
- 🌐 MongoDB integration

---

## 🧱 Tech Stack

- **FastAPI** — Web framework
- **MongoDB** — Database
- **LangChain + HuggingFace** — Vector search for knowledge base
- **DeepSeek / OpenAI** — LLM for smart replies
- **Uvicorn** — ASGI server
- **smtplib** — For sending email verification links
- **JWT Auth** — Secured user auth with OAuth2PasswordBearer

---

## ⚙️ Project Structure

app/
├── api/
│ ├── v2/ # Chat endpoints, AI routing
│ ├── auth/ # Token decoding, auth logic
├── db/ # MongoDB models and setup
├── core/ # Business logic (chatbot, summarization)
├── utils/ # Task generation, notification logic
main.py # FastAPI app entry point



---

## 📦 Setup Instructions

### 1. Clone and run the 
##Auto-setup script
 
 python setup_and_run.py

##Manually steup
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run your app
uvicorn app.main:app --host 0.0.0.0 --post 8000 --reload

☑️ This script creates a virtual environment, installs dependencies, and launches the server.




🔁 Automated Task Generation
Daily tasks based on the user's schedule:

Task	Trigger Field in schedule
Breakfast	breakfast_time
Meditation	meditation_time
Exercise	exercise_time
Reflection	sleep_time - 1 hour

🔔 Push Notification Example
"🍽️ Don’t forget to log your breakfast!"

Auto-triggered when a task becomes due and isn't completed.

📓 Journal + Health Data Summary
Users can log entries like:

"I had oats and green tea for breakfast"

"I meditated for 10 minutes"

AI summarizes journal entries by tag (food, mood, sleep, etc.)

🧠 AI Reply Flow
Classifies user message: journal_entry, mongo_query, knowledge_query, etc.

If a summary is requested: builds summary from MongoDB and/or AI

Uses HuggingFace/FAISS + DeepSeek/OpenAI for enhanced responses

📬 Example .env (if you re-enable it)
.env
OPENAI_API_MODEL=deepseek-chat
OPENAI_API_KEY=your@key
OPENAI_API_BASE_URL=
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
MONGO_URL = 
