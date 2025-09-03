# from groq import Groq
import os
import openai
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print(OPENAI_API_KEY)
# client = Groq(api_key=OPENAI_API_KEY,base_url=os.getenv("OPENAI_API_BASE_URL"))
client = Groq(api_key=os.getenv("OPENAI_API_KEY"),base_url=os.getenv("OPENAI_API_BASE_URL"))

def get_response(query_context: list) -> str:
    system_message = {
        "role": "system",
        "content": (
            "Rephrase the current user query based on previous query and response. "
            "Return only the clarified query. For general/document search."
        )
    }

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_API_MODEL"),
        messages=[system_message] + query_context
    )

    return response.choices[0].message.content.strip()


def get_sql_format_response(query_context: list) -> str:
    system_message = {
        "role": "system",
        "content": (
            "Rephrase the current user query for a database system. "
            "Keep the output natural-language and descriptive. No SQL code."
        )
    }

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_API_MODEL"),
        messages=[system_message] + query_context
    )

    return response.choices[0].message.content.strip()
