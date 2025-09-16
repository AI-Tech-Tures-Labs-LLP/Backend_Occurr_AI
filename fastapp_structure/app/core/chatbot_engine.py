

from swarm import Swarm, Agent
from app.utils.llm_chat_sql import sql_response_gen
from app.utils.optimized_code_rag import query_documents
from app.utils.identify_query import get_response, get_sql_format_response
# from langchain_community.vectorstores import FAISS
import os
from app.utils.optimized_code_rag import load_faiss_index
FAISS_FOLDER_PATH = os.path.join("data", "faiss_indexes")
import os
from groq import Groq as OpenAIClient




import os
from typing import List
from groq import Groq
from app.utils.optimized_code_rag import load_faiss_index
from app.utils.optimized_code_rag import query_documents

FAISS_FOLDER_PATH = os.path.join("data", "faiss_indexes")
client = Groq(api_key=os.getenv("OPENAI_API_KEY"),base_url=os.getenv("OPENAI_API_BASE_URL"))
loaded_indexes = {}

def extract_context(messages):
    previous_query = previous_response = current_query = None
    for msg in reversed(messages):
        if msg["role"] == "user":
            if current_query is None:
                current_query = msg["content"]
            elif previous_query is None:
                previous_query = msg["content"]
        elif msg["role"] == "assistant" and previous_response is None:
            previous_response = msg["content"]
        if previous_query and previous_response and current_query:
            break
    return previous_query, previous_response, current_query

def normalize(text: str) -> str:
    return text.strip()

def apply_personality(raw_answer: str, mode: str) -> str:
    mode = mode.lower()
    if mode == "krishna":
        return (
            "🌸 Wisdom from the Bhagavad Gita 🌸\n\n"
            f"🕉️ {raw_answer}\n\n"
            "Let us reflect on this divine insight as Lord Krishna guides us."
        )
    else:
        return (
            "💡Health Insight💡\n\n"
            f"{raw_answer}\n\n"
            "Let me know if you have any more health-related questions!"
        )

def choose_best_answer(user_query: str, candidates: List[str]) -> str:
    if not candidates:
        return "⚠️ No answers to evaluate."

    system_prompt = (
        "You are a smart medical assistant. Given a user's health-related question and a list of candidate answers, "
        "choose the most relevant and medically accurate one. Return only that answer."
    )

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_API_MODEL")

,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User Question: {user_query}\n\n" +
                                            "\n\n---\n\n".join(candidates)},
                {"role": "user", "content": "Choose the most relevant answer."}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("GPT ranking failed:", e)
        return candidates[0]

def process_query(messages, mode="friendlymode"):
    previous_query, previous_response, current_query = extract_context(messages)
    formatted_query = normalize(current_query)
    mode = mode.lower()

    if mode == "krishna":
        result = query_documents(formatted_query, path=os.path.join(FAISS_FOLDER_PATH, "Gita"))
        return apply_personality(result or "🙏 No Gita verse matched.", mode)

    all_matches = []

    for index_name in os.listdir(FAISS_FOLDER_PATH):
            index_path = os.path.join(FAISS_FOLDER_PATH, index_name)
            if os.path.isdir(index_path) and index_name.lower() != "gita":
                if index_name not in loaded_indexes:
                    loaded_indexes[index_name] = load_faiss_index(index_path)

                index = loaded_indexes[index_name]
                retriever = index.as_retriever(search_kwargs={"k": 3})  # increase k for better coverage
                result_docs = retriever.invoke(formatted_query)

                all_matches.extend([doc.page_content.strip() for doc in result_docs])


    # for index_name in os.listdir(FAISS_FOLDER_PATH):
    #     if index_name.lower() == "gita":
    #         continue
    #     index_path = os.path.join(FAISS_FOLDER_PATH, index_name)
    #     if os.path.isdir(index_path):
    #         if index_name not in loaded_indexes:
    #             loaded_indexes[index_name] = load_faiss_index(index_path)

    #         index = loaded_indexes[index_name]
    #         retriever = index.as_retriever(search_kwargs={"k": 2})
    #         results = retriever.invoke(formatted_query)

    #         all_matches.extend([doc.page_content.strip() for doc in results])

    if not all_matches:
        return apply_personality("⚠️ No answer found in the knowledge base.", mode)

    best = choose_best_answer(formatted_query, all_matches)
    return apply_personality(best, mode)



def main(messages, mode="System"):
    return process_query(messages, mode)

