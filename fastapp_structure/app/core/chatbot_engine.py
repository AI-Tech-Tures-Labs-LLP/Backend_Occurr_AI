

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

# # FAISS_FOLDER_PATH = r"faiss_indexes"
# # from langchain_community.embeddings import OpenAIEmbeddings
# # from langchain.llms import OpenAI
# # from langchain_openai import OpenAI, OpenAIEmbeddings
# # from langchain_community.vectorstores import FAISS

#  # Cache FAISS indexes




# def apply_personality(raw_answer: str, mode: str) -> str:
#     mode = mode.lower()
    
#     if mode == "krishna":
#         return (
#             "🌸 **Wisdom from the Bhagavad Gita** 🌸\n\n"
#             f"🕉️ {raw_answer}\n\n"
#             "Let us reflect on this divine insight as Lord Krishna guides us."
#         )
#     else:
#         return (
#             "💡 **Health Insight** 💡\n\n"
#             f"{raw_answer}\n\n"
#             "Let me know if you have any more health-related questions!"
#         )


# # Agent setup (still here in case you expand later)
# rag_agent = Agent(
#     name="RAG Agent",
#     instructions="Handles document/general knowledge queries using retrieval-based QA.",
#     functions=[query_documents]
# )

# nl2sql_agent = Agent(
#     name="NL2SQL Agent",
#     instructions="Handles database-style queries like heart rate or time-series data.",
#     functions=[sql_response_gen]
# )

# central_agent = Agent(
#     name="Central Agent",
#     instructions="You are the decision maker. Route query to either RAG or SQL agent.",
#     functions=[lambda: rag_agent, lambda: nl2sql_agent]
# )

# client = Swarm()


# def extract_context(messages):
#     previous_query = previous_response = current_query = None

#     for msg in reversed(messages):
#         if msg["role"] == "user":
#             if current_query is None:
#                 current_query = msg["content"]
#             elif previous_query is None:
#                 previous_query = msg["content"]
#         elif msg["role"] == "assistant" and previous_response is None:
#             previous_response = msg["content"]
#         if previous_query and previous_response and current_query:
#             break

#     return previous_query, previous_response, current_query

# from groq import Groq  # or your OpenAI client setup
# import os

# client = Groq(api_key=os.getenv("OPENAI_API_KEY"))  # or however you initialize your client

# def gpt_fallback(user_query: str) -> str:
#     """
#     Generate a fallback response using OpenAI GPT if nothing is found in the FAISS docs.
#     """
#     system_prompt = (
#         "You are a friendly and knowledgeable medical assistant. "
#         "Always try to provide a helpful, medically sound response. "
#         "Only say 'I don’t know' if you truly have no possible answer. "
#         "Avoid repeating 'I don’t know' multiple times."
#     )

#     try:
#         response = client.chat.completions.create(
#             model=os.getenv("OPENAI_API_MODEL")

# or gpt-3.5-turbo or whatever model you're using
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_query}
#             ]
#         )
#         return response.choices[0].message.content.strip()

#     except Exception as e:
#         print("❌ GPT fallback failed:", e)
#         return "⚠️ Sorry, I couldn't retrieve an answer at the moment."

    

# loaded_indexes = {}  # cache index instances globally

# def normalize(text: str) -> str:
#     return text.strip()  # don't lower, to preserve TB vs tb, RLS vs rls

# def process_query(messages, mode="friendlymode"):
#     previous_query, previous_response, current_query = extract_context(messages)

#     query_context = [
#         {"role": "system", "content": f"Previous Query: {previous_query}"},
#         {"role": "assistant", "content": f"Previous Response: {previous_response}"},
#         {"role": "user", "content": current_query},
#     ]

#     formatted_query = normalize(current_query)
#     mode = mode.lower()

#     if mode == "krishna":
#         result = query_documents(formatted_query, path=os.path.join(FAISS_FOLDER_PATH, "Gita"))
#         if result:
#             answer = result
#         else:
#             answer = "🙏 Sorry, I couldn't find anything in the Bhagavad Gita matching your query."

#     else:
#         all_matches = []

#         for index_name in os.listdir(FAISS_FOLDER_PATH):
#             index_path = os.path.join(FAISS_FOLDER_PATH, index_name)
#             if os.path.isdir(index_path) and index_name.lower() != "gita":
#                 if index_name not in loaded_indexes:
#                     loaded_indexes[index_name] = load_faiss_index(index_path)

#                 index = loaded_indexes[index_name]
#                 retriever = index.as_retriever(search_kwargs={"k": 3})  # increase k for better coverage
#                 result_docs = retriever.invoke(formatted_query)

#                 for doc in result_docs:
#                     content = doc.page_content.strip()
#                     # only append if query words match
#                     if any(word in content.lower() for word in formatted_query.lower().split()):
#                         all_matches.append(content)

#         if not all_matches:
#             answer = "⚠️ Sorry, I couldn't find anything matching your query in the knowledge base."
#         else:
#             answer = all_matches[0]  # return top valid match

#     return apply_personality(answer, mode)



# def process_query(messages, mode="friendlymode"):
#     previous_query, previous_response, current_query = extract_context(messages)

#     query_context = [
#         {"role": "system", "content": f"Previous Query: {previous_query}"},
#         {"role": "assistant", "content": f"Previous Response: {previous_response}"},
#         {"role": "user", "content": current_query},
#     ]

#     formatted_query = normalize(get_response(query_context))
#     mode = mode.lower()

#     if mode == "krishna":
#         result = query_documents(formatted_query, path=os.path.join(FAISS_FOLDER_PATH, "Gita"))
#         if result:
#             answer = f"🌸 **Wisdom from the Gita** 🌸\n\n{result}"
#         else:
#             answer = "🧘 Sorry, I couldn't find this in the Gita. But here’s a thoughtful insight:\n" + gpt_fallback(formatted_query)
#     else:
#         combined_answer = ""
#         found_any = False

#         for index_name in os.listdir(FAISS_FOLDER_PATH):
#             index_path = os.path.join(FAISS_FOLDER_PATH, index_name)
#             if os.path.isdir(index_path) and index_name.lower() != "gita":
#                 result = query_documents(formatted_query, path=index_path)
#                 if result:
#                     found_any = True
#                     combined_answer += f"\n{result.strip()}\n"  # ✅ No source prefix

#         if not found_any:
#             fallback = gpt_fallback(formatted_query)
#             combined_answer = fallback  # ✅ Natural fallback with no label

#         answer = combined_answer.strip()

#     return apply_personality(answer, mode)


# def process_query(messages, mode="friendlymode"):
#     previous_query, previous_response, current_query = extract_context(messages)

#     query_context = [
#         {"role": "system", "content": f"Previous Query: {previous_query}"},
#         {"role": "assistant", "content": f"Previous Response: {previous_response}"},
#         {"role": "user", "content": current_query},
#     ]

#     formatted_query = get_response(query_context)
#     mode = mode.lower()

#     if mode == "krishna":
#         answer = query_documents(formatted_query, path=os.path.join(FAISS_FOLDER_PATH, "Gita"))
#         print("🕉️ Queried Gita Index")
#     else:
#         # Automatically scan all subfolders (except 'Gita')
#         combined_answer = ""
#         for index_name in os.listdir(FAISS_FOLDER_PATH):
#             index_path = os.path.join(FAISS_FOLDER_PATH, index_name)
#             if os.path.isdir(index_path) and index_name.lower() != "gita":
#                 result = query_documents(formatted_query, path=index_path)
#                 if result:
#                     combined_answer += f"\n📚 From {index_name}:\n{result}\n"

#         answer = combined_answer.strip() or "I couldn't find a reliable answer in the health documents."

#     return apply_personality(answer, mode)



# def process_query(messages, mode="friendlymode"):
#     previous_query, previous_response, current_query = extract_context(messages)
    
#     query_context = [
#         {"role": "system", "content": f"Previous Query: {previous_query}"},
#         {"role": "assistant", "content": f"Previous Response: {previous_response}"},
#         {"role": "user", "content": current_query},
#     ]

#     mode = mode.lower()

#     if mode == "krishna":
#         formatted_query = get_response(query_context)
#         answer = query_documents(formatted_query, path="data/faiss_indexes/Gita")
#         print("from Gita")
        

#         # formatted_query = get_response(query_context)

#         # # Load the FAISS index
#         # vector_path = "data/faiss_indexes/Gita"
#         # embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
#         # vectorstore = FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)
#         # retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

#         # # Get top matching verse
#         # # top_doc = retriever.get_relevant_documents(formatted_query)[0]
#         # top_doc = retriever.

#         # exact_verse = top_doc.page_content.strip()
#         # verse_meta = top_doc.metadata.get("verse", "Bhagavad Gita Verse")

#         # # Ask OpenAI to explain the verse
#         # openai_client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
#         # gpt_response = openai_client.chat.completions.create(
#         #     model=os.getenv("OPENAI_API_MODEL")


#         #     messages=[
#         #         {
#         #             "role": "system",
#         #             "content": "Explain the following Bhagavad Gita verse in simple and spiritual language:"
#         #         },
#         #         {
#         #             "role": "user",
#         #             "content": exact_verse
#         #         }
#         #     ]
#         # )
#         # explanation = gpt_response.choices[0].message.content.strip()

#         # # Combine verse + explanation
#         # answer = (
#         #     f"🌸 **Wisdom from the Bhagavad Gita** 🌸\n\n"
#         #     f"🕉️ *{verse_meta}*\n\n"
#         #     f"📖 \"{exact_verse}\"\n\n"
#         #     f"🙏 *Simple Meaning:* {explanation}"
#         # )

#     else:  # both scientific and friendlymode use SleepParent
#         formatted_query = get_response(query_context)
#         answer = query_documents(formatted_query, path="data/faiss_indexes/Sleep_Knowledge_Complete_Guide")

#     return apply_personality(answer, mode)


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

