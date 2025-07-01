from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()



def load_faiss_index(path):
    try:
        embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE_URL"),
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        )
        
        index = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        print(f"✅ Successfully loaded FAISS index from {path}")
        return index
        
    except Exception as e:
        print(f"❌ Failed to load FAISS index from {path}: {e}")
        return None



# def query_documents(query: str, path: str) -> str:
#     # print("-------------------FolderPath-------------------", FAISS_FOLDER_PATH)
#     # vector_path = "C:\\Users\\phefa\\OneDrive\\Desktop\\Teach Jewellery\\Knowledge_Base_chatbot\\fastapp_structure\\data\\faiss_indexes\\Parkinson disease"
#     vector_path = path
#     # Use Hugging Face embeddings (free, runs locally)
#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
#         encode_kwargs={'normalize_embeddings': True}
#     )
    
#     vectorstore = FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

#     # Use DeepSeek for chat/LLM
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=ChatOpenAI(
#             api_key=os.getenv("DEEPSEEK_API_KEY"),
#             base_url="https://api.deepseek.com/v1",
#             model="deepseek-chat"
#         ),
#         retriever=retriever,
#         return_source_documents=False
#     )

#     # Use invoke instead of deprecated run method
#     return qa_chain.invoke({"query": query})["result"]


def query_documents(query: str, path: str, top_k: int = 3) -> List[str]:
    """
    Retrieve top-k relevant documents from a FAISS vector store using HuggingFace embeddings.
    No LLM is used here. Returns a list of strings (document content).
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    results = retriever.invoke(query)

    return [doc.page_content.strip() for doc in results]

if __name__ == "__main__":
    query = "What is Parkinson disease?"
    result = query_documents(query)
    print(f"🤖 Answer: {result}")

