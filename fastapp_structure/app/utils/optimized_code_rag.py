# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from typing import List
# import os
# from dotenv import load_dotenv

# load_dotenv()


# def load_faiss_index(path):
#     try:
#         embeddings = OpenAIEmbeddings(
#             api_key=os.getenv("OPENAI_API_KEY"),
#             base_url=os.getenv("OPENAI_API_BASE_URL"),
#             model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
#         )
        
#         index = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
#         print(f"✅ Successfully loaded FAISS index from {path}")
#         return index
        
#     except Exception as e:
#         print(f"❌ Failed to load FAISS index from {path}: {e}")
#         return None



# # def query_documents(query: str, path: str) -> str:
# #     # print("-------------------FolderPath-------------------", FAISS_FOLDER_PATH)
# #     # vector_path = "C:\\Users\\phefa\\OneDrive\\Desktop\\Teach Jewellery\\Knowledge_Base_chatbot\\fastapp_structure\\data\\faiss_indexes\\Parkinson disease"
# #     vector_path = path
# #     # Use Hugging Face embeddings (free, runs locally)
# #     embeddings = HuggingFaceEmbeddings(
# #         model_name="sentence-transformers/all-MiniLM-L6-v2",
# #         model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
# #         encode_kwargs={'normalize_embeddings': True}
# #     )
    
# #     vectorstore = FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)
# #     retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# #     # Use DeepSeek for chat/LLM
# #     qa_chain = RetrievalQA.from_chain_type(
# #         llm=ChatOpenAI(
# #             api_key=os.getenv("DEEPSEEK_API_KEY"),
# #             base_url="https://api.deepseek.com/v1",
# #             model="deepseek-chat"
# #         ),
# #         retriever=retriever,
# #         return_source_documents=False
# #     )

# #     # Use invoke instead of deprecated run method
# #     return qa_chain.invoke({"query": query})["result"]


# def query_documents(query: str, path: str, top_k: int = 3) -> List[str]:
#     """
#     Retrieve top-k relevant documents from a FAISS vector store using HuggingFace embeddings.
#     No LLM is used here. Returns a list of strings (document content).
#     """
#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={'device': 'cpu'},
#         encode_kwargs={'normalize_embeddings': True}
#     )

#     vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
#     retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
#     results = retriever.invoke(query)

#     return [doc.page_content.strip() for doc in results]

# if __name__ == "__main__":
#     query = "What is Parkinson disease?"
#     result = query_documents(query)
#     print(f"🤖 Answer: {result}")


import os, time
from functools import lru_cache
from typing import List, Literal
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

EMBEDDER_BACKEND: Literal["hf","openai"] = os.getenv("KB_EMBEDDINGS","hf").lower()
HF_MODEL   = os.getenv("KB_HF_MODEL","sentence-transformers/all-MiniLM-L6-v2")
HF_DEVICE  = os.getenv("KB_HF_DEVICE","cpu")
HF_NORM    = os.getenv("KB_HF_NORMALIZE","true").lower()=="true"
OPENAI_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL","text-embedding-3-small")
OPENAI_KEY   = os.getenv("OPENAI_API_KEYS")
OPENAI_BASE  = os.getenv("OPENAI_API_BASE_URL")

def _canon(p: str) -> str:
    # Make the cache key stable on Windows: same path string → same cache entry
    return os.path.normcase(os.path.realpath(os.path.abspath(p)))

@lru_cache(maxsize=1)
def get_embeddings():
    if EMBEDDER_BACKEND == "openai":
        if not OPENAI_KEY:
            raise RuntimeError("OPENAI_API_KEY missing but KB_EMBEDDINGS=openai")
        return OpenAIEmbeddings(api_key=OPENAI_KEY, base_url=OPENAI_BASE, model=OPENAI_MODEL)
    return HuggingFaceEmbeddings(
        model_name=HF_MODEL,
        model_kwargs={"device": HF_DEVICE},
        encode_kwargs={"normalize_embeddings": HF_NORM},
    )

def _assert_dims_match(vs: FAISS, embeddings, index_path: str):
    test_vec = embeddings.embed_query("dim_check")
    if len(test_vec) != vs.index.d:
        raise RuntimeError(
            f"Embedding dim mismatch for {index_path}: embedder={len(test_vec)} vs index={vs.index.d}. "
            "Use the SAME embedder at build and query time."
        )

@lru_cache(maxsize=128)
def get_vectorstore(index_dir: str) -> FAISS:
    embeddings = get_embeddings()
    vs = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    _assert_dims_match(vs, embeddings, index_dir)
    print(f"✅ FAISS ready: {index_dir}")
    return vs

def load_faiss_index(path: str) -> FAISS:
    return get_vectorstore(_canon(path))


KB_MIN_SCORE = 0.15  # tune this (0.3–0.4 works well)
KB_TOPK = 3
KB_FINAL_K = 3

def query_documents(query: str, path: str) -> list[str]:
    vs = get_vectorstore(_canon(path))
    docs_scores = vs.similarity_search_with_relevance_scores(query, k=KB_TOPK)

    kept: list[str] = []
    for doc, score in docs_scores:
        if score is None or score < KB_MIN_SCORE:
            continue
        kept.append(doc.page_content.strip())
        if len(kept) >= KB_FINAL_K:
            break

    return kept

# def query_documents(query: str, path: str, top_k: int = 3) -> List[str]:
#     t0 = time.perf_counter()
#     vs = get_vectorstore(_canon(path))   # cold: loads once; warm: ~0ms
#     t1 = time.perf_counter()
#     docs = vs.similarity_search(query, k=top_k)
#     t2 = time.perf_counter()
#     print("⏱️ timings | load_cache={:.3f}s | search={:.3f}s | total={:.3f}s"
#           .format(t1-t0, t2-t1, t2-t0))
#     return [d.page_content.strip() for d in docs]
# def query_documents(query: str, path: str, top_k: int = 3) -> List[str]:
#     """
#     Retrieve top-k relevant documents from a FAISS vector store using HuggingFace embeddings.
#     No LLM is used here. Returns a list of strings (document content).
#     """
#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={'device': 'cpu'},
#         encode_kwargs={'normalize_embeddings': True}
#     )

#     vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
#     retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
#     results = retriever.invoke(query)

#     return [doc.page_content.strip() for doc in results]


# --- Prewarm helpers ---
def list_index_dirs(base_dir: str) -> list[str]:
    base = _canon(base_dir)
    out = []
    for name in os.listdir(base):
        d = _canon(os.path.join(base, name))
        if os.path.isdir(d) and os.path.exists(os.path.join(d,"index.faiss")) and os.path.exists(os.path.join(d,"index.pkl")):
            out.append(d)
    return out

def prewarm_indexes(base_dir: str, limit: int | None = None) -> int:
    """Load embeddings and the first N indexes into memory at startup."""
    get_embeddings()  # init once
    count = 0
    for d in list_index_dirs(base_dir):
        get_vectorstore(d)  # cached after first load
        count += 1
        if limit and count >= limit:
            break
    print(f"🔥 Prewarmed {count} FAISS index(es).")
    return count

