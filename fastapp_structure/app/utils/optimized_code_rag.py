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

from __future__ import annotations

import os
import re
import json
import logging
import textwrap
from typing import Any, Dict, List, Tuple, Optional

import groq
from groq import Groq

import os, time
from functools import lru_cache
from typing import List, Literal
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import os, textwrap, re
from openai import OpenAI
from groq import Groq
import os, json, math, re, textwrap
from typing import Any, Dict, List, Optional, Union

client = Groq(api_key=os.getenv("OPENAI_API_KEY"))

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

# def query_documents(query: str, path: str) -> list[str]:
#     vs = get_vectorstore(_canon(path))
#     docs_scores = vs.similarity_search_with_relevance_scores(query, k=KB_TOPK)

#     kept: list[str] = []
#     for doc, score in docs_scores:
#         if score is None or score < KB_MIN_SCORE:
#             continue
#         kept.append(doc.page_content.strip())
#         if len(kept) >= KB_FINAL_K:
#             break

#     return kept


def query_documents(query: str, path: str) -> list[dict[str, Any]]:
    vs = get_vectorstore(_canon(path))
    docs_scores = vs.similarity_search_with_relevance_scores(query, k=KB_TOPK)

    kept: list[dict[str, Any]] = []
    for doc, score in docs_scores:
        if score is None or score < KB_MIN_SCORE:
            continue
        kept.append({
            "text": doc.page_content.strip(),
            "metadata": getattr(doc, "metadata", {}) or {},
            "score": float(score),
        })
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




# to get formated response
"""
kb_answer_groq.py
-----------------

Groq-grounded answer generator with robust error handling.

- Uses Groq chat models via the official 'groq' SDK
- Normalizes KB snippets (strings -> dicts with text/source/score)
- Guarantees string return (never None) to avoid "Context Reply None"
- Provides a helper to do KB-first, then fallback to a general model

Env:
  GROQ_API_KEY

pip:
  pip install groq
"""



# ───────────────────────── Logging ─────────────────────────

logger = logging.getLogger("kb")
if not logger.handlers:
    _h = logging.StreamHandler()
    _fmt = logging.Formatter("[%(levelname)s] %(message)s")
    _h.setFormatter(_fmt)
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ─────────────────────── Groq Client ───────────────────────

# Common production Groq chat models (update as needed).
SUPPORTED_GROQ_CHAT_MODELS = {
    "llama-3.1-8b-instant",     # Fast + very cheap, 128k ctx
    "llama-3.3-70b-versatile",  # Higher quality, 128k ctx
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "openai/gpt-oss-20b",       # Example OpenAI-compat OSS model on Groq
}

DEFAULT_MODEL = "llama-3.1-8b-instant"


def groq_client() -> Groq:
    api_key = os.getenv("OPENAI_API_KEY") 
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in environment")
    # Base URL defaults to Groq's API; override via GROQ_BASE_URL if needed.
    return Groq(api_key=api_key)


def normalize_model(m: Optional[str]) -> str:
    if not m:
        return DEFAULT_MODEL
    m = m.strip()
    # Be lenient about casing
    lo = m.lower()
    for allowed in SUPPORTED_GROQ_CHAT_MODELS:
        if lo == allowed.lower():
            return allowed
    # Fallback to a safe default if unknown (prevents "'model' ... anyOf" errors)
    return DEFAULT_MODEL


# ───────────────────── Utility: Errors ─────────────────────

def _format_groq_error(exc: Exception) -> str:
    """
    Produce a compact, readable error string for Groq SDK errors.
    """
    # The groq SDK exposes rich error classes with .status_code and .response
    if isinstance(exc, groq.APIStatusError):
        try:
            body = exc.response.json()
        except Exception:
            body = exc.response.text if getattr(exc, "response", None) else str(exc)
        code = getattr(exc, "status_code", "unknown")
        return f"Error code: {code} - {json.dumps(body) if isinstance(body, dict) else body}"
    if isinstance(exc, groq.APIConnectionError):
        return "Connection error contacting Groq API"
    if isinstance(exc, groq.RateLimitError):
        return "Rate limit exceeded on Groq API"
    # Generic fallback
    return str(exc)


# ─────────────── Utility: KB Snippet Preparation ───────────────

def _normalize_kb_snippets(kb_snippets: List[Any]) -> List[Dict[str, Any]]:
    """
    Ensure each snippet has: {"text": str, "source": str, "score": float}
    Drop empty texts. Sort by score desc.
    """
    norm: List[Dict[str, Any]] = []
    for i, s in enumerate(kb_snippets, start=1):
        if isinstance(s, str):
            d = {"text": s, "source": f"snippet-{i}", "score": 0.0}
        elif isinstance(s, dict):
            d = {
                "text": (s.get("text") or "").strip(),
                "source": s.get("source", f"snippet-{i}"),
                "score": float(s.get("score", 0.0)),
            }
        else:
            # Coerce unexpected types to string
            d = {"text": str(s), "source": f"snippet-{i}", "score": 0.0}

        if d["text"]:
            norm.append(d)

    norm.sort(key=lambda x: x["score"], reverse=True)
    return norm


def _build_context_block(sorted_snips: List[Dict[str, Any]], max_snippets: int = 12) -> str:
    parts: List[str] = []
    for i, s in enumerate(sorted_snips[:max_snippets], start=1):
        txt = re.sub(r"\s+", " ", s["text"]).strip()
        src = s["source"]
        parts.append(f"[S{i}] Source: {src}\n{txt}")
    return "\n\n".join(parts)


def _dedupe_lines(text: str) -> str:
    """
    Remove exact duplicate lines while preserving spacing reasonably.
    Also collapses multiple blank lines into a single blank line.
    """
    seen = set()
    out = []
    last_blank = False
    for raw in text.splitlines():
        line = raw.rstrip()
        key = line.strip()
        if not key:
            if not last_blank and out:
                out.append("")   # keep a single blank line
            last_blank = True
            continue
        last_blank = False
        if key not in seen:
            out.append(line)
            seen.add(key)
    # Final pass: trim extra blank lines at start/end
    result = "\n".join(out).strip()
    return result


# ─────────────── KB-grounded Answer Generator ───────────────

def generate_answer_from_context(
    question: str,
    kb_snippets: List[Any],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 600,
) -> str:
    """
    Create a grounded answer from user question and knowledge base snippets.

    Returns a STRING always. On any failure, returns a "Knowledge base error: ..." message.

    kb_snippets can be a list of dicts (text, source, score) or a list of strings.
    """
    try:
        if not kb_snippets:
            return "I couldn’t find enough information in the knowledge base to answer that."

        normalized = _normalize_kb_snippets(kb_snippets)
        if not normalized:
            return "I couldn’t find enough information in the knowledge base to answer that."

        context = _build_context_block(normalized)

        system_msg = textwrap.dedent("""
        You are a precise assistant. Use ONLY the provided sources to answer.
        - Support each claim with inline citations like [S1].
        - Provide a single, concise answer (no repeated lists or sentences).
        - Do not restate the same information twice.
        - If the sources don't contain the answer, say so.
        """).strip()

        
        user_msg = f"QUESTION:\n{question.strip()}\n\nSOURCES:\n{context}"

        client = groq_client()
        safe_model = normalize_model(model)

        resp = client.chat.completions.create(
            model=safe_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = (resp.choices[0].message.content or "").strip()
        content = _dedupe_lines(content)  # ✅ remove duplicate lines/sentences
        return content if content else "I couldn’t find enough information in the knowledge base to answer that."
    except Exception as exc:
        return f"Knowledge base error: {_format_groq_error(exc)}"




# ─────────────── Generic (non-KB) Fallback on Groq ───────────────

def generate_general_answer(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 500,
) -> str:
    """
    Simple general Groq chat call to use as fallback if KB fails.
    Always returns a string; on error, returns an error string.
    """
    try:
        client = groq_client()
        safe_model = normalize_model(model)
        resp = client.chat.completions.create(
            model=safe_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        return f"General model error: {_format_groq_error(exc)}"


# ─────────────── High-level: KB-first then fallback ───────────────

def answer_with_kb_then_fallback(
    question: str,
    kb_snippets: List[Any],
    model: str = DEFAULT_MODEL,
) -> Tuple[Optional[str], str]:
    """
    Try KB-grounded answer first. If it fails or has no KB info, fall back.

    Returns: (context_reply, final_reply)
      - context_reply: None if KB unavailable/insufficient; else the KB-grounded text
      - final_reply: what you should show to user (KB answer if present, else fallback)
    """
    kb_text = generate_answer_from_context(question, kb_snippets, model=model)
    logger.debug("KB text (first 2k chars): %s", kb_text[:2000])

    context_reply: Optional[str] = None
    if kb_text.startswith("Knowledge base error:"):
        logger.error(kb_text)  # keep the underlying error visible
    elif "I couldn’t find enough information" in kb_text:
        logger.info("KB had insufficient info for the question.")
    else:
        context_reply = kb_text

    if context_reply:
        return context_reply, context_reply

    # Fallback to a general, safe answer
    fallback_prompt = (
        "Answer the user's question clearly and concisely.\n\n"
        f"Question: {question.strip()}"
    )
    general = generate_general_answer(fallback_prompt, model=model)
    return None, general


# ─────────────── Example FAISS retrieval (stub) ───────────────
# Replace with your actual FAISS retrieval (must return list[dict]).

def dummy_retrieve_from_faiss(question: str) -> List[Dict[str, Any]]:
    return []


# ───────────────────────── CLI demo ─────────────────────────

if __name__ == "__main__":
    q = "What is a healthy resting heart rate for adults?"
    kb = [
        {"text": "Normal resting heart rate for adults is typically 60–100 bpm.", "source": "heartrate_guide.md", "score": 0.92},
        {"text": "Athletes may have lower resting heart rates.", "source": "sports_cardio.pdf", "score": 0.55},
    ]
    ctx, reply = answer_with_kb_then_fallback(q, kb, model=DEFAULT_MODEL)
    print("Context Reply:", ctx)
    print("Final Reply:", reply)



MongoLike = Union[str, Dict[str, Any], List[Any]]

def format_mongo_answer_llm(
    question: str,
    mongo_result: MongoLike,
    *,
    use_llm: bool = True,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 250
) -> str:
    """
    Accepts:
      - str (already formatted narrative) -> returns as-is
      - dict (metrics OR summary keys) -> formats nicely
      - list (of dicts/rows) -> summarizes
    """

    # 0) Nothing?
    if mongo_result is None:
        return "I couldn’t find any data for your request."

    # 1) Already formatted narrative (string) -> pass through
    if isinstance(mongo_result, str):
        s = mongo_result.strip()
        return s if s else "I couldn’t find any data for your request."

    # 2) If list -> collapse/summarize
    if isinstance(mongo_result, list):
        if len(mongo_result) == 0:
            return "I couldn’t find any data for your request."
        if len(mongo_result) == 1:
            # recurse on the single item
            return format_mongo_answer_llm(
                question, mongo_result[0],
                use_llm=use_llm, model=model, temperature=temperature, max_tokens=max_tokens
            )
        # multiple rows -> LLM summarize or compact bullet list fallback
        if use_llm and client:
            data_json = json.dumps({"question": question, "rows": mongo_result}, ensure_ascii=False)
            system_msg = (
                "You are a precise assistant. Produce a short, user-friendly answer "
                "ONLY using the provided JSON rows. If numeric, summarize with clear units."
            )
            user_msg = f"Format these rows for the user:\n{data_json}\nReturn only the final answer."
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role":"system","content":system_msg},{"role":"user","content":user_msg}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                ans = (resp.choices[0].message.content or "").strip()
                if ans:
                    return ans
            except Exception:
                pass

        # fallback compact bullets
        lines = []
        for i, row in enumerate(mongo_result[:10], start=1):
            if isinstance(row, dict):
                row2 = {k:v for k,v in row.items() if k != "_id"}
                pretty = ", ".join(f"{k}: {row2[k]}" for k in list(row2)[:4])
                lines.append(f"- Row {i}: {pretty}")
            else:
                lines.append(f"- Row {i}: {row}")
        more = "" if len(mongo_result) <= 10 else f"\n(+{len(mongo_result)-10} more)"
        return "Here’s a quick summary:\n" + "\n".join(lines) + more

    # 3) Dict case
    if isinstance(mongo_result, dict):
        # remove _id noise
        doc = {k: v for k, v in mongo_result.items() if k != "_id"}

        if not doc:
            return "I couldn’t find any data for your request."

        # 3a) If the dict already carries narrative keys → return that content
        for k in ["personal_context", "summary", "text", "message", "formatted", "answer"]:
            if k in doc and isinstance(doc[k], str) and doc[k].strip():
                return doc[k].strip()

        # 3b) Metrics dict → either LLM or deterministic format
        # Normalize floats (round)
        norm = {}
        for k, v in doc.items():
            if isinstance(v, float):
                v = round(v) if abs(v - round(v)) < 1e-6 else round(v, 2)
            norm[k] = v

        # If exactly one metric, try to phrase naturally (deterministic)
        if len(norm) == 1 and not use_llm:
            k, v = next(iter(norm.items()))
            key = k.lower()
            if any(t in key for t in ["heartrate", "heart_rate", "hr"]):
                return f"Your average heart rate is {v} bpm."
            if "spo2" in key:
                return f"Your average SpO₂ is {v}%."
            if "steps" in key:
                try:
                    return f"You took {int(v):,} steps."
                except Exception:
                    return f"You took {v} steps."
            if "sleep" in key or key.endswith("_hours"):
                return f"Your total sleep duration is {v} hours."
            return f"{k.replace('_',' ').capitalize()}: {v}"

        # LLM formatting path for nicer language/units
        if use_llm and client:
            # Unit hints
            unit_hints = {}
            for k in norm:
                kl = k.lower()
                if "heartrate" in kl or "heart_rate" in kl or kl.endswith("_hr") or "bpm" in kl:
                    unit_hints[k] = "bpm"
                elif "steps" in kl:
                    unit_hints[k] = "steps"
                elif "spo2" in kl:
                    unit_hints[k] = "%"
                elif "sleep" in kl or kl.endswith("_hours") or "hour" in kl:
                    unit_hints[k] = "hours"
                elif "calorie" in kl or "kcal" in kl:
                    unit_hints[k] = "kcal"

            payload = json.dumps(
                {"question": question, "fields": norm, "units": unit_hints},
                ensure_ascii=False
            )
            system_msg = textwrap.dedent("""
            You are a precise assistant. Answer ONLY using the provided JSON.
            - Do NOT invent values.
            - Use 'units' hints (bpm, %, hours, steps, kcal) where relevant.
            - One field -> short natural sentence.
            - Multiple fields -> concise readable summary (short lines/bullets).
            - Keep it friendly and factual. No extra markdown unless bullets.
            """).strip()
            user_msg = f"Format this for the user:\n{payload}\nReturn only the final answer."

            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role":"system","content":system_msg},{"role":"user","content":user_msg}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                ans = (resp.choices[0].message.content or "").strip()
                if ans:
                    return ans
            except Exception:
                pass

        # deterministic fallback (no LLM or error)
        if len(norm) == 1:
            k, v = next(iter(norm.items()))
            pretty = k.replace("_", " ").capitalize()
            # quick units
            u = ""
            kl = k.lower()
            if any(t in kl for t in ["heartrate", "heart_rate", "hr"]): u = " bpm"
            elif "spo2" in kl: u = " %"
            elif "steps" in kl: u = " steps"
            elif "sleep" in kl or kl.endswith("_hours"): u = " hours"
            elif "calorie" in kl or "kcal" in kl: u = " kcal"
            try:
                if u.strip() == "steps" and isinstance(v, (int, float)):
                    v = f"{int(v):,}"
            except Exception:
                pass
            return f"{pretty}: {v}{u}".strip()

        lines = []
        for k, v in norm.items():
            pretty = k.replace("_", " ").capitalize()
            u = ""
            kl = k.lower()
            if any(t in kl for t in ["heartrate", "heart_rate", "hr"]): u = " bpm"
            elif "spo2" in kl: u = " %"
            elif "steps" in kl:
                try: v = f"{int(v):,}"
                except Exception: pass
                u = " steps"
            elif "sleep" in kl or kl.endswith("_hours"): u = " hours"
            elif "calorie" in kl or "kcal" in kl: u = " kcal"
            lines.append(f"- {pretty}: {v}{u}".strip())
        return "Here’s what I found:\n" + "\n".join(lines)

    # 4) Unknown type
    return "I couldn’t interpret the data returned for your request."
