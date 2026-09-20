"""
rag_engine.py
-------------
Retrieval-Augmented Generation core:

  query -> embed -> ChromaDB similarity search -> relevance filter
        -> Gemini answers using ONLY the retrieved chunks

Design choices that matter:
  * The collection handle is fetched fresh on each call (never cached), which
    avoids stale-handle errors when the app restarts or two users hit it at once.
  * Chunks further away than RELEVANCE_DISTANCE_CUTOFF are dropped, so the
    assistant says "I don't have that information" instead of hallucinating.
  * Every failure (missing key, quota, bad model name) returns a friendly
    message instead of crashing the Streamlit page with a red traceback.
"""

import chromadb
import google.generativeai as genai
from chromadb.utils import embedding_functions

from config import (
    COLLECTION_NAME,
    DB_DIR,
    EMBED_MODEL_NAME,
    GEMINI_MODEL,
    RELEVANCE_DISTANCE_CUTOFF,
    TOP_K,
    get_secret,
)

SYSTEM_PROMPT = """You are the IEEE RAS chapter assistant.

Answer ONLY using the context passages provided below. If they do not contain
enough information to answer confidently, say plainly that you don't have that
information, and suggest the user check the official IEEE RAS pages — do not
guess or use outside knowledge.

Style rules:
- Be concise, warm and direct. No greetings, no "Hello!", no filler openers.
- Never mention file names (e.g. "about_ieee_ras.txt") or the words "context",
  "chunk" or "passage" in your answer — sources are shown separately in the UI.
- Prefer short paragraphs or bullets. Finish your sentences.
"""

NO_ANSWER_MESSAGE = (
    "I don't have information about that in my knowledge base. "
    "Try asking about IEEE RAS, chapter membership, or chapter events."
)

# The embedding model is heavy to load, so keep one instance per process.
_embed_fn = None


def _get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        _embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL_NAME
        )
    return _embed_fn


def _get_collection():
    """
    Always return a FRESH collection handle.

    Caching the handle caused InvalidCollectionException whenever the
    underlying database was rebuilt (app restart, concurrent first load).
    Re-fetching is cheap; the embedding model is what's expensive, and that
    is still cached above.
    """
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_get_embed_fn(),
        metadata={"hnsw:space": "cosine"},
    )


def retrieve(query: str, k: int = TOP_K):
    """Return the top-k chunks for a query, each with its source and distance."""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(k, collection.count()),
    )

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    chunks = []
    for text, meta, distance in zip(documents, metadatas, distances):
        chunks.append(
            {
                "text": text,
                "source": (meta or {}).get("source", "unknown"),
                "distance": float(distance),
            }
        )
    return chunks


def _build_prompt(query: str, chunks, chat_history=None):
    """Assemble system rules + recent conversation + retrieved context."""
    context_block = "\n\n".join(
        f"[Passage {i + 1}]\n{c['text']}" for i, c in enumerate(chunks)
    )

    history_block = ""
    if chat_history:
        recent = chat_history[-6:]  # keep the last ~3 exchanges for follow-ups
        lines = []
        for message in recent:
            speaker = "User" if message.get("role") == "user" else "Assistant"
            lines.append(f"{speaker}: {message.get('content', '')}")
        history_block = "Recent conversation (for follow-up context):\n" + "\n".join(lines)

    parts = [SYSTEM_PROMPT]
    if history_block:
        parts.append(history_block)
    parts.append("Context passages:\n" + context_block)
    parts.append(f"User question: {query}\n\nAnswer:")
    return "\n\n---\n\n".join(parts)


def answer_question(query: str, chat_history=None):
    """
    Main entry point used by app.py.

    Returns: {"answer": str, "sources": [str], "chunks": [dict]}
    """
    query = (query or "").strip()
    if not query:
        return {"answer": NO_ANSWER_MESSAGE, "sources": [], "chunks": []}

    try:
        chunks = retrieve(query)
    except Exception as exc:  # database missing / corrupted
        return {
            "answer": (
                "I couldn't reach the knowledge base just now. "
                f"Please refresh the page and try again. (Details: {exc})"
            ),
            "sources": [],
            "chunks": [],
        }

    relevant = [c for c in chunks if c["distance"] <= RELEVANCE_DISTANCE_CUTOFF]
    if not relevant:
        # Honest fallback — this is the behaviour that separates a real RAG
        # system from a chatbot that confidently makes things up.
        return {"answer": NO_ANSWER_MESSAGE, "sources": [], "chunks": chunks}

    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        return {
            "answer": (
                "No Gemini API key is configured. Add GEMINI_API_KEY to your .env "
                "file locally, or to Secrets in Streamlit Cloud when deployed."
            ),
            "sources": [],
            "chunks": relevant,
        }

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            _build_prompt(query, relevant, chat_history),
            generation_config={"temperature": 0.2, "max_output_tokens": 800},
        )
        answer = (getattr(response, "text", "") or "").strip() or NO_ANSWER_MESSAGE
    except Exception as exc:
        answer = _friendly_error(exc)
        return {"answer": answer, "sources": [], "chunks": relevant}

    sources = []
    for chunk in relevant:
        if chunk["source"] not in sources:
            sources.append(chunk["source"])

    return {"answer": answer, "sources": sources, "chunks": relevant}


def _friendly_error(exc: Exception) -> str:
    """Turn raw API exceptions into something a user can act on."""
    name = type(exc).__name__
    text = str(exc)

    if "ResourceExhausted" in name or "429" in text or "quota" in text.lower():
        return (
            "The Gemini free-tier quota is currently exhausted, so I can't generate "
            "an answer right now. Per-minute limits clear in about a minute; daily "
            f"limits reset at midnight Pacific time. (Model: {GEMINI_MODEL})"
        )
    if "NotFound" in name or "404" in text:
        return (
            f"The model '{GEMINI_MODEL}' isn't available for this API key. "
            "Set GEMINI_MODEL in your .env (or Streamlit Secrets) to a current "
            "model listed at https://ai.google.dev/gemini-api/docs/models"
        )
    if "PermissionDenied" in name or "API key" in text or "401" in text or "403" in text:
        return (
            "The Gemini API key was rejected. Generate a new one at "
            "https://aistudio.google.com/apikey and update GEMINI_API_KEY."
        )
    return f"Something went wrong while generating the answer: {name}: {text}"
