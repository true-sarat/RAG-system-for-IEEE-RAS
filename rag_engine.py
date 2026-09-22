import google.generativeai as genai

from config import (
    COLLECTION_NAME,
    GEMINI_MODEL,
    RELEVANCE_DISTANCE_CUTOFF,
    TOP_K,
    get_client,
    get_embed_fn,
    get_secret,
)

SYSTEM_PROMPT = """You are the IEEE RAS chapter assistant.

Answer ONLY using the context passages provided below. If they do not contain
enough information to answer confidently, say plainly that you don't have that
information, and suggest the user check the official IEEE RAS pages - do not
guess or use outside knowledge.

Style rules:
- Be warm and direct, but thorough: aim for a short paragraph, or 3-5 sentences,
  rather than a single line. Use bullet points for lists (benefits, steps,
  event types) when the passages support it.
- No greetings, no "Hello!", no filler openers - get straight into the answer.
- Never mention file names (e.g. "about_ieee_ras.txt") or the words "context",
  "chunk" or "passage" - sources are shown separately in the UI.
- Always finish your sentences.
"""

NO_ANSWER_MESSAGE = (
    "I don't have information about that in my knowledge base. "
    "Try asking about IEEE RAS, chapter membership, or chapter events."
)

GENERATION_CONFIG = {
    "temperature": 0.3,
    "max_output_tokens": 800,
}


def _get_collection():
    return get_client().get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embed_fn(),
        metadata={"hnsw:space": "cosine"},
    )


def retrieve(query: str, k: int = TOP_K):
    collection = _get_collection()
    total = collection.count()
    if total == 0:
        return []

    results = collection.query(query_texts=[query], n_results=min(k, total))

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
    context_block = "\n\n".join(
        f"[Passage {i + 1}]\n{c['text']}" for i, c in enumerate(chunks)
    )

    history_block = ""
    if chat_history:
        recent = chat_history[-4:]
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


def _generate(prompt: str) -> str:
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt, generation_config=GENERATION_CONFIG)
        return (getattr(response, "text", "") or "").strip() or NO_ANSWER_MESSAGE
    except Exception as exc:
        return _friendly_error(exc)


def _generate_stream(prompt: str):
    produced = False
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            prompt, generation_config=GENERATION_CONFIG, stream=True
        )
        for part in response:
            text = getattr(part, "text", "") or ""
            if text:
                produced = True
                yield text
    except Exception as exc:
        yield ("" if produced else "") + _friendly_error(exc)
        return

    if not produced:
        yield NO_ANSWER_MESSAGE


def answer_question(query: str, chat_history=None, stream: bool = False):
    query = (query or "").strip()
    if not query:
        return {"answer": NO_ANSWER_MESSAGE, "sources": [], "chunks": []}

    try:
        chunks = retrieve(query)
    except Exception as exc:
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

    sources = []
    for chunk in relevant:
        if chunk["source"] not in sources:
            sources.append(chunk["source"])

    genai.configure(api_key=api_key)
    prompt = _build_prompt(query, relevant, chat_history)

    if stream:
        return {
            "answer": _generate_stream(prompt),
            "sources": sources,
            "chunks": relevant,
        }

    return {"answer": _generate(prompt), "sources": sources, "chunks": relevant}


def _friendly_error(exc: Exception) -> str:
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