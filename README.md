# IEEE RAS Chapter Assistant (RAG)

A retrieval-augmented generation chatbot that answers questions about IEEE RAS using
a curated local knowledge base rather than the language model's general training data.
Every answer is grounded in retrieved passages, cites its sources, and the assistant
says "I don't have that information" when nothing relevant is found.

**Live app:** _add your Streamlit link here_
**Repo:** _this repository_

## Architecture

```
User question
     |
     v
[Embed query]  sentence-transformers / all-MiniLM-L6-v2  (local, free)
     |
     v
[ChromaDB cosine search]  ->  top-k passages + source + distance
     |
     v
[Relevance cutoff]  ->  nothing close enough?  ->  "I don't have that information"
     |
     v
[Gemini]  answers using ONLY the retrieved passages
     |
     v
[Streamlit UI]  answer + source citations + expandable retrieved context
```

## Stack

| Layer        | Tool                                    | Why                                   |
|--------------|-----------------------------------------|---------------------------------------|
| Embeddings   | sentence-transformers (all-MiniLM-L6-v2)| Free, runs locally, no API cost       |
| Vector store | ChromaDB (persistent)                   | Zero-config, file-based               |
| LLM          | Gemini API                              | Free tier, fast, good quality         |
| UI           | Streamlit                               | Quick to build, free hosting          |

## Project structure

```
app.py            Streamlit UI (chat, citations, retrieved-context panel)
rag_engine.py     retrieval + relevance filtering + Gemini generation
ingest.py         chunk, embed and store everything in data/
config.py         shared paths, model names, secret loading
data/             the knowledge base (.txt files) - REPLACE THE PLACEHOLDERS
requirements.txt
.env.example
```



## Example Q&A

what should give a reply: "What is IEEE RAS?" 
"How do I join the chapter?"
     "What events does the chapter run?"
     "Do I need robotics experience to join?"
     "Is there a membership fee?"
what would give a "i don't know" reply:
     "What's the capital of France?"

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `404 model not found` | Model name retired. Set `GEMINI_MODEL` in `.env` **and** in Streamlit Secrets to a current model from https://ai.google.dev/gemini-api/docs/models |
| Quota / `ResourceExhausted` | Free-tier limit. Per-minute limits clear in ~60s; daily limits reset at midnight Pacific. Switching to `gemini-3.5-flash-lite` gives more headroom. |
| "No Gemini API key is configured" | `.env` missing locally, or Secrets not saved on Streamlit Cloud. They are separate — updating one does not update the other. |
| `Failed to send telemetry event` | Harmless ChromaDB analytics noise. Ignore it. |
| Answer wrongly says "I don't have that information" | Raise `RELEVANCE_DISTANCE_CUTOFF` in `config.py` (cosine distance, 0–2; 0.75 default). Lower it to make the assistant stricter. |

## Limitations and future work

- Knowledge is limited to what's in `data/`; there is no live web access.
- Chunking is word-count based; sentence-aware splitting would give cleaner boundaries.
- Chat history resets on page reload.
- A thumbs up/down control per answer would help identify weak retrieval cases.

## How RAG works, briefly

Instead of asking a language model to answer from memory, RAG first retrieves the most
relevant passages of real, trusted text and then asks the model to answer using only
those passages. This reduces hallucination and means the system's knowledge can be
updated simply by editing documents — no retraining required.
