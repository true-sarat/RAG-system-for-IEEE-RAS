"""
config.py
---------
Single source of truth for paths and settings, so ingest.py and rag_engine.py
can never drift apart (mismatched DB paths were a real bug earlier).
"""

import os
import tempfile

from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# IMPORTANT: on Streamlit Community Cloud the app folder (/mount/src/...) is
# NOT writable, so ChromaDB cannot create its SQLite file there. The OS temp
# folder is writable everywhere, so the database lives there instead.
DB_DIR = os.path.join(tempfile.gettempdir(), "ieee_ras_chroma_db")

# --- Retrieval settings ------------------------------------------------
COLLECTION_NAME = "ieee_ras_knowledge"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 4

# Cosine distance: 0 = identical, 2 = opposite. Chunks further away than this
# are treated as irrelevant, which is what triggers the honest "I don't have
# that information" answer instead of a hallucination.
RELEVANCE_DISTANCE_CUTOFF = 2.00


# --- Secrets -----------------------------------------------------------
def get_secret(name: str, default: str | None = None) -> str | None:
    """
    Read a setting from the environment (.env locally) and fall back to
    Streamlit Cloud's secrets store when running deployed.
    """
    value = os.getenv(name)
    if value:
        return value.strip()

    try:  # only available when running inside Streamlit
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass

    return default


# Model name is configurable so a future Gemini rename only needs a .env edit
# (no code change). Alternatives if you hit free-tier quota limits:
#   gemini-3.5-flash-lite  <- cheapest / highest throughput, best for free tier
#   gemini-3.6-flash       <- stable workhorse
#   gemini-3.7-flash / gemini-3.8-flash  <- newer, heavier
GEMINI_MODEL = get_secret("GEMINI_MODEL", "gemini-3.6-flash")
