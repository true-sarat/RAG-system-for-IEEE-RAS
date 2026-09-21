import os
import tempfile

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Streamlit Cloud's app folder is read-only; the OS temp folder is always writable.
DB_DIR = os.path.join(tempfile.gettempdir(), "ieee_ras_chroma_db")

COLLECTION_NAME = "ieee_ras_knowledge"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3
RELEVANCE_DISTANCE_CUTOFF = 0.75


def get_secret(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value:
        return value.strip()

    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass

    return default


GEMINI_MODEL = get_secret("GEMINI_MODEL", "gemini-3.6-flash")

_embed_fn = None
_client = None


def get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        from chromadb.utils import embedding_functions

        _embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL_NAME
        )
    return _embed_fn


def get_client():
    global _client
    if _client is None:
        import chromadb

        os.makedirs(DB_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=DB_DIR)
    return _client"""
config.py
"""

import os
import tempfile

from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

DB_DIR = os.path.join(tempfile.gettempdir(), "ieee_ras_chroma_db")

# --- Retrieval settings ------------------------------------------------
COLLECTION_NAME = "ieee_ras_knowledge"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 4

RELEVANCE_DISTANCE_CUTOFF = 1.80


# --- Secrets -----------------------------------------------------------
def get_secret(name: str, default: str | None = None) -> str | None:
    """
    Read a setting from the environment (.env locally) and fall back to
    Streamlit Cloud's secrets store when running deployed.
    """
    value = os.getenv(name)
    if value:
        return value.strip()

    try: 
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass

    return default



GEMINI_MODEL = get_secret("GEMINI_MODEL", "gemini-3.6-flash")
