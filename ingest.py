"""
ingest.py
---------
Reads every .txt / .md file in data/, splits it into overlapping chunks,
embeds them locally with sentence-transformers, and stores them in a
persistent ChromaDB collection.

Run manually with:  python ingest.py
It is also imported and called by app.py on startup 

Safe to run repeatedly: it upserts with deterministic IDs, so re-running
updates existing chunks instead of duplicating them.
"""

import hashlib
import os

import chromadb
from chromadb.utils import embedding_functions

from config import COLLECTION_NAME, DATA_DIR, DB_DIR, EMBED_MODEL_NAME

CHUNK_WORDS = 220      # ~ a few sentences; small chunks retrieve more precisely
CHUNK_OVERLAP = 40     # overlap keeps sentences from being cut mid-thought


def load_documents(data_dir: str = DATA_DIR):
    """Return a list of (filename, text) for every text file in data/."""
    docs = []
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data folder '{data_dir}' not found. Create it and add .txt files."
        )

    for filename in sorted(os.listdir(data_dir)):
        if not filename.lower().endswith((".txt", ".md")):
            continue
        path = os.path.join(data_dir, filename)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        if text:
            docs.append((filename, text))

    if not docs:
        raise FileNotFoundError(
            f"No .txt or .md files with content found in '{data_dir}'."
        )
    return docs


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping word-count chunks."""
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(1, chunk_words - overlap)
    for start in range(0, len(words), step):
        piece = words[start:start + chunk_words]
        if piece:
            chunks.append(" ".join(piece))
        if start + chunk_words >= len(words):
            break
    return chunks


def build_chunks():
    """Return ids, texts, metadatas ready for ChromaDB."""
    ids, texts, metadatas = [], [], []

    for filename, text in load_documents():
        for i, chunk in enumerate(chunk_text(text)):
            # Deterministic ID: same content + position -> same ID, so an
            # upsert overwrites instead of creating duplicates.
            digest = hashlib.md5(f"{filename}:{i}:{chunk}".encode("utf-8")).hexdigest()
            ids.append(digest)
            texts.append(chunk)
            metadatas.append({"source": filename, "chunk_index": i})

    return ids, texts, metadatas


def main():
    """Build (or refresh) the vector database. Returns the number of chunks."""
    print(f"Loading and chunking documents from {DATA_DIR}/ ...")
    ids, texts, metadatas = build_chunks()
    print(f"  -> {len(texts)} chunks")

    print(f"Loading local embedding model ({EMBED_MODEL_NAME}) ...")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL_NAME
    )

    os.makedirs(DB_DIR, exist_ok=True)
    print(f"Writing to persistent ChromaDB at '{DB_DIR}' ...")
    client = chromadb.PersistentClient(path=DB_DIR)

    # get_or_create (never delete) — deleting mid-flight is what caused the
    # "Collection does not exist" / InvalidCollectionException races.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Upsert in batches so large knowledge bases don't blow up memory.
    batch = 64
    for start in range(0, len(ids), batch):
        collection.upsert(
            ids=ids[start:start + batch],
            documents=texts[start:start + batch],
            metadatas=metadatas[start:start + batch],
        )

    print(f"Done. Collection '{COLLECTION_NAME}' now holds {collection.count()} chunks.")
    return collection.count()


if __name__ == "__main__":
    main()
