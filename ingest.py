import hashlib
import os

from config import COLLECTION_NAME, DATA_DIR, DB_DIR, EMBED_MODEL_NAME, get_client, get_embed_fn

CHUNK_WORDS = 220
CHUNK_OVERLAP = 40


def load_documents(data_dir: str = DATA_DIR):
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
    ids, texts, metadatas = [], [], []

    for filename, text in load_documents():
        for i, chunk in enumerate(chunk_text(text)):
            digest = hashlib.md5(f"{filename}:{i}:{chunk}".encode("utf-8")).hexdigest()
            ids.append(digest)
            texts.append(chunk)
            metadatas.append({"source": filename, "chunk_index": i})

    return ids, texts, metadatas


def main():
    print(f"Loading and chunking documents from {DATA_DIR}/ ...")
    ids, texts, metadatas = build_chunks()
    print(f"  -> {len(texts)} chunks")

    print(f"Loading local embedding model ({EMBED_MODEL_NAME}) ...")
    embed_fn = get_embed_fn()

    print(f"Writing to persistent ChromaDB at '{DB_DIR}' ...")
    client = get_client()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    batch = 64
    for start in range(0, len(ids), batch):
        collection.upsert(
            ids=ids[start:start + batch],
            documents=texts[start:start + batch],
            metadatas=metadatas[start:start + batch],
        )

    total = collection.count()
    print(f"Done. Collection '{COLLECTION_NAME}' now holds {total} chunks.")
    return total


if __name__ == "__main__":
    main()
