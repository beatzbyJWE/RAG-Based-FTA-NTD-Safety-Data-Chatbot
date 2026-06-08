"""
embeddings.py
-------------
Semantic vector store using Voyage AI embeddings + numpy.

Voyage AI (Anthropic's recommended embedding partner) provides semantic embeddings
via a pure-Python API client — no torch, no onnxruntime, no platform-specific binaries.
Vectors are stored as a compressed numpy file alongside a JSON metadata index.

Model: voyage-3-lite — fast, cost-efficient, excellent for retrieval tasks.
Free tier: 50M tokens/month at dash.voyageai.com
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import voyageai

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"
VECTORS_FILE = DATA_DIR / "vectors.npz"
METADATA_FILE = DATA_DIR / "metadata.json"
CHECKPOINT_FILE = DATA_DIR / "embed_checkpoint.json"

EMBED_MODEL = "voyage-3-lite"
BATCH_SIZE = 64   # Smaller batches = less data lost on timeout
MAX_RETRIES = 5


def _get_secret(key: str) -> str | None:
    """Read a secret from env vars (local) or st.secrets (Streamlit Cloud)."""
    value = os.getenv(key)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(key)
    except Exception:
        return None


def _client() -> voyageai.Client:
    api_key = _get_secret("VOYAGE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "VOYAGE_API_KEY not set. Add it to your .env file (local) "
            "or Streamlit secrets (cloud). Get a free key at https://dash.voyageai.com"
        )
    return voyageai.Client(api_key=api_key)


def _embed_batch_with_retry(client: voyageai.Client, texts: list[str]) -> list:
    """Embed a single batch with exponential backoff on timeout/rate-limit errors."""
    for attempt in range(MAX_RETRIES):
        try:
            result = client.embed(texts, model=EMBED_MODEL, input_type="document")
            return result.embeddings
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt  # 1s, 2s, 4s, 8s, 16s
            print(f"  Batch failed ({e}), retrying in {wait}s...")
            time.sleep(wait)


def index_chunks(chunks: list[dict]) -> None:
    """
    Embed all chunks via Voyage AI and save to disk as compressed numpy arrays.
    Resumes from a checkpoint if a previous run was interrupted.
    """
    DATA_DIR.mkdir(exist_ok=True)
    client = _client()

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Load checkpoint if one exists
    start_batch = 0
    all_embeddings = []
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            checkpoint = json.load(f)
        if checkpoint.get("total") == len(chunks):
            all_embeddings = checkpoint["embeddings"]
            start_batch = len(all_embeddings) // BATCH_SIZE
            print(f"Resuming from checkpoint — {len(all_embeddings)} embeddings already done.")
        else:
            print("Chunk count changed, ignoring old checkpoint.")

    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Embedding {len(texts)} chunks via Voyage AI ({EMBED_MODEL})...")

    for batch_num in range(start_batch, total_batches):
        i = batch_num * BATCH_SIZE
        batch = texts[i : i + BATCH_SIZE]
        embeddings = _embed_batch_with_retry(client, batch)
        all_embeddings.extend(embeddings)

        # Save checkpoint every 10 batches (~640 records)
        if (batch_num + 1) % 10 == 0 or batch_num == total_batches - 1:
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump({"total": len(chunks), "embeddings": all_embeddings}, f)
            print(f"  [{batch_num + 1}/{total_batches}] {len(all_embeddings)}/{len(texts)} embedded")

    vectors = np.array(all_embeddings, dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-10)

    np.savez_compressed(VECTORS_FILE, vectors=vectors)
    with open(METADATA_FILE, "w") as f:
        json.dump({"ids": ids, "texts": texts, "metadatas": metadatas}, f)

    # Clean up checkpoint on success
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

    print(f"Saved {len(chunks)} vectors → {VECTORS_FILE}")


def _load_index():
    if not VECTORS_FILE.exists() or not METADATA_FILE.exists():
        raise ValueError("Index not found. Run `python -m src.build_index` first.")
    data = np.load(VECTORS_FILE)
    vectors = data["vectors"]
    with open(METADATA_FILE) as f:
        meta = json.load(f)
    return vectors, meta["ids"], meta["texts"], meta["metadatas"]


def index_count() -> int:
    try:
        _, ids, _, _ = _load_index()
        return len(ids)
    except ValueError:
        return 0


def index_summary() -> dict:
    """Return a breakdown of indexed documents by source type."""
    try:
        _, _, _, metadatas = _load_index()
        events = sum(1 for m in metadatas if m.get("source") != "manual")
        manual = sum(1 for m in metadatas if m.get("source") == "manual")
        return {"total": len(metadatas), "events": events, "manual_sections": manual}
    except ValueError:
        return {"total": 0, "events": 0, "manual_sections": 0}


def retrieve(query: str, n_results: int = 8) -> list[dict]:
    """
    Embed the query via Voyage AI and return top-n chunks by cosine similarity.
    """
    client = _client()
    vectors, ids, texts, metadatas = _load_index()

    result = client.embed([query], model=EMBED_MODEL, input_type="query")
    query_vec = np.array(result.embeddings[0], dtype="float32")
    query_vec = query_vec / max(np.linalg.norm(query_vec), 1e-10)

    scores = vectors @ query_vec
    top_indices = np.argsort(scores)[::-1][:n_results]

    return [
        {
            "text": texts[i],
            "metadata": metadatas[i],
            "score": float(scores[i]),
        }
        for i in top_indices
    ]


if __name__ == "__main__":
    results = retrieve("fatal incidents on New York City subway")
    for r in results[:3]:
        print(r["text"])
        print(f"score: {r['score']:.3f}")
        print("---")
