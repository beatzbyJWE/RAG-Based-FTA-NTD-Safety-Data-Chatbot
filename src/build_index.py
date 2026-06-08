"""
build_index.py
--------------
One-time setup script: fetches FTA safety event records AND the NTD Policy Manual,
merges them into a single vector index so Claude can retrieve both event records
and governing definitions in the same query.

Run once before launching the Streamlit app:
    python -m src.build_index
"""

from src.ingest import fetch_fta_data, prepare_chunks
from src.ingest_manual import prepare_manual_chunks
from src.embeddings import index_chunks, index_count


def main(record_limit: int = 50_000):
    existing = index_count()
    if existing > 0:
        print(f"Index already contains {existing} documents.")
        answer = input("Re-index from scratch? (y/N): ")
        if answer.strip().lower() != "y":
            print("Keeping existing index.")
            return

    # ── Step 1: Safety event records ─────────────────────────────────────────
    print("Step 1/3: Fetching FTA safety event records...")
    df = fetch_fta_data(limit=record_limit, use_cache=True)
    event_chunks = prepare_chunks(df)
    print(f"  → {len(event_chunks)} event chunks ready.")

    # ── Step 2: NTD Policy Manual ─────────────────────────────────────────────
    print("\nStep 2/3: Fetching NTD Safety & Security Policy Manual...")
    manual_chunks = prepare_manual_chunks(use_cache=True)
    print(f"  → {len(manual_chunks)} manual chunks ready.")

    # ── Step 3: Embed and index everything together ───────────────────────────
    all_chunks = event_chunks + manual_chunks
    print(f"\nStep 3/3: Embedding {len(all_chunks)} total chunks "
          f"({len(event_chunks)} events + {len(manual_chunks)} manual sections)...")
    index_chunks(all_chunks)

    print(f"\nIndex ready — {index_count():,} documents total.")
    print("Run `streamlit run app.py` to start the assistant.")


if __name__ == "__main__":
    main()
