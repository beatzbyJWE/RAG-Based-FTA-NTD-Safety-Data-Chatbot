"""
build_index.py
--------------
One-time setup script: fetches FTA data, processes it into chunks,
and indexes everything into the local ChromaDB vector store.

Run once before launching the Streamlit app:
    python -m src.build_index
"""

from src.ingest import fetch_fta_data, prepare_chunks
from src.embeddings import index_chunks, index_count


def main():
    existing_count = index_count()

    if existing_count > 0:
        print(f"Index already contains {existing_count} documents.")
        answer = input("Re-index from scratch? This will overwrite existing data. (y/N): ")
        if answer.strip().lower() != "y":
            print("Keeping existing index.")
            return

    print("Step 1/2: Fetching FTA safety events...")
    df = fetch_fta_data(limit=50000, use_cache=True)

    print(f"\nStep 2/2: Embedding and indexing {len(df)} records...")
    chunks = prepare_chunks(df)
    index_chunks(chunks)

    print("\nIndex ready. Run `streamlit run app.py` to start the assistant.")


if __name__ == "__main__":
    main()
