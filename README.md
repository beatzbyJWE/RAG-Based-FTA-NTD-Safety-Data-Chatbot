# FTA Safety Insights Assistant

A RAG (Retrieval-Augmented Generation) chatbot that lets you ask plain-English questions about Federal Transit Administration Major Safety Events data — powered by Claude (Anthropic).

**Live demo:** *(deploy link goes here after Streamlit Cloud deployment)*

---

## What it does

Transit agencies, policy teams, and researchers sit on large safety datasets that are hard to query without SQL expertise. This assistant makes that data conversational:

> *"What are the most common causes of fatalities on bus routes?"*  
> *"Show me fatal incidents in New York City since 2020."*  
> *"Which agencies have the highest injury rates?"*

The assistant finds the most relevant records, passes them to Claude, and returns a cited, plain-English answer grounded in the actual data — not a hallucination.

---

## Architecture

```
User question
      │
      ▼
 Embed question          ← sentence-transformers (local, no API key)
      │
      ▼
 Vector similarity       ← ChromaDB (local persistent store)
 search over ~50k FTA
 safety event records
      │
      ▼
 Top X matching          ← ranked by cosine similarity
 event records
      │
      ▼
 Claude (Anthropic API)  ← grounded answer generation with citations
      │
      ▼
 Answer + source list
```

This pattern — **RAG (Retrieval-Augmented Generation)** — is the dominant architecture for organizations that want to deploy AI on their own data without fine-tuning a model or exposing sensitive information to a general-purpose chatbot.

### Why RAG matters for organizations

Most enterprise AI use cases boil down to: *"I have data. I want to ask questions about it."* RAG solves this by:

- **Keeping your data local** — records stay in your own vector store, not uploaded to a model
- **Grounding answers in real records** — the LLM can only answer from what's retrieved, reducing hallucination
- **Being auditable** — every answer shows which source records were used
- **Scaling without retraining** — add new data by re-indexing, no model changes needed

---

## Data source

FTA Major Safety Events — published by the Federal Transit Administration on the DOT Open Data Portal:  
https://data.transportation.gov/Public-Transit/Major-Safety-Events/9ivb-8ae9

This is data I helped validate and publish during my time managing the National Transit Database program at Boyd Caton Group. The dataset covers transit safety incidents (collisions, derailments, fires, personal casualties) across U.S. transit agencies from 2014–present.

---

## Setup

### Prerequisites
- Python 3.10+
- An Anthropic API key ([get one here](https://console.anthropic.com))

### Install

```bash
git clone https://github.com/beatzbyJWE/fta-safety-assistant.git
cd fta-safety-assistant
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Build the index (one time)

```bash
python -m src.build_index
```

This downloads ~50k safety events from the DOT API, converts each record to a text chunk, embeds them using a local sentence-transformers model (`all-MiniLM-L6-v2`), and stores them in a local ChromaDB database. Takes about 5–10 minutes on first run.

### Run the app

```bash
streamlit run app.py
```

Open the local host in your browser.

---

## Project structure

```
fta-safety-assistant/
├── app.py                  # Streamlit UI
├── requirements.txt
├── .env.example
├── src/
│   ├── ingest.py           # Fetch + parse FTA data from Socrata API
│   ├── embeddings.py       # Embed chunks + ChromaDB vector store
│   ├── assistant.py        # RAG pipeline: retrieve → prompt → Claude
│   └── build_index.py      # One-time index build script
└── data/                   # Created at runtime (gitignored)
    ├── fta_safety_events.json
    └── chroma_db/
```

---

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Under **Advanced settings → Secrets**, add:
   ```
   ANTHROPIC_API_KEY = "your_key_here"
   ```
4. Note: Streamlit Cloud has an ephemeral filesystem. You'll need to build the index on first startup or use a persistent vector store (e.g., Pinecone, Qdrant Cloud) for production deployments.

---

## Consulting context

This project is a proof of concept for a pattern I apply when advising organizations on AI adoption:

1. **What data do you already have?** Most organizations have structured datasets they've been collecting for years.
2. **What questions do people actually ask about it?** Usually in plain English, not SQL.
3. **Build the bridge.** A RAG pipeline connects your existing data to a conversational interface — without replacing your database, training a custom model, or sending sensitive data to a general-purpose chatbot.

The transit safety domain is one I know well. The same architecture applies to HR incident logs, customer support tickets, regulatory filings, clinical notes, or any other corpus of semi-structured records where people need to ask questions faster than a data team can write queries.


## Author

**Joseph Eldredge** | PMP, CPMAI  
[LinkedIn](https://www.linkedin.com/in/joseph-eldredge-75ab6349/) · [GitHub](https://github.com/beatzbyJWE)
