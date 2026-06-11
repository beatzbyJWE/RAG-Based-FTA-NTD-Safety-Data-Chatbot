"""
assistant.py
------------
RAG pipeline: retrieves relevant FTA safety events and passes them to Claude
to generate a grounded, cited answer.
"""

import os
import anthropic
from dotenv import load_dotenv
from src.embeddings import retrieve

load_dotenv()

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are a transit safety data analyst assistant. You help transit agency staff,
policy makers, and researchers understand patterns in the public Major Safety Events dataset
published by the Federal Transit Administration (FTA). You are an independent tool and not an
FTA or government product.

You have access to two types of context:
1. SAFETY EVENT RECORDS — actual incidents reported to the NTD by transit agencies.
2. NTD POLICY MANUAL SECTIONS — the official 2026 NTD Safety & Security Policy Manual,
   which defines event types, reporting thresholds, and what is and is not collected.

Rules for using context:
- When a manual section is provided, treat it as the authoritative definition. It overrides
  any assumptions about what the dataset does or does not contain.
- When answering "what is" or "what counts as" questions, prioritize manual definitions.
- When answering "what happened" questions, prioritize event records.
- If the manual explicitly says a type of event is NOT collected (e.g., medical emergencies
  unrelated to a transit incident), say so clearly and cite the manual.
- Never speculate about what categories might include. If the manual defines the scope, use it.
- Cite your sources: for events, give agency + date + event type; for manual, give the section name.

Respond in clear, plain English suitable for both technical and non-technical audiences. Keep
answers concise but complete. Use bullet points when listing multiple events or findings."""


def build_context(hits: list[dict]) -> str:
    """Format retrieved chunks into a context block for the prompt."""
    if not hits:
        return "No relevant safety events found in the database."

    lines = ["RELEVANT FTA SAFETY EVENTS (retrieved from database):\n"]
    for i, hit in enumerate(hits, 1):
        lines.append(f"[Event {i}]")
        lines.append(hit["text"])
        lines.append("")

    return "\n".join(lines)


def ask(question: str, n_results: int = 8) -> dict:
    """
    Full RAG pipeline:
    1. Retrieve relevant events from ChromaDB
    2. Build a prompt with the retrieved context
    3. Send to Claude and return the answer + sources

    Returns dict with 'answer' (str) and 'sources' (list of metadata dicts).
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("ANTHROPIC_API_KEY")
        except Exception:
            pass
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key."
        )

    # Step 1: Retrieve
    hits = retrieve(question, n_results=n_results)

    # Step 2: Build context
    context = build_context(hits)

    # Step 3: Call Claude
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"{context}\n\nQuestion: {question}",
            }
        ],
    )

    answer = message.content[0].text
    sources = [h["metadata"] for h in hits]

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    result = ask("What are the most common causes of fatalities on bus routes?")
    print(result["answer"])
    print("\nSources used:")
    for s in result["sources"]:
        print(f"  - {s.get('date', '?')} | {s.get('agency', '?')} | {s.get('event_type', '?')}")
