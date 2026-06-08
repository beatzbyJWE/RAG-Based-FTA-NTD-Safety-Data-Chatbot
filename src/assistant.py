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

SYSTEM_PROMPT = """You are an FTA Transit Safety Analyst assistant. You help transit agency staff,
policy makers, and researchers understand patterns in Federal Transit Administration (FTA) Major
Safety Events data.

You answer questions using only the safety event records provided to you. Be specific and cite
the events you reference (agency, date, event type). When you see patterns across multiple events,
summarize them clearly. If the provided records don't contain enough information to answer the
question, say so honestly — do not fabricate data.

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
