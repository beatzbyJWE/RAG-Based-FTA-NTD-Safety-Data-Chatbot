"""
ingest.py
---------
Fetches FTA Major Safety Events data from the DOT open data portal (Socrata API)
and converts each record into a text chunk suitable for embedding.

Dataset: https://data.transportation.gov/Public-Transit/Major-Safety-Events/9ivb-8ae9
"""

import os
import json
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_FILE = DATA_DIR / "fta_safety_events.json"

# Socrata dataset identifier on data.transportation.gov
DATASET_ID = "9ivb-8ae9"
SOCRATA_BASE = "https://data.transportation.gov/resource"


def fetch_fta_data(limit: int = 50000, use_cache: bool = True) -> pd.DataFrame:
    """
    Download FTA Major Safety Events from the DOT open data portal.
    Caches locally so repeated runs don't re-download.
    """
    DATA_DIR.mkdir(exist_ok=True)

    if use_cache and CACHE_FILE.exists():
        print(f"Loading cached data from {CACHE_FILE}")
        with open(CACHE_FILE) as f:
            records = json.load(f)
        return pd.DataFrame(records)

    print(f"Fetching up to {limit} records from data.transportation.gov...")
    app_token = os.getenv("SOCRATA_APP_TOKEN", "")
    headers = {"X-App-Token": app_token} if app_token else {}

    url = f"{SOCRATA_BASE}/{DATASET_ID}.json"
    params = {"$limit": limit, "$order": "incident_date DESC"}

    response = requests.get(url, headers=headers, params=params, timeout=60)
    response.raise_for_status()
    records = response.json()

    with open(CACHE_FILE, "w") as f:
        json.dump(records, f)

    print(f"Fetched {len(records)} records.")
    return pd.DataFrame(records)


def record_to_text(row: pd.Series) -> str:
    """
    Convert a single FTA safety event record into a plain-English text chunk.
    This is what gets embedded and retrieved.
    """
    parts = []

    # Date
    date = row.get("incident_date", "Unknown date")
    if date and len(str(date)) >= 10:
        date = str(date)[:10]
    parts.append(f"Date: {date}")

    # Agency and location
    agency = row.get("ntd_id_reporter_name") or row.get("reporter_name") or "Unknown agency"
    state = row.get("state", "")
    city = row.get("city", "")
    location_parts = [p for p in [city, state] if p]
    location = ", ".join(location_parts) if location_parts else "Unknown location"
    parts.append(f"Agency: {agency} ({location})")

    # Transit mode
    mode = row.get("mode_name") or row.get("mode", "Unknown mode")
    parts.append(f"Mode: {mode}")

    # Event type
    event_type = row.get("event_type_desc") or row.get("event_type", "Unknown event type")
    parts.append(f"Event type: {event_type}")

    # Casualties
    fatalities = row.get("total_fatalities") or row.get("fatalities", "0")
    injuries = row.get("total_injuries") or row.get("injuries", "0")
    parts.append(f"Fatalities: {fatalities} | Injuries: {injuries}")

    # Contributing factors / cause
    cause = row.get("primary_cause") or row.get("cause", "")
    if cause:
        parts.append(f"Primary cause: {cause}")

    # Narrative / description
    narrative = row.get("narrative") or row.get("description") or row.get("event_narrative", "")
    if narrative:
        # Trim very long narratives
        if len(narrative) > 600:
            narrative = narrative[:600] + "..."
        parts.append(f"Description: {narrative}")

    return "\n".join(parts)


def prepare_chunks(df: pd.DataFrame) -> list[dict]:
    """
    Convert dataframe rows into a list of dicts with 'id', 'text', and 'metadata'.
    These are fed into the vector store.
    """
    chunks = []
    for idx, row in df.iterrows():
        text = record_to_text(row)
        if len(text.strip()) < 30:
            continue  # skip empty/malformed records

        metadata = {
            "date": str(row.get("incident_date", ""))[:10],
            "agency": str(row.get("ntd_id_reporter_name") or row.get("reporter_name") or ""),
            "state": str(row.get("state", "")),
            "mode": str(row.get("mode_name") or row.get("mode", "")),
            "event_type": str(row.get("event_type_desc") or row.get("event_type", "")),
            "fatalities": str(row.get("total_fatalities") or row.get("fatalities", "0")),
            "injuries": str(row.get("total_injuries") or row.get("injuries", "0")),
        }

        chunks.append({
            "id": f"event_{idx}",
            "text": text,
            "metadata": metadata,
        })

    print(f"Prepared {len(chunks)} chunks from {len(df)} records.")
    return chunks


if __name__ == "__main__":
    df = fetch_fta_data()
    chunks = prepare_chunks(df)
    print(f"\nSample chunk:\n{chunks[0]['text']}")
