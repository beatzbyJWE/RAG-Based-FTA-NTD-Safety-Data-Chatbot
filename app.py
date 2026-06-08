"""
app.py
------
Streamlit UI for the FTA Safety Insights Assistant.

Run with:
    streamlit run app.py
"""

import streamlit as st
from src.assistant import ask
from src.embeddings import index_count, index_chunks
from src.ingest import fetch_fta_data, prepare_chunks

CLOUD_RECORD_LIMIT = 10_000  # Keeps cold-start build under ~2 minutes

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FTA Safety Insights Assistant",
    page_icon="🚇",
    layout="centered",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚇 FTA Safety Insights Assistant")
st.caption(
    "Ask plain-English questions about Federal Transit Administration Major Safety Events data. "
    "Answers are grounded in the public dataset published on [data.transportation.gov]"
    "(https://data.transportation.gov/Public-Transit/Major-Safety-Events/9ivb-8ae9)."
)

# ── Index: auto-build if missing ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_or_build_index() -> int:
    count = index_count()
    if count > 0:
        return count

    # Index not found — build it now (first cold start on Streamlit Cloud)
    placeholder = st.empty()
    placeholder.info("⏳ First launch: building search index from FTA safety data. This takes about 2 minutes — please wait.")

    with st.spinner("Fetching safety events from data.transportation.gov..."):
        df = fetch_fta_data(limit=CLOUD_RECORD_LIMIT, use_cache=False)

    with st.spinner(f"Embedding {len(df):,} records via Voyage AI..."):
        chunks = prepare_chunks(df)
        index_chunks(chunks)

    placeholder.empty()
    return index_count()

doc_count = load_or_build_index()
st.success(f"✅ Index loaded — {doc_count:,} safety events available")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Retrieval settings")
    n_results = st.slider(
        "Source events to retrieve",
        min_value=4,
        max_value=20,
        value=8,
        help="How many safety events to pull from the index per query. "
             "Higher values give Claude more context but may slow responses.",
    )
    st.divider()
    st.markdown(
        "**Tips for good questions**\n\n"
        "This assistant retrieves *specific events* then summarizes them. "
        "It works best for:\n"
        "- Describing what happened in a place or agency\n"
        "- Finding common causes or patterns in an event type\n"
        "- Comparing incidents across transit modes\n\n"
        "For exact counts or totals, use the "
        "[full dataset](https://data.transportation.gov/Public-Transit/Major-Safety-Events/9ivb-8ae9) directly."
    )

# ── Example questions ─────────────────────────────────────────────────────────
st.markdown("#### Try asking:")
example_questions = [
    "What caused recent fatal incidents on heavy rail?",
    "Describe bus collision incidents in Texas.",
    "What types of events does the MTA report most often?",
    "What do derailment incidents typically involve?",
    "What safety issues appear in light rail incidents?",
    "Describe pedestrian fatality incidents on commuter rail.",
]

cols = st.columns(2)
for i, q in enumerate(example_questions):
    if cols[i % 2].button(q, use_container_width=True):
        st.session_state["prefill"] = q

# ── Chat interface ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} source events used"):
                for s in msg["sources"]:
                    st.markdown(
                        f"- **{s.get('date', '?')}** | {s.get('agency', 'Unknown agency')} | "
                        f"{s.get('mode', '?')} | {s.get('event_type', '?')} | "
                        f"Fatalities: {s.get('fatalities', '0')}"
                    )

# Handle prefilled question from example buttons
prefill = st.session_state.pop("prefill", None)

# Chat input
user_input = st.chat_input("Ask a question about FTA transit safety data...")
question = prefill or user_input

if question:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching safety events and generating answer..."):
            try:
                result = ask(question, n_results=n_results)
                answer = result["answer"]
                sources = result["sources"]
            except EnvironmentError as e:
                answer = f"⚠️ **Configuration error:** {e}"
                sources = []
            except ValueError as e:
                answer = f"⚠️ **Index error:** {e}"
                sources = []
            except Exception as e:
                answer = f"⚠️ **Unexpected error:** {e}"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander(f"📎 {len(sources)} source events used"):
                for s in sources:
                    st.markdown(
                        f"- **{s.get('date', '?')}** | {s.get('agency', 'Unknown agency')} | "
                        f"{s.get('mode', '?')} | {s.get('event_type', '?')} | "
                        f"Fatalities: {s.get('fatalities', '0')}"
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built by [Joseph Eldredge](https://eldredgemgmtconsulting.com) · "
    "Data: FTA National Transit Database via DOT Open Data Portal · "
    "AI: Claude (Anthropic) · "
    "[Source code](https://github.com/beatzbyJWE/fta-safety-assistant)"
)
