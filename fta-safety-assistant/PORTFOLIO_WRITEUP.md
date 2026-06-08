# Portfolio Write-Up: FTA Safety Insights Assistant
## For beatzbyjwe.github.io — add under Professional Work

---

### FTA Safety Insights Assistant — RAG Chatbot on Federal Transit Data

#### [View on GitHub](https://github.com/beatzbyJWE/fta-safety-assistant) | [Live Demo](#) *(add Streamlit Cloud link)*

This project builds a conversational AI assistant on top of the FTA Major Safety Events
dataset — the same public transit safety data my team helped validate and publish to
DOT's open data portal during my time at Boyd Caton Group.

The goal was practical: transit agencies, policy teams, and researchers need to ask
questions about this data without writing SQL. A safety officer should be able to type
*"What do derailment incidents on light rail typically involve?"* and get a grounded,
cited answer — not a hallucination.

---

#### How it works: the RAG pipeline

RAG — Retrieval-Augmented Generation — is the dominant architecture for deploying AI on
organizational data. Rather than fine-tuning a model on your data (expensive, slow,
requires ML expertise) or feeding everything to a general chatbot (context limits,
privacy concerns), RAG does three things:

**1. Index your data semantically**  
Each of ~50,000 FTA safety event records is converted to a text representation and
embedded as a vector using [Voyage AI](https://www.voyageai.com/) (Anthropic's
recommended embedding partner). Semantic embeddings capture *meaning*, not just
keywords — so "person struck by train" and "pedestrian fatality" are understood as
related concepts.

**2. Retrieve the most relevant records**  
When a user asks a question, the question is also embedded and compared against all
50,000 record vectors using cosine similarity. The top matches are retrieved — fast,
without scanning every record.

**3. Generate a grounded answer**  
The retrieved records are passed as context to Claude (Anthropic), which synthesizes
a plain-English answer citing specific events. Claude can only answer from what was
retrieved — reducing hallucination and making every answer auditable.

```
Question → Voyage AI embedding → cosine similarity over 50k vectors
        → top 8–15 matching events → Claude → cited answer
```

---

#### Why this matters for organizations

Most organizations asking about AI for their internal data are really asking: *"Can we
build something like this for our data?"* The answer is usually yes, and the architecture
is the same regardless of domain:

- HR incident reports → ask "what complaints recur in the Chicago office?"
- Customer support tickets → ask "what issues cause the most escalations?"
- Regulatory filings → ask "what compliance gaps appear most often?"
- Clinical notes → ask "what symptoms precede readmission?"

This project demonstrates the pattern end-to-end. The transit safety domain is one I
know well; the stack transfers directly.

---

#### Stack

| Component | Tool | Why |
|---|---|---|
| Data source | DOT Socrata API | Public FTA dataset I know intimately |
| Embeddings | Voyage AI (`voyage-3-lite`) | Semantic search, pure Python, Anthropic ecosystem |
| Vector store | NumPy (compressed `.npz`) | No database required; transparent and portable |
| Generation | Claude (`claude-sonnet-4-6`) | Grounded answers with citations |
| UI | Streamlit | Fast to build, easy to demo to non-technical audiences |

---

#### What I learned / consulting takeaways

- **Keyword search breaks on heterogeneous data.** The FTA dataset uses inconsistent
  terminology across agencies and years. Semantic embeddings handle this gracefully;
  TF-IDF does not. This is a common surprise for organizations evaluating RAG.

- **Chunking strategy matters more than model choice.** How you represent each record
  as text — which fields you include, how you format them — has a larger impact on
  retrieval quality than switching between embedding models.

- **RAG is auditable by design.** Every answer surfaces the source records used.
  For regulated industries (healthcare, finance, federal programs), this is essential.

- **The demo is the conversation starter.** Non-technical stakeholders understand the
  value immediately when they can type a question and get a cited answer. You don't
  need to explain embeddings or vector stores — the result speaks for itself.

---

#### Related work
- [FTA Safety Analysis & Visualization](#professional-work) — geospatial maps and
  time-slider visualizations of this same dataset (earlier phase of the same project thread)
- [NLP for Data Validation](#professional-work) — prior proof-of-concept applying NLP
  to transit data quality

---
