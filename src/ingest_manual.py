"""
ingest_manual.py
----------------
Reads the 2026 NTD Safety & Security Policy Manual from a local PDF file,
splits it into section-level chunks, and returns them in the same format
as ingest.py so they can be merged into the same vector index.

Place the PDF at: data/ntd_manual.pdf
Download from: https://www.transit.dot.gov/sites/fta.dot.gov/files/2026-05/2026%20Safety%20%26%20Security%20Manual%20V1.pdf

Having the manual in the index means Claude can retrieve:
  - definitions of event types alongside actual event records
  - threshold rules (what counts as a fatality, injury, property damage)
  - explicit scope limits (what IS and IS NOT collected in this dataset)
"""

import re
from pathlib import Path
from pypdf import PdfReader

DATA_DIR = Path(__file__).parent.parent / "data"
MANUAL_PDF = DATA_DIR / "ntd_manual.pdf"
MANUAL_CACHE = DATA_DIR / "ntd_manual.txt"

# Section header patterns found in the manual's table of contents.
# Used to split the raw text into meaningful chunks.
SECTION_PATTERNS = [
    r"^Introduction[:.]",
    r"^History\b",
    r"^Safety and Security Reporting Requirements",
    r"^Frequently Asked Questions",
    r"^Who Reports",
    r"^Reporting Overview",
    r"^When to Report",
    r"^Where to Report",
    r"^Failure to Report",
    r"^S&S-\d+:",
    r"^Overview\b",
    r"^Major Event Threshold",
    r"^Fatality\b",
    r"^Injury\b",
    r"^Property Damage\b",
    r"^Evacuation\b",
    r"^Derailment",
    r"^Collisions?\b",
    r"^Runaway Train",
    r"^Event Types\b",
    r"^Fires?\b",
    r"^Hazardous Material",
    r"^Acts of God",
    r"^Security Events?",
    r"^Other Safety Events?",
    r"^Non-Major",
    r"^Appendix",
]

_SECTION_RE = re.compile(
    "|".join(f"({p})" for p in SECTION_PATTERNS),
    re.MULTILINE | re.IGNORECASE,
)


def fetch_manual_text(use_cache: bool = True) -> str:
    """
    Extract text from the local NTD manual PDF and return it as a string.
    Caches the extracted text so subsequent runs don't re-parse the PDF.
    """
    DATA_DIR.mkdir(exist_ok=True)

    if use_cache and MANUAL_CACHE.exists():
        return MANUAL_CACHE.read_text(encoding="utf-8", errors="replace")

    if not MANUAL_PDF.exists():
        raise FileNotFoundError(
            f"NTD manual PDF not found at {MANUAL_PDF}.\n"
            "Download it from:\n"
            "https://www.transit.dot.gov/sites/fta.dot.gov/files/2026-05/"
            "2026%20Safety%20%26%20Security%20Manual%20V1.pdf\n"
            "and save it as data/ntd_manual.pdf"
        )

    print(f"Extracting text from {MANUAL_PDF}...")
    reader = PdfReader(str(MANUAL_PDF))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages.append(page_text)

    text = "\n\n".join(pages)

    # Basic cleanup
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    MANUAL_CACHE.write_text(text, encoding="utf-8")
    print(f"Extracted {len(text):,} chars from {len(pages)} pages → cached to {MANUAL_CACHE}")
    return text


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """
    Split the manual text into (section_title, section_body) pairs.
    Sections are delimited by known header patterns from the TOC.
    """
    lines = text.splitlines()
    sections = []
    current_title = "Introduction"
    current_lines = []

    for line in lines:
        stripped = line.strip()
        if _SECTION_RE.match(stripped) and len(stripped) < 120:
            # Save previous section if it has meaningful content
            body = "\n".join(current_lines).strip()
            if len(body) > 100:
                sections.append((current_title, body))
            current_title = stripped
            current_lines = []
        else:
            current_lines.append(line)

    # Don't forget the last section
    body = "\n".join(current_lines).strip()
    if len(body) > 100:
        sections.append((current_title, body))

    return sections


def _chunk_section(title: str, body: str, max_chars: int = 1200) -> list[str]:
    """
    If a section body is longer than max_chars, split into overlapping chunks
    on paragraph boundaries so no chunk loses context at its edges.
    """
    if len(body) <= max_chars:
        return [f"[NTD Manual — {title}]\n\n{body}"]

    paragraphs = re.split(r"\n{2,}", body)
    chunks = []
    current = f"[NTD Manual — {title}]\n\n"

    for para in paragraphs:
        if len(current) + len(para) > max_chars and len(current) > len(f"[NTD Manual — {title}]\n\n"):
            chunks.append(current.strip())
            current = f"[NTD Manual — {title} (cont.)]\n\n{para}\n\n"
        else:
            current += para + "\n\n"

    if current.strip():
        chunks.append(current.strip())

    return chunks


def prepare_manual_chunks(use_cache: bool = True) -> list[dict]:
    """
    Fetch the manual, split into sections, and return a list of chunk dicts
    compatible with embeddings.index_chunks().
    """
    text = fetch_manual_text(use_cache=use_cache)
    sections = _split_into_sections(text)
    print(f"Split manual into {len(sections)} sections.")

    chunks = []
    chunk_idx = 0
    for title, body in sections:
        for chunk_text in _chunk_section(title, body):
            chunks.append({
                "id": f"manual_{chunk_idx}",
                "text": chunk_text,
                "metadata": {
                    "source": "manual",
                    "section": title,
                    "date": "2026-01",
                    "agency": "FTA / NTD",
                    "mode": "",
                    "event_type": "",
                    "fatalities": "",
                    "injuries": "",
                },
            })
            chunk_idx += 1

    print(f"Prepared {len(chunks)} manual chunks.")
    return chunks


if __name__ == "__main__":
    chunks = prepare_manual_chunks()
    for c in chunks[:3]:
        print(c["text"][:300])
        print("---")
