"""
Clause-level chunker for the GFPA standards documents.

Chunks by document structure, NOT fixed-length windows:
  - A numbered requirement clause (e.g. "3.2. Class B facilities shall...")
    becomes its own chunk, citable as "GFPA-201 Section 3.2".
  - A section with no numbered clauses (e.g. a bulleted definitions section)
    becomes one chunk for the whole section, citable as "GFPA-201 Section 2".

This granularity is what lets the citation-validity eval axis (built later)
check that a cited source actually contains the claim, down to the clause
level, instead of only confirming "somewhere in this document."

Each chunk carries two text fields:
  - `text`: the exact clause/section content, unmodified — this is what
    citation-validity checking should match against, so it must stay exact.
  - `embedding_text`: `text` prefixed with document + section title context,
    used only for computing the embedding. A bare clause like "shall
    maintain coverage of no less than 55%..." often doesn't restate words
    like "surveillance" or "commercial office" that a real query would use —
    the section/document title supplies that context for retrieval without
    contaminating the exact text used for citation checking.
"""
import re
from dataclasses import asdict, dataclass
from pathlib import Path

STANDARDS_DIR = Path(__file__).resolve().parent.parent / "knowledge_base" / "standards"

DOC_TITLE_RE = re.compile(r"^#\s+([A-Z]+-\d+):\s+(.*)$")
SECTION_RE = re.compile(r"^##\s+(\d+)\.\s+(.*)$")
CLAUSE_RE = re.compile(r"^(\d+\.\d+)\.\s+(.*)$")

# Classification/definition chunks map the standard's own internal labels
# (Class A/B/C, Tier 1/2/3) to concrete real-world examples. A plain-language
# query ("what surveillance coverage does a museum need") uses the concrete
# term, not the label — but the REQUIREMENT clauses that state the actual
# numbers (e.g. GFPA-201 3.1) only ever say "Class A," never "museum." That
# leaves the definition chunk as the only bridge between query vocabulary and
# document vocabulary, and it was losing the retrieval race to the more
# narrowly-on-topic requirement clauses despite containing the right keyword
# once, in passing, inside a longer sentence about classification generally.
#
# A first attempt at fixing this by APPENDING a plain keyword list ("museum,
# art gallery, retail store, ...") to the embedding_text measurably made
# retrieval WORSE (0.655 -> 0.616 cosine similarity against a real museum
# query) — dense sentence embeddings like bge-small score on overall semantic
# gist, not term presence, so a bag of loosely-related nouns dilutes the
# vector rather than sharpening it. What actually worked, verified
# empirically: PREPENDING a short sentence that mirrors how a real question
# would be phrased ("What surveillance coverage is required for a museum...
# A museum is a Class A Public Assembly facility") lifted similarity to a
# real museum query from 0.655 to 0.75+, comfortably above the other
# candidates' ~0.69-0.79 range. Position (prepended, read first) and
# question-like phrasing both mattered more than raw keyword coverage.
#
# This dict's text is PREPENDED to embedding_text ONLY (never to `text`,
# which stays exact for citation checking).
EMBEDDING_ENRICHMENT: dict[tuple[str, str], str] = {
    ("GFPA-201", "2"): (
        "What surveillance coverage is required for a museum, retail mall, office "
        "building, or utility substation? A museum or retail mall is a Class A Public "
        "Assembly facility. An office building or office tower is a Class B Commercial "
        "Occupancy facility. A utility substation or industrial site is a Class C "
        "Critical Infrastructure facility. "
    ),
    ("GFPA-215", "2.1"): (
        "How quickly must a critical vulnerability be fixed? A critical vulnerability "
        "is classified as Tier 1 under this vulnerability severity and remediation standard. "
    ),
    ("GFPA-215", "2.2"): (
        "How quickly must an elevated or moderate-severity vulnerability be fixed? An "
        "elevated or moderate-severity vulnerability is classified as Tier 2 under this "
        "vulnerability severity and remediation standard. "
    ),
    ("GFPA-215", "2.3"): (
        "How quickly must a minor vulnerability be fixed? A minor vulnerability is "
        "classified as Tier 3 under this vulnerability severity and remediation standard. "
    ),
}


@dataclass
class Chunk:
    id: str
    document_id: str
    document_title: str
    section_id: str
    section_title: str
    citation: str
    text: str
    embedding_text: str


def _make_chunk(document_id, document_title, section_num, section_title, section_id, text) -> Chunk:
    citation = f"{document_id} Section {section_id}"
    embedding_text = f"{document_title} ({document_id}), Section {section_id} — {section_title}: {text}"
    enrichment = EMBEDDING_ENRICHMENT.get((document_id, section_id))
    if enrichment:
        embedding_text = f"{enrichment}{embedding_text}"
    return Chunk(
        id=f"{document_id}_{section_id}",
        document_id=document_id,
        document_title=document_title,
        section_id=section_id,
        section_title=section_title,
        citation=citation,
        text=text,
        embedding_text=embedding_text,
    )


def _make_clause_chunks(document_id, document_title, section_num, section_title, clause_num, text_lines) -> list[Chunk]:
    """A numbered clause is normally one citable chunk. But a clause whose
    body is itself a bulleted list of distinct topics (e.g. GFPA-240 3.1's
    checklist: fire suppression, egress, structural, access control,
    evacuation) is really several independent facts wearing one clause
    number — squashed into one chunk, a query about any single topic has to
    compete with four unrelated ones diluting the same embedding (the same
    dilution mechanism diagnosed earlier for GFPA-201 Section 2, but that
    section is a genuine single-purpose classification lookup table and is
    deliberately NOT split here — only numbered clauses, never `## `
    sections, go through this path).
    When bullet lines ("- ...") are present, split into one sub-chunk per
    bullet (id "3.1.1", "3.1.2", ...), each carrying the clause's own lead-in
    sentence so it still reads as a complete, self-contained citable fact."""
    lines = [line.strip() for line in text_lines if line.strip()]
    bullet_idx = [i for i, line in enumerate(lines) if line.startswith("- ")]
    if not bullet_idx:
        text = " ".join(lines)
        return [_make_chunk(document_id, document_title, section_num, section_title, clause_num, text)]

    lead_in = " ".join(lines[: bullet_idx[0]])
    chunks = []
    for n, i in enumerate(bullet_idx, start=1):
        bullet_text = lines[i][2:].strip()  # drop the leading "- "
        combined = f"{lead_in} {bullet_text}" if lead_in else bullet_text
        sub_id = f"{clause_num}.{n}"
        chunks.append(_make_chunk(document_id, document_title, section_num, section_title, sub_id, combined))
    return chunks


def parse_document(path: Path) -> list[Chunk]:
    lines = path.read_text(encoding="utf-8").splitlines()

    doc_id, doc_title = None, None
    chunks: list[Chunk] = []

    section_num, section_title = None, None
    section_buffer: list[str] = []
    clause_num: str | None = None
    clause_buffer: list[str] = []

    def flush():
        nonlocal clause_num, clause_buffer, section_buffer
        if clause_num is not None:
            chunks.extend(_make_clause_chunks(doc_id, doc_title, section_num, section_title, clause_num, clause_buffer))
            clause_num, clause_buffer = None, []
        elif section_buffer:
            text = " ".join(line.strip() for line in section_buffer if line.strip())
            chunks.append(_make_chunk(doc_id, doc_title, section_num, section_title, section_num, text))
            section_buffer = []

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        m_title = DOC_TITLE_RE.match(line)
        if m_title:
            doc_id, doc_title = m_title.group(1), m_title.group(2)
            continue

        m_section = SECTION_RE.match(line)
        if m_section:
            flush()
            section_num, section_title = m_section.group(1), m_section.group(2)
            continue

        m_clause = CLAUSE_RE.match(line.strip())
        if m_clause and section_num is not None:
            flush()
            clause_num = m_clause.group(1)
            clause_buffer = [m_clause.group(2)]
            continue

        if section_num is None:
            continue  # front matter (title/disclaimer/issuing body) - not a citable clause
        if clause_num is not None:
            if line.strip():
                clause_buffer.append(line.strip())
        else:
            if line.strip():
                section_buffer.append(line.strip("- ").strip())

    flush()
    return chunks


def load_all_chunks() -> list[Chunk]:
    chunks = []
    for path in sorted(STANDARDS_DIR.glob("*.md")):
        chunks.extend(parse_document(path))
    return chunks


def main():
    for stem in ["GFPA-201", "GFPA-215"]:
        path = next(STANDARDS_DIR.glob(f"{stem}_*.md"))
        print(f"=== {path.name} ===")
        for chunk in parse_document(path):
            print(f"\n[{chunk.citation}]  (id={chunk.id})")
            print(f"  section_title: {chunk.section_title}")
            print(f"  text: {chunk.text}")
        print()


if __name__ == "__main__":
    main()
