"""
Full RAG query pipeline: retrieve top-k chunks from ChromaDB, then either
refuse or generate a cited answer via the Claude API.

REFUSAL LOGIC — two distinct, separately-logged reasons, neither of them a
guessed magic number:

  - "referenced_but_missing": the query names a GFPA-NNN document ID that the
    corpus itself mentions as a cross-reference but never actually indexed
    (e.g. "see GFPA-001, not included in this demonstration corpus"). This is
    detected deterministically by scanning the corpus for its own
    missing-document markers — not a similarity score at all — so it should
    never misfire.
  - "low_similarity": the best retrieved chunk's similarity score falls below
    SIMILARITY_REFUSAL_THRESHOLD, with no known-missing document mentioned.
    Covers genuinely off-topic questions.
  - "model_declined": a third, defensive category — retrieval passed the
    threshold, but the model itself decided the excerpts don't actually
    support an answer (structured `can_answer: false`). The similarity
    threshold is an imperfect proxy; this is the final backstop.

SIMILARITY_REFUSAL_THRESHOLD IS A PLACEHOLDER, NOT A FINDING. It is set from
the two ad hoc sanity-check queries in retrieval_sanity_check.py (in-corpus
~0.70-0.79, one off-topic example ~0.50) — two data points, not a
distribution. Treat it as a configurable knob and recalibrate it against the
full ~15-20 question eval reference set's actual in-corpus vs. out-of-corpus
score spread once that set exists (next phase) — do not ship this number
as-is.
"""
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import anthropic
import chromadb
import numpy as np
from fastembed import TextEmbedding
from pydantic import BaseModel

from build_index import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL
from chunk_documents import STANDARDS_DIR

MODEL_NAME = "claude-opus-5"
TOP_K = 5

SIMILARITY_REFUSAL_THRESHOLD = 0.55  # PLACEHOLDER — see module docstring; calibrate against the eval set, not this value

GFPA_ID_RE = re.compile(r"GFPA-\d+")


def _known_and_missing_document_ids() -> tuple[set[str], set[str]]:
    """Scans the corpus for every GFPA-NNN it mentions, split into IDs that
    were actually indexed vs. IDs only ever referenced (cross-references to
    documents deliberately left out of this demonstration corpus)."""
    indexed_ids = {p.name.split("_")[0] for p in STANDARDS_DIR.glob("*.md")}
    all_mentioned: set[str] = set()
    for path in STANDARDS_DIR.glob("*.md"):
        all_mentioned.update(GFPA_ID_RE.findall(path.read_text(encoding="utf-8")))
    return indexed_ids, all_mentioned - indexed_ids


INDEXED_DOCUMENT_IDS, KNOWN_MISSING_DOCUMENT_IDS = _known_and_missing_document_ids()


class Citation(BaseModel):
    document_id: str
    section_id: str
    supporting_quote: str


class GroundedAnswer(BaseModel):
    can_answer: bool
    answer: str
    citations: list[Citation]


class SchemaParseError(Exception):
    """Raised when the API call itself succeeded but the response could not
    be parsed into GroundedAnswer — kept distinct from a genuine model
    decline (can_answer: false on a successfully-parsed response) so the two
    are never conflated under one refusal_reason."""


@dataclass
class RagResult:
    query: str
    refused: bool
    refusal_reason: Optional[str]  # None | "referenced_but_missing" | "low_similarity" | "model_declined" | "schema_parse_failure"
    retrieved: list[dict]
    answer: Optional[str] = None
    citations: list[dict] = field(default_factory=list)
    generation_skipped: bool = False  # true when no API key was available to attempt generation
    raw_model_output: Optional[str] = None  # populated on model_declined / schema_parse_failure for diagnosis


SYSTEM_PROMPT = """You are a Q&A assistant that answers ONLY using the provided GFPA standards excerpts.

Rules:
- Every factual claim in your answer must be traceable to one of the provided excerpts.
- For every claim, include a citation with the exact document_id and section_id it came from, and a short supporting_quote copied verbatim from that excerpt.
- If the provided excerpts do not contain enough information to answer the question, set can_answer to false, leave citations empty, and explain briefly in `answer` why the corpus doesn't cover it. Do not use outside knowledge.
- Never state a number, threshold, or requirement that is not explicitly present in the excerpts."""


# Concrete facility-type terms that map to GFPA-201 Section 2's Class A/B/C
# definitions (mirrors the terms used in that chunk's own EMBEDDING_ENRICHMENT
# in chunk_documents.py, plus the corpus's own wording). When a query names
# one of these, GFPA-201 Section 2 (the classification bridge) is needed to
# answer correctly — but dense retrieval sometimes doesn't surface it in the
# top-k (rag/README.md, known-limitations point 2). Diagnosed in Phase 4: a
# calling agent that reformulates the user's question before calling this
# pipeline adds ITS OWN sampling-dependent variance on top of that gap — the
# same request can retrieve differently depending on how the agent happens
# to paraphrase it (rag/README.md point 4). Rather than relying on every
# caller (or the agent's own retry judgment) to notice and compensate, this
# closes the specific facility-type gap once, here, for every caller.
FACILITY_TYPE_KEYWORDS = [
    "museum", "mall", "shopping mall", "shopping center", "retail center", "retail store",
    "office building", "office tower", "corporate headquarters", "administrative building",
    "utility substation", "power substation", "substation", "industrial site", "industrial plant", "warehouse",
]
CLASSIFICATION_BRIDGE_CHUNK_ID = "GFPA-201_2"

# Word-boundary matching, not plain substring containment: a naive `"mall" in
# query.lower()` check would false-trigger on "small," "smaller," "mallet,"
# etc. — caught in testing before this shipped (see: "is my small facility
# exempt" contains the substring "mall"). A plain \bmall\b, in turn, missed
# the actual T2 request ("...for malls?") — plural forms have no word
# boundary before their trailing "s". `s?` before the closing \b handles
# both single- and multi-word entries correctly (it lands right after
# whichever entry matched, which for a phrase is that phrase's last word).
_FACILITY_TYPE_RE = re.compile(
    r"\b(" + "|".join(re.escape(kw) for kw in FACILITY_TYPE_KEYWORDS) + r")s?\b",
    re.IGNORECASE,
)


def _mentions_facility_type(query: str) -> bool:
    return bool(_FACILITY_TYPE_RE.search(query))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


def _ensure_classification_bridge(query: str, query_embedding, chunks: list[dict], collection) -> list[dict]:
    """Deterministic fix, not a second approximate search: if the query
    names a concrete facility type and GFPA-201 Section 2 isn't already
    among the retrieved chunks, fetch that exact chunk BY ID (we know
    precisely which chunk answers this — a second embedding search could
    itself miss it the same way the first one did) and append it. Computes
    a real cosine similarity against the query (not a placeholder) so the
    injected chunk's score is honest and behaves correctly in the
    low_similarity threshold check downstream."""
    if not _mentions_facility_type(query):
        return chunks
    already_present = any(c["document_id"] == "GFPA-201" and c["section_id"] == "2" for c in chunks)
    if already_present:
        return chunks

    fetched = collection.get(ids=[CLASSIFICATION_BRIDGE_CHUNK_ID], include=["metadatas", "documents", "embeddings"])
    if not fetched["ids"]:
        return chunks  # defensive: chunk id changed or corpus rebuilt without it — don't crash, just skip the fix

    meta = fetched["metadatas"][0]
    similarity = _cosine_similarity(query_embedding, fetched["embeddings"][0])
    bridge_chunk = {**meta, "text": fetched["documents"][0], "similarity": similarity, "injected": True}
    return chunks + [bridge_chunk]


def retrieve(query: str, embed_model: TextEmbedding, collection, k: int = TOP_K) -> list[dict]:
    query_embedding = list(embed_model.query_embed([query]))[0]
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=k)
    chunks = [
        {**meta, "text": doc, "similarity": 1 - dist}
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ]
    return _ensure_classification_bridge(query, query_embedding, chunks, collection)


def classify_pre_generation_refusal(query: str, retrieved: list[dict]) -> Optional[str]:
    mentioned_ids = set(GFPA_ID_RE.findall(query))
    if mentioned_ids & KNOWN_MISSING_DOCUMENT_IDS:
        return "referenced_but_missing"
    best_similarity = max((c["similarity"] for c in retrieved), default=0.0)
    if best_similarity < SIMILARITY_REFUSAL_THRESHOLD:
        return "low_similarity"
    return None


_LITERAL_UNICODE_ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})")


def _fix_literal_unicode_escapes(text: str) -> str:
    """claude-opus-5 (and the rest of the Fable 5 / Opus 5 / 4.6+ family) can
    occasionally double-escape a backslash when writing a non-ASCII character
    into generated JSON — emitting the 7 raw characters \\u2014 instead of the
    correct 6-character JSON escape — for an em-dash. Standards-compliant
    JSON decoding of \\u2014 correctly produces the LITERAL 6-character text
    "—" (not the em-dash) — this isn't a bug in our parsing, it's the
    model's own generated JSON being malformed-but-decodable. Documented,
    known quirk (see the Claude API skill's "Tool call JSON parsing" note for
    this model family). Confirmed live in this project: 4/4 grounding-check
    failures in the Phase 4 composite report traced to exactly this, always
    on an em-dash inside a quote (GFPA-201 Section 2 / GFPA-250 Section 5.1),
    the SAME source text correct in some calls and broken in others within
    the same run — a per-generation slip, not anything deterministic in our
    own code. Fixed once here, at the single point raw model output enters
    the system, rather than normalized separately by every downstream
    consumer (citation_utils.py's normalize_section_id fixes a DIFFERENT
    problem — a formatting choice the model makes consistently within one
    field — and stays a shared cross-consumer utility for that reason; this
    is a source-data corruption issue, so it belongs at the source instead)."""
    return _LITERAL_UNICODE_ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 16)), text)


def generate_answer(query: str, retrieved: list[dict], client: anthropic.Anthropic) -> GroundedAnswer:
    """Raises SchemaParseError (distinct from a genuine model decline) if the
    API call succeeded but the response didn't parse into GroundedAnswer —
    this is what lets answer_question tell "the model validly said it
    couldn't answer" apart from "something broke in our own parsing."""
    context = "\n\n".join(f"[{c['citation']}]\n{c['text']}" for c in retrieved)
    user_content = f"Excerpts:\n{context}\n\nQuestion: {query}"
    response = client.messages.parse(
        model=MODEL_NAME,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
        output_format=GroundedAnswer,
    )
    if response.parsed_output is None:
        raw_text = next((b.text for b in response.content if b.type == "text"), "")
        raise SchemaParseError(
            f"stop_reason={response.stop_reason!r}, raw content: {raw_text!r}"
        )
    parsed = response.parsed_output
    parsed.answer = _fix_literal_unicode_escapes(parsed.answer)
    for citation in parsed.citations:
        citation.supporting_quote = _fix_literal_unicode_escapes(citation.supporting_quote)
    return parsed


def answer_question(
    query: str,
    embed_model: TextEmbedding,
    collection,
    client: Optional[anthropic.Anthropic],
    k: int = TOP_K,
) -> RagResult:
    retrieved = retrieve(query, embed_model, collection, k=k)

    refusal_reason = classify_pre_generation_refusal(query, retrieved)
    if refusal_reason:
        return RagResult(query=query, refused=True, refusal_reason=refusal_reason, retrieved=retrieved)

    if client is None:
        return RagResult(query=query, refused=False, refusal_reason=None, retrieved=retrieved, generation_skipped=True)

    try:
        parsed = generate_answer(query, retrieved, client)
    except SchemaParseError as e:
        return RagResult(
            query=query, refused=True, refusal_reason="schema_parse_failure",
            retrieved=retrieved, raw_model_output=str(e),
        )

    if not parsed.can_answer:
        return RagResult(
            query=query, refused=True, refusal_reason="model_declined",
            retrieved=retrieved, answer=parsed.answer, raw_model_output=parsed.model_dump_json(),
        )
    return RagResult(
        query=query, refused=False, refusal_reason=None, retrieved=retrieved,
        answer=parsed.answer, citations=[c.model_dump() for c in parsed.citations],
    )


def main():
    demo_queries = [
        "what surveillance coverage is required for a museum",
        "how quickly must a critical structural vulnerability be fixed",
        "what does GFPA-001 say about camera resolution standards",
        "what does GFPA-999 say about combining multiple minor vulnerabilities",
        "what is the capital of France",
    ]

    embed_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_collection(COLLECTION_NAME)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    claude_client = anthropic.Anthropic() if api_key else None
    if claude_client is None:
        print("NOTE: no ANTHROPIC_API_KEY set — generation will be skipped; "
              "showing retrieval + refusal-classification results only.\n")

    for query in demo_queries:
        result = answer_question(query, embed_model, collection, claude_client)
        print(f"=== Query: {query!r} ===")
        # Full retrieved set (all k), not a truncated top-3 — a prior version
        # of this demo only printed the top 3 while all 5 were actually sent
        # to the model, which hid exactly the chunk the model did/didn't see.
        print(f"All {len(result.retrieved)} retrieved chunks (this is the full context sent to the model):")
        for c in result.retrieved:
            print(f"  [{c['citation']}] sim={c['similarity']:.3f}  {c['text'][:100]}{'...' if len(c['text']) > 100 else ''}")
        if result.refused:
            print(f"REFUSED — reason: {result.refusal_reason}")
            if result.answer:
                print(f"  model's stated reasoning: {result.answer}")
            if result.raw_model_output:
                print(f"  raw_model_output: {result.raw_model_output}")
        elif result.generation_skipped:
            print("Retrieval passed threshold; generation skipped (no API key).")
        else:
            print(f"ANSWER: {result.answer}")
            for c in result.citations:
                print(f"  cite: {c['document_id']} Section {c['section_id']} — \"{c['supporting_quote']}\"")
        print()


if __name__ == "__main__":
    main()
