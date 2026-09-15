"""
Shared citation-formatting utilities used across the RAG pipeline, the eval
harness, and the agent layer.

This exists because normalize_section_id was independently duplicated in
rag/eval/run_eval.py and agent/draft_risk_report.py — same logic, same root
cause, written twice. Consolidated here so every citation-key comparison in
the project uses the same normalization and can't silently drift apart again.
"""


def normalize_section_id(section_id: str) -> str:
    """Strip a leading "Section " (any case) that a model sometimes echoes
    from the "[GFPA-201 Section 3.1]" citation label it sees in its prompt
    context, even though callers ask for the bare section id ("3.1"). Without
    this, a citation naming the exact right section fails a (document_id,
    section_id) lookup on a pure string mismatch — diagnosed as the root
    cause of a citation_validity/faithfulness divergence in the Phase 3 eval
    run (29 of 36 real citations used this "Section 3.1" form) and confirmed
    to recur in the Phase 4 composite report's own citation display."""
    s = section_id.strip()
    if s.lower().startswith("section "):
        s = s[len("section "):].strip()
    return s
