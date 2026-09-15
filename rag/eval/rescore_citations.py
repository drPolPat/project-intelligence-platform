"""
One-off re-scoring utility: re-applies the fixed score_citation_validity()
to an existing eval_results.json WITHOUT calling the API again. The raw
model_citations (document_id, section_id, supporting_quote) are already on
disk from the original run — only the validator's key-matching logic was
buggy, so this rebuilds the corpus lookup fresh from ChromaDB (also free,
no API key needed) and re-scores every row in place.
"""
import json
from pathlib import Path

import chromadb

from build_index import CHROMA_DIR, COLLECTION_NAME
from run_eval import RESULTS_PATH, build_corpus_lookup, score_citation_validity


def main():
    rows = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    corpus_lookup = build_corpus_lookup(collection)

    print(f"{'id':4s} {'old citval':>10s} {'new citval':>10s}")
    changed = 0
    for row in rows:
        old = row["citation_validity"]
        if not row["model_citations"]:
            continue  # refusal cases - stays None
        new = score_citation_validity(row["model_citations"], corpus_lookup)
        marker = "" if old == new else "  <- changed"
        print(f"{row['id']:4s} {str(old):>10s} {str(round(new, 3)):>10s}{marker}")
        if old != new:
            changed += 1
        row["citation_validity"] = new

    RESULTS_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    answered = [r for r in rows if r["citation_validity"] is not None]
    new_overall = sum(r["citation_validity"] for r in answered) / len(answered)
    print(f"\n{changed}/{len(answered)} rows changed.")
    print(f"Overall citation_validity (fixed validator): {new_overall:.1%}  (n={len(answered)})")
    print(f"Rewrote {RESULTS_PATH} with corrected citation_validity values.")


if __name__ == "__main__":
    main()
