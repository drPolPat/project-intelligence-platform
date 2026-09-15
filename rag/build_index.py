"""
Embeds all GFPA standards chunks with fastembed and stores them in a local,
persistent ChromaDB collection. No external DB dependency — everything
(model weights, vector index) lives under rag/ and data caches locally.
"""
from pathlib import Path

import chromadb
from fastembed import TextEmbedding

from chunk_documents import load_all_chunks

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # 384-dim, small, local, no API calls
CHROMA_DIR = str(Path(__file__).resolve().parent / "chroma_db")
COLLECTION_NAME = "gfpa_standards"


def build_index():
    chunks = load_all_chunks()

    model = TextEmbedding(model_name=EMBEDDING_MODEL)
    # passage_embed applies this model's document-side prefix; queries use
    # query_embed with its own prefix (see retrieval_sanity_check.py) —
    # required for BGE-family models' asymmetric retrieval to work well.
    embeddings = list(model.passage_embed([c.embedding_text for c in chunks]))

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)  # rebuild fresh each run
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    collection.add(
        ids=[c.id for c in chunks],
        embeddings=[e.tolist() for e in embeddings],
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "document_id": c.document_id,
                "document_title": c.document_title,
                "section_id": c.section_id,
                "section_title": c.section_title,
                "citation": c.citation,
            }
            for c in chunks
        ],
    )
    return collection, len(chunks)


if __name__ == "__main__":
    _, n = build_index()
    print(f"Indexed {n} chunks into ChromaDB collection '{COLLECTION_NAME}' at {CHROMA_DIR}")
