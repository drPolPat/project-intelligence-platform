"""
Retrieval sanity check — run BEFORE building the full RAG query pipeline.
Embeds a few test queries and prints the top-k retrieved chunks with
similarity scores, to confirm the index is pulling the right document and
section before any generation logic is layered on top.
"""
import chromadb
from fastembed import TextEmbedding

from build_index import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

TEST_QUERIES = [
    "what surveillance coverage is required for a museum",
    # bonus checks, not required for this step but cheap and informative:
    "how quickly must a critical structural vulnerability be fixed",
    "what is the capital of France",  # deliberately out-of-corpus
]


def search(query: str, model: TextEmbedding, collection, k: int = 3):
    query_embedding = list(model.query_embed([query]))[0]
    return collection.query(query_embeddings=[query_embedding.tolist()], n_results=k)


def main():
    model = TextEmbedding(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    for query in TEST_QUERIES:
        print(f"\n=== Query: {query!r} ===")
        results = search(query, model, collection, k=3)
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
            similarity = 1 - dist  # cosine space: distance = 1 - cosine similarity
            print(f"{rank}. [{meta['citation']}] similarity={similarity:.3f}")
            print(f"   {doc}")


if __name__ == "__main__":
    main()
