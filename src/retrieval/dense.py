"""Dense (semantic) retrieval over the ChromaDB corpus."""
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_db"
COLLECTION = "medagent"
MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class Chunk:
    pmid: str
    title: str
    text: str          # the embedded title + abstract
    year: str
    authors: str
    distance: float    # lower = more similar (ChromaDB default L2)


# Lazy singletons: load the model and DB once, reuse across all queries.
_model = None
_collection = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(
            path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False)
        )
        _collection = client.get_collection(COLLECTION)
    return _collection


def dense_search(query: str, k: int = 5) -> list[Chunk]:
    """Return the k chunks most semantically similar to the query."""
    query_emb = _get_model().encode([query]).tolist()
    res = _get_collection().query(query_embeddings=query_emb, n_results=k)

    chunks = []
    for pmid, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        chunks.append(Chunk(
            pmid=pmid,
            title=meta.get("title", ""),
            text=doc,
            year=meta.get("year", ""),
            authors=meta.get("authors", ""),
            distance=dist,
        ))
    return chunks


if __name__ == "__main__":
    test_queries = [
        "machine learning for predicting drug toxicity",
        "protein structure prediction with deep learning",
        "graph neural networks for molecular property prediction",
        "transformer models for protein sequences",
        "generative models for de novo drug design",
        "predicting drug-target binding affinity",
        "single-cell RNA sequencing analysis methods",
        "reinforcement learning for molecule optimization",
        "antibody design with deep learning",
        "explainability in clinical machine learning models",
    ]
    for q in test_queries:
        print(f"\nQ: {q}")
        for c in dense_search(q, k=3):
            print(f"   [{c.distance:.3f}] {c.title[:80]}")