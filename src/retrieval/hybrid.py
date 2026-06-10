"""Hybrid retrieval: fuse dense + BM25 rankings with Reciprocal Rank Fusion."""
from src.retrieval.dense import Chunk, dense_search
from src.retrieval.bm25 import bm25_search

RRF_K = 60     # standard RRF constant; softens the gap between top ranks
POOL = 25      # how many to pull from each retriever before fusing
TOP_K = 5      # final results returned


def hybrid_search(query: str, k: int = TOP_K) -> list[Chunk]:
    dense_hits = dense_search(query, k=POOL)
    bm25_hits = bm25_search(query, k=POOL)

    scores = {}
    chunk_by_pmid = {}
    for hits in (dense_hits, bm25_hits):
        for rank, c in enumerate(hits, start=1):
            scores[c.pmid] = scores.get(c.pmid, 0.0) + 1.0 / (RRF_K + rank)
            chunk_by_pmid.setdefault(c.pmid, c)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    results = []
    for pmid, fused in ranked:
        c = chunk_by_pmid[pmid]
        c.distance = 1.0 / (1.0 + fused)   # carry fused score as uniform "distance" (lower = better)
        results.append(c)
    return results


if __name__ == "__main__":
    queries = [
        "graph neural network for molecular property prediction",
        "AlphaFold protein structure prediction",
        "machine learning drug toxicity",
    ]
    for q in queries:
        print(f"\nQ: {q}")
        for c in hybrid_search(q):
            print(f"   - {c.title[:80]}")