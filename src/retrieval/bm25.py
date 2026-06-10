"""BM25 (keyword) retrieval over the same PubMed corpus."""
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.retrieval.dense import Chunk  # reuse the shared Chunk

CORPUS = Path("data/raw/pubmed.jsonl")
_TOKEN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


_bm25 = None
_records = None


def _build_index():
    global _bm25, _records
    if _bm25 is None:
        print("Building BM25 index (a few seconds)...")
        _records = []
        with CORPUS.open(encoding="utf-8") as f:
            for line in f:
                _records.append(json.loads(line))
        tokenized = [_tokenize(f"{r['title']} {r['abstract']}") for r in _records]
        _bm25 = BM25Okapi(tokenized)
    return _bm25, _records


def bm25_search(query: str, k: int = 5) -> list[Chunk]:
    bm25, records = _build_index()
    scores = bm25.get_scores(_tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    chunks = []
    for i in top:
        r = records[i]
        chunks.append(Chunk(
            pmid=r["pmid"],
            title=r["title"],
            text=f"{r['title']}\n\n{r['abstract']}",
            year=r["year"],
            authors="; ".join(r["authors"]),
            distance=1.0 / (1.0 + scores[i]),  # higher BM25 score -> lower "distance", uniform with dense
        ))
    return chunks


if __name__ == "__main__":
    from src.retrieval.dense import dense_search

    queries = [
        "machine learning for predicting drug toxicity",
        "AlphaFold protein structure",
        "SMILES molecular representation",
        "scRNA-seq",
        "graph neural network",
    ]
    for q in queries:
        print(f"\nQ: {q}")
        print("  DENSE (meaning):")
        for c in dense_search(q, k=3):
            print(f"    - {c.title[:75]}")
        print("  BM25 (keywords):")
        for c in bm25_search(q, k=3):
            print(f"    - {c.title[:75]}")