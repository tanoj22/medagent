"""Embed the PubMed corpus and index it into ChromaDB for semantic search."""
import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CORPUS = Path("data/raw/pubmed.jsonl")
CHROMA_PATH = "./chroma_db"
COLLECTION = "medagent"
MODEL_NAME = "all-MiniLM-L6-v2"
ADD_BATCH = 2000


def load_corpus():
    with CORPUS.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    print("Loading corpus...")
    records = load_corpus()
    print(f"  {len(records)} records")

    print(f"Loading model {MODEL_NAME} (downloads ~80MB the first time)...")
    model = SentenceTransformer(MODEL_NAME)

    # We embed title + abstract together so the title's keywords help retrieval.
    texts = [f"{r['title']}\n\n{r['abstract']}" for r in records]

    print("Embedding on CPU — this takes a few minutes...")
    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    )

    print("Indexing into ChromaDB...")
    client = chromadb.PersistentClient(
        path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False)
    )
    # Drop any existing collection so re-running doesn't duplicate entries.
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION)

    ids = [r["pmid"] for r in records]
    metadatas = [
        {
            "pmid": r["pmid"],
            "title": r["title"],
            "year": r["year"],
            "authors": "; ".join(r["authors"]),
        }
        for r in records
    ]

    for i in range(0, len(records), ADD_BATCH):
        j = min(i + ADD_BATCH, len(records))
        collection.add(
            ids=ids[i:j],
            embeddings=embeddings[i:j].tolist(),
            documents=texts[i:j],
            metadatas=metadatas[i:j],
        )
        print(f"  indexed {j}/{len(records)}")

    print(f"\nDone. Collection '{COLLECTION}' has {collection.count()} entries.")


if __name__ == "__main__":
    main()