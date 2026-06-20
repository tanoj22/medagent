"""Download the prebuilt ChromaDB index AND the raw corpus from HuggingFace if missing.
Runs once at container startup; a no-op when files already exist locally."""
import os
import shutil
from huggingface_hub import snapshot_download

REPO_ID = "Tanoj22/medagent-index"
LOCAL_DIR = "./hf_index"
TARGET_INDEX = "./chroma_db"
TARGET_CORPUS = "./data/raw/pubmed.jsonl"


def ensure_data():
    have_index = os.path.exists(os.path.join(TARGET_INDEX, "chroma.sqlite3"))
    have_corpus = os.path.exists(TARGET_CORPUS)
    if have_index and have_corpus:
        print("[data] index and corpus already present, skipping download")
        return

    print(f"[data] downloading from {REPO_ID} ...")
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=LOCAL_DIR)

    # Move the index into place
    src_index = os.path.join(LOCAL_DIR, "chroma_db")
    if os.path.isdir(src_index) and not have_index:
        shutil.move(src_index, TARGET_INDEX)

    # Move the corpus into place
    src_corpus = os.path.join(LOCAL_DIR, "data", "raw", "pubmed.jsonl")
    if os.path.isfile(src_corpus) and not have_corpus:
        os.makedirs(os.path.dirname(TARGET_CORPUS), exist_ok=True)
        shutil.move(src_corpus, TARGET_CORPUS)

    print("[data] index ready at", TARGET_INDEX, "| corpus ready at", TARGET_CORPUS)


if __name__ == "__main__":
    ensure_data()