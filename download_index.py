"""Download the prebuilt ChromaDB index from HuggingFace if it isn't already present.
Runs once at container startup; a no-op when the index already exists locally."""
import os
import shutil
from huggingface_hub import snapshot_download

REPO_ID = "Tanoj22/medagent-index"
LOCAL_DIR = "./hf_index"          # snapshot lands here
TARGET = "./chroma_db"            # where dense.py expects the index


def ensure_index():
    # If a usable index is already here (e.g. local dev), do nothing.
    if os.path.exists(os.path.join(TARGET, "chroma.sqlite3")):
        print("[index] chroma_db already present, skipping download")
        return

    print(f"[index] downloading index from {REPO_ID} ...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
    )

    # The repo stores the index under chroma_db/ ; move it to ./chroma_db
    src = os.path.join(LOCAL_DIR, "chroma_db")
    if not os.path.isdir(src):
        raise RuntimeError(f"expected {src} in the downloaded snapshot")
    shutil.move(src, TARGET)
    print("[index] index ready at", TARGET)


if __name__ == "__main__":
    ensure_index()