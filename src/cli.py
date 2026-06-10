"""MedAgent CLI: ask a question, get a cited answer from the PubMed corpus."""
from src.retrieval.hybrid import hybrid_search
from src.generation.synthesize import synthesize


def answer_question(query: str):
    chunks = hybrid_search(query, k=5)
    answer = synthesize(query, chunks)

    print("\n" + "=" * 72)
    print("ANSWER\n")
    print(answer.text)
    print("\nSOURCES")
    for cid, pmid, title in answer.citations:
        print(f"  [{cid}] PMID {pmid} — {title}")
    print("=" * 72 + "\n")


def main():
    print("MedAgent — ask a biomedical research question ('quit' to exit).\n")
    while True:
        try:
            query = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not query:
            continue
        if query.lower() in {"quit", "exit"}:
            print("Bye.")
            break
        answer_question(query)


if __name__ == "__main__":
    main()