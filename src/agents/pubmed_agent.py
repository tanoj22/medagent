"""PubMed literature agent: hybrid retrieval + synthesis, behind a 3-layer guard."""
from src.agents.base import AgentResponse
from src.generation.synthesize import synthesize
from src.guards.grounding import check
from src.retrieval.hybrid import hybrid_search

REFUSAL = ("I couldn't produce an answer that's reliably grounded in the retrieved "
           "literature, so I'm not going to guess. Try rephrasing, or ask about a "
           "topic more central to the indexed corpus.")


def run(query: str, k: int = 8) -> AgentResponse:
    chunks = hybrid_search(query, k=k)
    if not chunks:
        return AgentResponse("pubmed", REFUSAL, ok=False)

    tag = query[:40]

    # First attempt
    answer = synthesize(query, chunks)
    verdict = check(query, answer.text, chunks)
    print(f"  [guard] '{tag}' first attempt: ok={verdict.ok} layer={verdict.layer} {verdict.reason}", flush=True)

    # One stricter retry if the guard flagged it
    if not verdict.ok:
        print(f"  [guard] '{tag}' retrying strict...", flush=True)
        answer = synthesize(query, chunks, strict=True)
        verdict = check(query, answer.text, chunks)
        print(f"  [guard] '{tag}' after retry: ok={verdict.ok} layer={verdict.layer} {verdict.reason}", flush=True)

    # Still bad -> refuse rather than show a possibly-hallucinated answer
    if not verdict.ok:
        return AgentResponse("pubmed", REFUSAL, ok=False,
                             sources=[{"guard": "failed", "reason": verdict.reason}])

    sources = [{"citation": cid, "pmid": pmid, "title": title}
               for cid, pmid, title in answer.citations]
    sources.append({"guard": "passed"})   # lets the UI show a "grounded ✓" badge
    return AgentResponse("pubmed", answer.text, ok=True, sources=sources)


if __name__ == "__main__":
    for q in ["What deep learning methods predict protein structure?",
              "What is the airspeed velocity of an unladen swallow?"]:
        r = run(q)
        print(f"\nQ: {q}\nOK: {r.ok}\n{r.text[:400]}", flush=True)