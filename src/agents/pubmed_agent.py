"""PubMed literature agent: Week 1 retrieval + synthesis behind the agent interface."""
from src.agents.base import AgentResponse
from src.generation.synthesize import synthesize
from src.retrieval.hybrid import hybrid_search


def run(query: str, k: int = 5) -> AgentResponse:
    chunks = hybrid_search(query, k=k)
    answer = synthesize(query, chunks)
    sources = [{"citation": cid, "pmid": pmid, "title": title}
               for cid, pmid, title in answer.citations]
    return AgentResponse(agent="pubmed", text=answer.text, ok=True, sources=sources)


if __name__ == "__main__":
    r = run("What deep learning methods predict protein structure?")
    print(r.text)
    print("\nSOURCES:", [s["pmid"] for s in r.sources])