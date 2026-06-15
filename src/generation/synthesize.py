"""Synthesize a grounded, cited answer from retrieved chunks using Groq."""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from groq import Groq

from src.retrieval.dense import Chunk

load_dotenv()
_client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a biomedical research assistant. Answer the user's \
question using ONLY the numbered sources provided.

Rules:
- Ground every claim in the sources. Do not use outside knowledge.
- Cite sources inline with bracketed numbers like [1], [2] right after the claim \
they support. Combine when needed, e.g. [1][3].
- If the sources don't contain enough information to answer, say so plainly. \
Do not guess or fill gaps.
- Be concise and factual. Write for a researcher."""

STRICT_SUFFIX = """

IMPORTANT: A previous attempt was flagged as insufficiently grounded. Be extra \
conservative: include ONLY claims you can point to a specific source for, cite every \
sentence, and if the sources don't clearly answer the question, say so plainly rather \
than inferring."""


@dataclass
class Answer:
    text: str
    citations: list = field(default_factory=list)  # (citation_id, pmid, title)


def _format_sources(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks, start=1))


def synthesize(query: str, chunks: list[Chunk], strict: bool = False) -> Answer:
    system = SYSTEM_PROMPT + (STRICT_SUFFIX if strict else "")
    user_prompt = f"Question: {query}\n\nSources:\n{_format_sources(chunks)}"
    resp = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0 if strict else 0.2,
    )
    text = resp.choices[0].message.content
    citations = [(i + 1, c.pmid, c.title) for i, c in enumerate(chunks)]
    return Answer(text=text, citations=citations)


if __name__ == "__main__":
    from src.retrieval.hybrid import hybrid_search

    q = "What machine learning methods are used to predict drug toxicity?"
    chunks = hybrid_search(q, k=5)
    answer = synthesize(q, chunks)
    print("ANSWER:\n")
    print(answer.text)
    print("\nSOURCES:")
    for cid, pmid, title in answer.citations:
        print(f"  [{cid}] PMID {pmid} — {title[:70]}")