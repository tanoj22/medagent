"""Three-layer hallucination guard for grounded (literature) answers.

Layer 1  heuristic citation check  -- cheap, no LLM
Layer 2  token-overlap grounding   -- cheap, no LLM
Layer 3  LLM-as-judge faithfulness -- one GPT-OSS 120B call
"""
import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv
from groq import Groq

from src.retrieval.dense import Chunk

load_dotenv()
_client = Groq(api_key=os.environ["GROQ_API_KEY"])
JUDGE_MODEL = "openai/gpt-oss-120b"   # stricter judging

# Tunables
MIN_OVERLAP = 0.18          # fraction of answer content-words that must appear in sources
_STOP = set("""a an the of to in for and or but with without is are was were be been being
that this these those it its as by on at from into than then so such can may might will would
we our you your they their he she his her i me my using use used based show shows showed study
studies paper papers source sources provided information data results method methods approach
approaches model models which who whom whose between among about also however therefore thus""".split())


@dataclass
class GuardResult:
    ok: bool
    layer: str          # which layer flagged it ("" if passed)
    reason: str


# ---------- Layer 1: citation sanity ----------
def _layer1_citations(answer: str, n_sources: int) -> GuardResult:
    cited = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
    # An honest "I can't answer from the sources" is allowed to have no citations.
    refusal = any(p in answer.lower() for p in
                  ("do not contain", "don't contain", "not contain", "cannot be determined",
                   "cannot determine", "no information", "not mention", "not provide",
                   "insufficient", "unable to"))
    if not cited and not refusal:
        return GuardResult(False, "L1", "answer makes claims with no citations")
    bad = [c for c in cited if c < 1 or c > n_sources]
    if bad:
        return GuardResult(False, "L1", f"cites non-existent source(s): {sorted(set(bad))}")
    return GuardResult(True, "", "")


# ---------- Layer 2: token overlap ----------
def _content_words(text: str) -> set:
    words = re.findall(r"[a-z0-9][a-z0-9\-]{2,}", text.lower())
    return {w for w in words if w not in _STOP}


def _layer2_overlap(answer: str, chunks: list[Chunk]) -> GuardResult:
    ans_words = _content_words(re.sub(r"\[\d+\]", "", answer))
    if not ans_words:
        return GuardResult(True, "", "")  # nothing to check (e.g. pure refusal)
    source_words = set()
    for c in chunks:
        source_words |= _content_words(c.text)
    overlap = len(ans_words & source_words) / len(ans_words)
    if overlap < MIN_OVERLAP:
        return GuardResult(False, "L2", f"low source overlap ({overlap:.0%})")
    return GuardResult(True, "", "")


# ---------- Layer 3: LLM-as-judge ----------
def _layer3_judge(query: str, answer: str, chunks: list[Chunk]) -> GuardResult:
    sources = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks, start=1))
    prompt = (
        "You are a strict fact-checker. Decide whether EVERY factual claim in the "
        "ANSWER is supported by the SOURCES. Saying the sources lack the information "
        "counts as SUPPORTED. If any claim adds facts not in the sources, it is "
        "UNSUPPORTED.\n\n"
        "Reply on one line: 'SUPPORTED' or 'UNSUPPORTED: <short reason>'.\n\n"
        f"QUESTION: {query}\n\nSOURCES:\n{sources}\n\nANSWER:\n{answer}"
    )
    resp = _client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=256,
        reasoning_effort="low",
        include_reasoning=False,
    )
    verdict = resp.choices[0].message.content.strip()
    if verdict.upper().startswith("SUPPORTED"):
        return GuardResult(True, "", "")
    return GuardResult(False, "L3", verdict[:160])


def check(query: str, answer: str, chunks: list[Chunk]) -> GuardResult:
    """Run all three layers in order; return the first failure, or a pass."""
    for fn in (
        lambda: _layer1_citations(answer, len(chunks)),
        lambda: _layer2_overlap(answer, chunks),
        lambda: _layer3_judge(query, answer, chunks),
    ):
        result = fn()
        if not result.ok:
            return result
    return GuardResult(True, "", "passed all three layers")