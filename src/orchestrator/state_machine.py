"""LangGraph orchestrator: decompose a question into focused per-agent sub-queries and route."""
import json
import os
from typing import Optional, TypedDict

from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import StateGraph, START, END

from src.agents import molecular, protein_agent, pubmed_agent
from src.agents.base import AgentResponse

load_dotenv()
_client = Groq(api_key=os.environ["GROQ_API_KEY"])
CLASSIFY_MODEL = "openai/gpt-oss-20b"

_VALID = {"pubmed", "molecular", "protein"}
_NODE = {"pubmed": "run_pubmed", "molecular": "run_molecular", "protein": "run_protein"}


class MedState(TypedDict):
    query: str
    routes: list
    subqueries: dict            # agent -> focused sub-query string
    pubmed: Optional[AgentResponse]
    molecular: Optional[AgentResponse]
    protein: Optional[AgentResponse]
    answer: str


def classify(state: MedState) -> dict:
    """Decide which agents are needed AND give each a focused sub-question.

    Returns routes plus a per-agent sub-query so a compound question like
    'imatinib's properties and the BCR-ABL literature' sends only 'imatinib' to the
    molecule agent and only 'BCR-ABL inhibitors' to the literature agent, instead of
    handing the whole sentence to every agent (which caused over-reach and mis-resolution).
    """
    prompt = (
        "You route a biomedical question to specialist agents and give each agent the "
        "focused sub-question it should answer.\n\n"
        "Agents:\n"
        "- pubmed: research literature, methods, findings, what studies show\n"
        "- molecular: ONE small molecule's chemical properties (MW, logP, drug-likeness)\n"
        "- protein: ONE protein's biochemical properties (length, MW, pI, stability)\n\n"
        "Rules:\n"
        "- Only include an agent if the question genuinely needs it.\n"
        "- Only include 'protein' if a specific protein is named or clearly implied as the "
        "subject of a biochemistry question. Do NOT add protein just because a drug or "
        "disease is mentioned.\n"
        "- For 'molecular', the sub-query must be just the molecule name (e.g. 'imatinib').\n"
        "- For 'protein', the sub-query must be just the protein name (e.g. 'EGFR').\n"
        "- For 'pubmed', the sub-query is the literature topic in a few words.\n"
        "- Respond with ONLY a JSON object mapping agent name to its sub-query. No prose.\n\n"
        "Examples:\n"
        'Q: Is aspirin drug-like?\n{"molecular": "aspirin"}\n'
        'Q: What deep learning methods predict protein structure?\n'
        '{"pubmed": "deep learning for protein structure prediction"}\n'
        'Q: Biochemical properties of insulin?\n{"protein": "insulin"}\n'
        'Q: imatinib\'s properties and the literature on BCR-ABL inhibitors\n'
        '{"molecular": "imatinib", "pubmed": "BCR-ABL inhibitors"}\n'
        'Q: EGFR therapy: gefitinib properties, the EGFR protein, and resistance literature\n'
        '{"molecular": "gefitinib", "protein": "EGFR", "pubmed": "EGFR inhibitor resistance"}\n\n'
        f"Q: {state['query']}\n"
    )
    resp = _client.chat.completions.create(
        model=CLASSIFY_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    subqueries = _parse_subqueries(raw)
    if not subqueries:                       # safe fallback: treat as a literature question
        subqueries = {"pubmed": state["query"]}
    routes = [a for a in ("pubmed", "molecular", "protein") if a in subqueries]
    return {"routes": routes, "subqueries": subqueries}


def _parse_subqueries(raw: str) -> dict:
    """Pull the JSON object out of the model's reply, keeping only valid agents."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        obj = json.loads(raw[start:end + 1])
    except Exception:
        return {}
    out = {}
    for k, v in obj.items():
        k = str(k).strip().lower()
        if k in _VALID and isinstance(v, str) and v.strip():
            out[k] = v.strip()
    return out


def run_pubmed(state: MedState) -> dict:
    q = state["subqueries"].get("pubmed", state["query"])
    return {"pubmed": pubmed_agent.run(q)}


def run_molecular(state: MedState) -> dict:
    q = state["subqueries"].get("molecular", state["query"])
    return {"molecular": molecular.run(q)}


def run_protein(state: MedState) -> dict:
    q = state["subqueries"].get("protein", state["query"])
    return {"protein": protein_agent.run(q)}


def combine(state: MedState) -> dict:
    parts = []
    if state.get("pubmed"):
        parts.append("From the literature:\n" + state["pubmed"].text)
    if state.get("molecular"):
        parts.append("Molecular analysis:\n" + state["molecular"].text)
    if state.get("protein"):
        parts.append("Protein analysis:\n" + state["protein"].text)
    return {"answer": "\n\n".join(parts) if parts else "No answer produced."}


def _route(state: MedState):
    return [_NODE[r] for r in state["routes"]]


def build_graph():
    g = StateGraph(MedState)
    g.add_node("classify", classify)
    g.add_node("run_pubmed", run_pubmed)
    g.add_node("run_molecular", run_molecular)
    g.add_node("run_protein", run_protein)
    g.add_node("combine", combine)

    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", _route, ["run_pubmed", "run_molecular", "run_protein"])
    g.add_edge("run_pubmed", "combine")
    g.add_edge("run_molecular", "combine")
    g.add_edge("run_protein", "combine")
    g.add_edge("combine", END)
    return g.compile()


GRAPH = build_graph()


def ask(query: str) -> str:
    return GRAPH.invoke({"query": query})["answer"]


if __name__ == "__main__":
    tests = [
        "What deep learning methods predict protein structure?",
        "What are the molecular properties of aspirin?",
        "What are the biochemical properties of insulin?",
        "imatinib's properties and the literature on BCR-ABL inhibitors",
        "For EGFR therapy: gefitinib properties, the EGFR protein, and resistance literature",
    ]
    for q in tests:
        print(f"\n{'='*70}\nQ: {q}")
        print(ask(q))