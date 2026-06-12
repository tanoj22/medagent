"""LangGraph orchestrator: classify a question and route to the needed agent(s)."""
import os
from typing import Optional, TypedDict

from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import StateGraph, START, END

from src.agents import molecular, protein_agent, pubmed_agent
from src.agents.base import AgentResponse

load_dotenv()
_client = Groq(api_key=os.environ["GROQ_API_KEY"])
CLASSIFY_MODEL = "llama-3.1-8b-instant"

_LABELS = {"PUBMED": "pubmed", "MOLECULAR": "molecular", "PROTEIN": "protein"}
_NODE = {"pubmed": "run_pubmed", "molecular": "run_molecular", "protein": "run_protein"}


class MedState(TypedDict):
    query: str
    routes: list
    pubmed: Optional[AgentResponse]
    molecular: Optional[AgentResponse]
    protein: Optional[AgentResponse]
    answer: str


def classify(state: MedState) -> dict:
    prompt = (
        "Decide which specialists are needed to answer the question. Options:\n"
        "- PUBMED: research literature, methods, findings, what studies show\n"
        "- MOLECULAR: a small molecule's chemical properties (MW, logP, drug-likeness)\n"
        "- PROTEIN: a protein's biochemical properties (length, MW, pI, stability)\n"
        "A question may need one or several. Respond with ONLY a comma-separated list "
        "of the needed labels, e.g. 'PUBMED' or 'PUBMED,MOLECULAR' or 'PROTEIN'.\n\n"
        f"Question: {state['query']}"
    )
    resp = _client.chat.completions.create(
        model=CLASSIFY_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip().upper()
    routes = [_LABELS[t] for t in raw.replace(" ", "").split(",") if t in _LABELS]
    return {"routes": routes or ["pubmed"]}  # default to literature


def run_pubmed(state: MedState) -> dict:
    return {"pubmed": pubmed_agent.run(state["query"])}


def run_molecular(state: MedState) -> dict:
    return {"molecular": molecular.run(state["query"])}


def run_protein(state: MedState) -> dict:
    return {"protein": protein_agent.run(state["query"])}


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
        "What deep learning methods predict protein structure?",      # PUBMED
        "What are the molecular properties of aspirin?",              # MOLECULAR
        "What are the biochemical properties of insulin?",            # PROTEIN
        "What does the literature say about logP, and what's the logP of ibuprofen?",  # PUBMED + MOLECULAR
    ]
    for q in tests:
        print(f"\n{'='*70}\nQ: {q}")
        print(ask(q))