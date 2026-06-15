"""FastAPI backend: exposes the MedAgent orchestrator over HTTP and serves the UI."""
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.orchestrator.state_machine import GRAPH

app = FastAPI(title="MedAgent")


class ChatRequest(BaseModel):
    query: str


@app.post("/api/chat")
def chat(req: ChatRequest):
    state = GRAPH.invoke({"query": req.query})
    agents = []
    for key in ("pubmed", "molecular", "protein"):
        resp = state.get(key)
        if resp:
            agents.append({
                "agent": resp.agent,
                "text": resp.text,
                "ok": resp.ok,
                "sources": resp.sources,
            })
    return {
        "routes": state.get("routes", []),
        "agents": agents,
        "answer": state["answer"],
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse("frontend/index.html")