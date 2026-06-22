---
title: MedAgent
emoji: 🧬
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# MedAgent 🧬

A grounded multi-agent assistant for biomedical and drug-discovery research.

**Live demo:** https://tanoj22-medagent.hf.space/
*(First load may take ~30s while the Space wakes up.)*

Answering one research question often means juggling three tools: PubMed for the literature, a cheminformatics tool for a molecule, a protein database for biochemistry. MedAgent takes a plain-English question, routes it to the right specialist agents, and returns one grounded answer. Every number is computed by a real tool, never the language model, and literature claims link to real sources.

## How it works

A LangGraph orchestrator reads each question, decomposes it into focused sub-questions, and dispatches only the agents that are actually needed:

- **Literature agent** runs hybrid retrieval (BM25 + dense embeddings, fused with Reciprocal Rank Fusion) over a fixed corpus of 25,160 PubMed abstracts, and answers with inline citations to real PMIDs.
- **Molecule agent** computes properties (molecular weight, logP, TPSA, H-bond donors/acceptors, Lipinski drug-likeness) with RDKit, resolving any drug by name through a live PubChem lookup.
- **Protein agent** computes biochemistry (length, molecular weight, isoelectric point, instability index, GRAVY, aromaticity) with Biopython, fetching sequences from a live UniProt lookup (resolved by primary gene name to avoid synonym collisions).

A three-layer grounding guard runs on literature answers: a citation check, token-overlap verification, and an LLM-as-judge faithfulness check. If an answer cannot be grounded, the system retries under stricter constraints and then refuses rather than guessing.

## Evaluation

A 22-case benchmark measures correctness against external references:

| Category | Result |
|---|---|
| Routing accuracy | 10/10 (100%) |
| Molecule correctness (vs PubChem) | 5/6 (83%) |
| Protein correctness (vs UniProt) | 3/3 (100%) |
| Refusal accuracy (out-of-scope) | 3/3 (100%) |
| **Overall** | **21/22 (95%)** |

The single molecule "miss" is a calibration nuance: RDKit's computed logP for acetaminophen (1.35) differs from the experimental value (0.34), which is expected since computed and measured logP are different quantities.

## What it uses (and what it does not)

- **Live:** PubChem (molecule structures) and UniProt (protein sequences) are queried in real time.
- **Fixed snapshot:** the literature corpus is a one-time harvest of ~25K PubMed abstracts focused on computational drug discovery. It is not a live PubMed search and does not include full-text papers.
- **Compute tools, not data sources:** RDKit and Biopython calculate properties; they are not databases.

This is a portfolio project, not a clinical tool.

## Tech stack

Python, LangGraph, Groq (Llama 3.3 70B and 3.1 8B), ChromaDB, sentence-transformers (all-MiniLM-L6-v2), rank-bm25, RDKit, Biopython, FastAPI, Docker. Deployed on HuggingFace Spaces.

## Running locally

```bash
pip install -r requirements.txt
# set GROQ_API_KEY in a .env file
python download_index.py          # pulls the prebuilt index + corpus from HuggingFace
uvicorn src.api.main:app --reload --port 8000
```

Then open http://localhost:8000.