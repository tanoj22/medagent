# Decisions

Running log of architectural choices and notable failures for MedAgent.
Each decision notes the alternative considered; each failure notes the fix.
Doubles as interview prep.

## Stack

- **LLM: Groq (GPT-OSS 120B + Qwen3.6 27B + GPT-OSS 20B)**, not OpenAI/Anthropic. Free tier, fast inference, keeps the whole project zero-cost. GPT-OSS 120B for synthesis and the LLM judge; Qwen3.6 27B for molecule/protein phrasing; GPT-OSS 20B for cheap steps (classification, name extraction). Replaced Llama 3.3 70B before Groq's 2026-08-16 shutdown (`openai/gpt-oss-120b` or `qwen/qwen3.6-27b`).
- **Embeddings: sentence-transformers all-MiniLM-L6-v2, local**, not a hosted embedding API. Free, runs on CPU, no per-call cost; good enough for abstract retrieval.
- **Vector DB: ChromaDB (local persistent)**, not Pinecone. Free, no hosted dependency, ships inside the Docker image as a single deployment unit.
- **Hybrid retrieval: dense + BM25 fused with RRF**, not dense-only. BM25 catches exact-term matches (gene/drug names) that dense embeddings blur.
- **Frontend: vanilla HTML + Alpine.js + Tailwind (CDN)**, not React or Streamlit. No build step, single FastAPI deployment unit, custom look that reads as full-stack work rather than a template.
- **Deploy: HuggingFace Spaces (Docker SDK)**, not AWS. Free CPU tier, public URL, no cloud billing.

## Corpus (Day 2)

- **Domain query**: ML/DL/protein-LM terms AND drug-discovery/comp-bio terms, 2020–2026. Deliberately tilted toward the biologics/drug-discovery interview target so demo questions land naturally.
- **Count-first instead of assuming 30K.** Actual match: 25,542 papers. Kept all (under the 30K cap); chose not to broaden the query just to hit a rounder number — no practical payoff.
- **Dropped records with no abstract** (a title alone is useless for retrieval). Final usable corpus: 25,160 abstracts.

## Failures (Day 2 fetch)

- **History-server session expiry.** First fetch used NCBI's Entrez history server (WebEnv) with retstart pagination. The session expired mid-run; every batch past ~10K returned HTTP 400. Lesson: don't lean on a server-side session for a long, slow job.
- **NCBI 10K result cap.** Second attempt paginated plain esearch to collect PMIDs, but NCBI caps any single query at 10,000 results, so the ID list silently truncated at 10K. The bug was invisible — it just stopped at the same number.
- **Year-by-year harvest (working fix).** Query each year separately (each ~2–6K, under the cap), merge/dedupe PMIDs, then fetch abstracts with *stateless* efetch by explicit ID batches. Sidesteps both the 10K cap and the flaky history server. Clean run: 25,160 abstracts.