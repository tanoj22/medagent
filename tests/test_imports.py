"""Lightweight CI smoke test: confirm all modules import without error."""

def test_imports():
    import src.orchestrator.state_machine
    import src.agents.molecular
    import src.agents.protein_agent
    import src.agents.pubmed_agent
    import src.retrieval.hybrid
    import src.retrieval.dense
    import src.retrieval.bm25
    import src.generation.synthesize
    import src.guards.grounding
    import src.api.main
