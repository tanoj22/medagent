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

Researchers lose time bouncing between tools to answer one question: PubMed for the literature, a cheminformatics tool for a molecule's properties, a protein database for biochemistry. MedAgent takes a plain-English question, routes it to the right specialist agents, and returns one grounded answer.

- **Literature agent** retrieves from a 25,000-abstract PubMed corpus (hybrid BM25 + dense search) and answers with inline citations.
- **Molecule agent** computes properties (molecular weight, logP, Lipinski drug-likeness) with RDKit, resolving any drug by name via PubChem.
- **Protein agent** computes biochemistry (length, MW, isoelectric point, stability) with Biopython over UniProt sequences.

Every number is computed by a real tool, every literature claim is traceable to a source, and a three-layer hallucination guard makes the system refuse rather than guess when it cannot ground an answer.