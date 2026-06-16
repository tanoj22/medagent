"""MedAgent evaluation harness.

Measures how accurate the system is across three dimensions:
  1. Routing accuracy   -- does the classifier send each question to the right agent(s)?
  2. Tool correctness   -- do the molecule/protein agents return values that match
                           known reference values (within tolerance)?
  3. Refusal accuracy   -- are out-of-scope / nonsense questions correctly refused?

Run from the project root:
    python -m eval.run_eval
"""
from src.agents import molecular, protein_agent
from src.orchestrator.state_machine import classify


# --------------------------------------------------------------------------
# Benchmark 1: ROUTING -- (question, expected set of agents)
# --------------------------------------------------------------------------
ROUTING_CASES = [
    ("What deep learning methods predict protein structure?", {"pubmed"}),
    ("What machine learning approaches predict drug-target interaction?", {"pubmed"}),
    ("What are the molecular properties of aspirin?", {"molecular"}),
    ("Is ibuprofen drug-like?", {"molecular"}),
    ("What is the molecular weight of caffeine?", {"molecular"}),
    ("What are the biochemical properties of insulin?", {"protein"}),
    ("Tell me the isoelectric point of lysozyme.", {"protein"}),
    ("Analyze the stability of hemoglobin.", {"protein"}),
    ("What does the literature say about logP, and what is the logP of ibuprofen?",
     {"pubmed", "molecular"}),
    ("What is the molecular weight of caffeine, and what research exists on its effects?",
     {"pubmed", "molecular"}),
]


# --------------------------------------------------------------------------
# Benchmark 2: MOLECULE CORRECTNESS -- reference values from PubChem.
# (metric, expected, absolute tolerance)
# --------------------------------------------------------------------------
MOLECULE_CASES = [
    # query,                         molecular_weight, logp,  tol_mw, tol_logp
    ("properties of aspirin",        180.16,  1.31, 0.5, 0.4),
    ("properties of ibuprofen",      206.28,  3.07, 0.5, 0.5),
    ("properties of caffeine",       194.19, -1.03, 0.5, 0.5),
    ("properties of acetaminophen",  151.16,  0.34, 0.5, 0.5),
    ("properties of atorvastatin",   558.64,  6.31, 1.0, 0.8),  # via PubChem fallback
    ("properties of gefitinib",      446.90,  4.28, 1.0, 0.8),  # via PubChem fallback
]


# --------------------------------------------------------------------------
# Benchmark 3: PROTEIN CORRECTNESS -- reference values from UniProt.
# (query, expected_length, length_tol)  -- length is the most stable cross-check
# --------------------------------------------------------------------------
PROTEIN_CASES = [
    ("biochemical properties of insulin", 110, 0),       # canonical precursor P01308
    ("properties of hemoglobin",          147, 3),        # beta subunit P68871
    ("properties of lysozyme",            148, 4),        # lysozyme C P61626 (incl. signal)
]


# --------------------------------------------------------------------------
# Benchmark 4: REFUSAL -- questions the system should NOT answer.
# (query, agent_to_test)
# --------------------------------------------------------------------------
REFUSAL_CASES = [
    ("What is the capital of France?", "protein"),
    ("What is the airspeed velocity of an unladen swallow?", "molecular"),
    ("Tell me a joke about cats.", "molecular"),
]


def _pct(n, d):
    return f"{(100.0 * n / d):.0f}%" if d else "n/a"


def eval_routing():
    print("\n=== 1. ROUTING ACCURACY ===")
    correct = 0
    for query, expected in ROUTING_CASES:
        got = set(classify({"query": query})["routes"])
        ok = got == expected
        correct += ok
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {query[:55]:55s} expected={sorted(expected)} got={sorted(got)}")
    print(f"  Routing accuracy: {correct}/{len(ROUTING_CASES)} ({_pct(correct, len(ROUTING_CASES))})")
    return correct, len(ROUTING_CASES)


def eval_molecules():
    print("\n=== 2. MOLECULE CORRECTNESS (vs PubChem reference) ===")
    correct = 0
    for query, exp_mw, exp_logp, tol_mw, tol_logp in MOLECULE_CASES:
        r = molecular.run(query)
        if not r.ok or not r.sources:
            print(f"  [FAIL] {query[:45]:45s} agent could not resolve")
            continue
        p = r.sources[0]["properties"]
        mw, logp = p["molecular_weight"], p["logp"]
        ok = abs(mw - exp_mw) <= tol_mw and abs(logp - exp_logp) <= tol_logp
        correct += ok
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {query[:32]:32s} MW {mw:7.2f} (exp {exp_mw:7.2f})  "
              f"logP {logp:6.2f} (exp {exp_logp:6.2f})")
    print(f"  Molecule correctness: {correct}/{len(MOLECULE_CASES)} ({_pct(correct, len(MOLECULE_CASES))})")
    return correct, len(MOLECULE_CASES)


def eval_proteins():
    print("\n=== 3. PROTEIN CORRECTNESS (vs UniProt reference) ===")
    correct = 0
    for query, exp_len, tol in PROTEIN_CASES:
        r = protein_agent.run(query)
        if not r.ok or not r.sources:
            print(f"  [FAIL] {query[:45]:45s} agent could not resolve")
            continue
        length = r.sources[0]["properties"]["length"]
        ok = abs(length - exp_len) <= tol
        correct += ok
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {query[:40]:40s} length {length:4d} (exp {exp_len} +/-{tol})")
    print(f"  Protein correctness: {correct}/{len(PROTEIN_CASES)} ({_pct(correct, len(PROTEIN_CASES))})")
    return correct, len(PROTEIN_CASES)


def eval_refusals():
    print("\n=== 4. REFUSAL ACCURACY (should NOT answer) ===")
    agents = {"molecular": molecular, "protein": protein_agent}
    correct = 0
    for query, which in REFUSAL_CASES:
        r = agents[which].run(query)
        ok = not r.ok            # correct behaviour is a refusal (ok=False)
        correct += ok
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {query[:50]:50s} refused={not r.ok}")
    print(f"  Refusal accuracy: {correct}/{len(REFUSAL_CASES)} ({_pct(correct, len(REFUSAL_CASES))})")
    return correct, len(REFUSAL_CASES)


if __name__ == "__main__":
    print("=" * 70)
    print("MedAgent evaluation")
    print("=" * 70)

    results = [
        ("Routing", *eval_routing()),
        ("Molecule correctness", *eval_molecules()),
        ("Protein correctness", *eval_proteins()),
        ("Refusal", *eval_refusals()),
    ]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    tot_c = tot_n = 0
    for name, c, n in results:
        tot_c += c
        tot_n += n
        print(f"  {name:24s} {c}/{n}  ({_pct(c, n)})")
    print(f"  {'OVERALL':24s} {tot_c}/{tot_n}  ({_pct(tot_c, tot_n)})")
    print("=" * 70)