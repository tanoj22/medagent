"""Protein agent: fetch a protein from UniProt, compute biochemical properties via Biopython."""
import json
import os
import urllib.parse
import urllib.request

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from dotenv import load_dotenv
from groq import Groq

from src.agents.base import AgentResponse

load_dotenv()
_client = Groq(api_key=os.environ["GROQ_API_KEY"])
EXTRACT_MODEL = "llama-3.1-8b-instant"
PRESENT_MODEL = "llama-3.3-70b-versatile"

_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _looks_like_sequence(s: str) -> bool:
    s = s.strip().upper()
    return len(s) >= 20 and all(c in _AA for c in s)


def _fetch_from_uniprot(name: str):
    """Look up a reviewed protein by name/symbol on UniProt. Returns dict or None.

    Fetches several candidates and prefers the entry whose PRIMARY gene name matches,
    which avoids ambiguous synonym collisions (e.g. 'ALB' is a synonym on an unrelated
    protein but the primary gene of albumin)."""
    query = f'({name}) AND (organism_id:9606) AND (reviewed:true)'
    url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode({
        "query": query,
        "format": "tsv",
        "fields": "accession,protein_name,organism_name,gene_primary,sequence",
        "size": "10",
    })
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MedAgent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode()
    except Exception:
        return None

    rows = [r.split("\t") for r in text.strip().split("\n")[1:] if r]
    rows = [r for r in rows if len(r) >= 5 and r[4]]   # must have a sequence

    if not rows:
        # fall back to any organism, reviewed
        return _fetch_any_organism(name)

    nl = name.lower()
    # 1) Prefer an exact primary-gene-name match (col index 3 = gene_primary)
    for acc, prot, org, gene, seq in rows:
        if gene.lower() == nl:
            return {"accession": acc, "name": prot, "organism": org, "sequence": seq}
    # 2) Otherwise prefer an exact protein-name match
    for acc, prot, org, gene, seq in rows:
        if prot.lower() == nl or prot.lower().startswith(nl + " "):
            return {"accession": acc, "name": prot, "organism": org, "sequence": seq}
    # 3) Otherwise the top hit
    acc, prot, org, gene, seq = rows[0]
    return {"accession": acc, "name": prot, "organism": org, "sequence": seq}


def _fetch_any_organism(name: str):
    url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode({
        "query": f'({name}) AND (reviewed:true)',
        "format": "tsv",
        "fields": "accession,protein_name,organism_name,gene_primary,sequence",
        "size": "5",
    })
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MedAgent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode()
    except Exception:
        return None
    for r in text.strip().split("\n")[1:]:
        parts = r.split("\t")
        if len(parts) >= 5 and parts[4]:
            return {"accession": parts[0], "name": parts[1],
                    "organism": parts[2], "sequence": parts[4]}
    return None

def compute_protein_properties(sequence: str) -> dict | None:
    seq = sequence.strip().upper()
    if not seq or any(c not in _AA for c in seq):
        return None  # non-standard residues (X, U, etc.) — skip
    pa = ProteinAnalysis(seq)
    return {
        "length": len(seq),
        "molecular_weight": round(pa.molecular_weight(), 2),
        "isoelectric_point": round(pa.isoelectric_point(), 2),
        "instability_index": round(pa.instability_index(), 2),
        "gravy": round(pa.gravy(), 3),
        "aromaticity": round(pa.aromaticity(), 3),
    }


def _extract_protein(query: str) -> str | None:
    prompt = (
        "Identify the protein the user is asking about. Respond with ONLY its standard "
        "gene symbol if one exists (e.g. EGFR, TP53, INS, ALB), otherwise its common name. "
        "Use the official symbol, not a description. No extra words. "
        "If there is no protein, respond exactly NONE.\n\n"
        f"User: {query}"
    )
    resp = _client.chat.completions.create(
        model=EXTRACT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0)
    answer = resp.choices[0].message.content.strip().strip('."\'')
    return None if answer.upper() == "NONE" else answer


def run(query: str) -> AgentResponse:
    ref = _extract_protein(query)
    if ref is None:
        return AgentResponse("protein", "No protein identified in the question.", ok=False)

    if _looks_like_sequence(ref):
        name, sequence = "the provided sequence", ref
    else:
        rec = _fetch_from_uniprot(ref)
        if rec is None:
            return AgentResponse("protein", f"Could not find protein '{ref}' on UniProt.", ok=False)
        name, sequence = f"{rec['name']} ({rec['organism']}, {rec['accession']})", rec["sequence"]

    props = compute_protein_properties(sequence)
    if props is None:
        return AgentResponse("protein", f"Could not analyze the sequence for '{name}'.", ok=False)

    present = _client.chat.completions.create(
        model=PRESENT_MODEL,
        messages=[{"role": "user", "content": (
            "You are a protein biochemistry assistant. Using ONLY these computed values, "
            "write a concise factual summary for a researcher: state the length, molecular "
            "weight, isoelectric point (pI), GRAVY hydropathy, instability index, and "
            "aromaticity. Briefly note what the values suggest (instability index > 40 "
            "suggests an unstable protein; positive GRAVY is hydrophobic, negative is "
            "hydrophilic). Add no outside facts.\n\n"
            f"Protein: {name}\nProperties: {json.dumps(props)}"
        )}],
        temperature=0.2,
    )
    return AgentResponse("protein", present.choices[0].message.content, ok=True,
                         sources=[{"protein": name, "properties": props}])


if __name__ == "__main__":
    for q in [
        "What are the biochemical properties of insulin?",
        "Tell me about human hemoglobin",
        "What's the capital of France?",
    ]:
        print(f"\nQ: {q}")
        print(run(q).text)