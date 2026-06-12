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
    """Look up a reviewed protein by name on UniProt. Returns dict or None."""
    url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode({
        "query": f"{name} AND reviewed:true",
        "format": "tsv",
        "fields": "accession,protein_name,organism_name,sequence",
        "size": "1",
    })
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MedAgent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode()
    except Exception:
        return None
    lines = text.strip().split("\n")
    if len(lines) < 2:
        return None
    parts = lines[1].split("\t")
    if len(parts) < 4:
        return None
    accession, prot_name, organism, sequence = parts[:4]
    return {"accession": accession, "name": prot_name,
            "organism": organism, "sequence": sequence}


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
        "Identify the protein the user asks about. Respond with ONLY the protein's "
        "common name (e.g. insulin, hemoglobin, p53), or the exact amino-acid sequence "
        "if the user provided one. If there is no protein, respond exactly NONE.\n\n"
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