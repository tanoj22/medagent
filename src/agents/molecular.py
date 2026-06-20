"""Molecular small-molecule agent: compute drug-discovery properties via RDKit."""
import json
import os
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from groq import Groq
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from src.agents.base import AgentResponse

load_dotenv()
_client = Groq(api_key=os.environ["GROQ_API_KEY"])
EXTRACT_MODEL = "llama-3.1-8b-instant"      # cheap model for extraction
PRESENT_MODEL = "llama-3.3-70b-versatile"   # stronger model for phrasing

# Fast-path name -> SMILES for small, common molecules with simple, verified structures.
# Anything not here (larger drugs, etc.) is resolved live via PubChem.
KNOWN_MOLECULES = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "caffeine": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "acetaminophen": "CC(=O)Nc1ccc(O)cc1",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "metformin": "CN(C)C(=N)N=C(N)N",
    "dopamine": "NCCc1ccc(O)c(O)c1",
    "serotonin": "NCCc1c[nH]c2ccc(O)cc12",
    "nicotine": "CN1CCCC1c1cccnc1",
    "glucose": "OCC1OC(O)C(O)C(O)C1O",
    "ethanol": "CCO",
    "benzene": "c1ccccc1",
    "naproxen": "COc1ccc2cc(C(C)C(=O)O)ccc2c1",
    "diclofenac": "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
}


def compute_properties(smiles: str) -> dict | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return {
        "canonical_smiles": Chem.MolToSmiles(mol),
        "molecular_weight": round(mw, 2),
        "logp": round(logp, 2),
        "h_bond_donors": hbd,
        "h_bond_acceptors": hba,
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "lipinski_violations": violations,
        "drug_like": violations <= 1,
    }


def _mol_block_3d(smiles: str) -> str | None:
    """Generate 3D atom coordinates and return them as a MOL block."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    return Chem.MolToMolBlock(mol)


def _pubchem_smiles(name: str) -> str | None:
    """Resolve a molecule/drug name to a SMILES string via the PubChem REST API."""
    try:
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{quote(name)}/property/CanonicalSMILES/TXT"
        )
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            smiles = r.text.strip().split("\n")[0].strip()
            if smiles and Chem.MolFromSmiles(smiles) is not None:
                return smiles
    except Exception:
        return None
    return None


def _extract_molecule(query: str) -> str | None:
    """Pull the molecule NAME (or a user-provided SMILES) out of the question.
    Never lets the model invent a SMILES from a name."""
    prompt = (
        "You extract the chemical the user is asking about.\n"
        "- If the user NAMES a chemical (e.g. atorvastatin, aspirin), output that NAME "
        "only, in lowercase.\n"
        "- If the user pasted a raw SMILES string, output that SMILES exactly.\n"
        "- NEVER convert a name into a SMILES. NEVER generate, recall, or guess a "
        "SMILES. Only output a SMILES if it literally appears in the user's message.\n"
        "- If there is no chemical, output exactly NONE.\n"
        "Output only the name or the SMILES, nothing else.\n\n"
        f"User: {query}\n"
        "Answer:"
    )
    resp = _client.chat.completions.create(
        model=EXTRACT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=60,
    )
    answer = resp.choices[0].message.content.strip().strip('."\'').split("\n")[0].strip()
    return None if not answer or answer.upper() == "NONE" else answer


def _resolve_to_smiles(ref: str):
    """Resolve a name or SMILES to (display_name, smiles). Falls back to PubChem."""
    key = ref.lower().strip()
    if key in KNOWN_MOLECULES:               # 1) fast-path known names
        return key, KNOWN_MOLECULES[key]
    if Chem.MolFromSmiles(ref) is not None:  # 2) already a valid SMILES
        return ref, ref
    smiles = _pubchem_smiles(ref)            # 3) live PubChem name lookup
    if smiles:
        return ref, smiles
    return None


def run(query: str) -> AgentResponse:
    """Agent entry point: question -> RDKit-computed properties + 3D coords + grounded summary."""
    ref = _extract_molecule(query)
    if ref is None:
        return AgentResponse("molecular", "No molecule identified in the question.", ok=False)

    resolved = _resolve_to_smiles(ref)
    if resolved is None:
        return AgentResponse(
            "molecular",
            f"Could not resolve '{ref}' to a structure (not a known name, a valid "
            "SMILES, or found in PubChem).",
            ok=False,
        )

    name, smiles = resolved
    props = compute_properties(smiles)
    if props is None:
        return AgentResponse("molecular", f"Could not parse the structure for '{name}'.", ok=False)

    molblock = _mol_block_3d(smiles)

    present = _client.chat.completions.create(
        model=PRESENT_MODEL,
        messages=[{
            "role": "user",
            "content": (
                "You are a cheminformatics assistant. Using ONLY the RDKit-computed "
                "values provided below, write a short factual summary of this ONE "
                "molecule for a researcher: state molecular weight, logP, H-bond "
                "donors, H-bond acceptors, TPSA, and rotatable bonds, then give the "
                "Lipinski Rule of Five drug-likeness verdict and briefly why.\n\n"
                "STRICT RULES:\n"
                "- Use ONLY the numbers below. Never invent, recall, or adjust a value.\n"
                "- You have NO literature access. Never cite, name, list, or invent "
                "papers, authors, years, journals, or references. Literature is a "
                "different agent's job.\n"
                "- Do not compare to other molecules; describe only this one.\n"
                "- Start directly with the summary. No preamble, no restating the name "
                "as a heading, no quotation marks around the molecule name.\n\n"
                f"Molecule: {name}\n"
                f"Properties: {json.dumps(props)}"
            ),
        }],
        temperature=0.2,
    )
    text = present.choices[0].message.content.strip().lstrip("'\" ").strip()
    return AgentResponse(
        "molecular", text, ok=True,
        sources=[{"molecule": name, "properties": props, "molblock": molblock}],
    )


if __name__ == "__main__":
    for q in ["Is atorvastatin drug-like?", "properties of remdesivir", "logP of gefitinib"]:
        r = run(q)
        print(f"\nQ: {q}\n{r.text}")