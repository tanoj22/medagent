"""Molecular small-molecule agent: compute drug-discovery properties via RDKit."""
import json
import os

from dotenv import load_dotenv
from groq import Groq
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski

from src.agents.base import AgentResponse

load_dotenv()
_client = Groq(api_key=os.environ["GROQ_API_KEY"])
EXTRACT_MODEL = "llama-3.1-8b-instant"      # cheap model for extraction
PRESENT_MODEL = "llama-3.3-70b-versatile"   # stronger model for phrasing

# Curated name -> SMILES for common molecules. Any valid SMILES also works directly.
KNOWN_MOLECULES = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "caffeine": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "acetaminophen": "CC(=O)Nc1ccc(O)cc1",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "metformin": "CN(C)C(=N)N=C(N)N",
    "warfarin": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
    "diazepam": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
    "dopamine": "NCCc1ccc(O)c(O)c1",
    "serotonin": "NCCc1c[nH]c2ccc(O)cc12",
    "nicotine": "CN1CCCC1c1cccnc1",
    "glucose": "OCC1OC(O)C(O)C(O)C1O",
    "ethanol": "CCO",
    "benzene": "c1ccccc1",
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


def _extract_molecule(query: str) -> str | None:
    """Cheap LLM call to pull the molecule name or SMILES out of the question."""
    prompt = (
        "Identify the chemical molecule the user asks about. Respond with ONLY "
        "the molecule's common name (e.g. aspirin), or the exact SMILES if the "
        "user provided one. Do NOT invent a SMILES. If there is no molecule, "
        "respond exactly NONE.\n\n"
        f"User: {query}"
    )
    resp = _client.chat.completions.create(
        model=EXTRACT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer = resp.choices[0].message.content.strip().strip('."\'')
    return None if answer.upper() == "NONE" else answer


def _resolve_to_smiles(ref: str):
    key = ref.lower().strip()
    if key in KNOWN_MOLECULES:
        return key, KNOWN_MOLECULES[key]
    if Chem.MolFromSmiles(ref) is not None:
        return ref, ref
    return None


def run(query: str) -> AgentResponse:
    """Agent entry point: question -> computed properties + natural-language answer."""
    ref = _extract_molecule(query)
    if ref is None:
        return AgentResponse("molecular", "No molecule identified in the question.", ok=False)

    resolved = _resolve_to_smiles(ref)
    if resolved is None:
        return AgentResponse("molecular", f"'{ref}' isn't in my known set — provide its SMILES.", ok=False)

    name, smiles = resolved
    props = compute_properties(smiles)
    if props is None:
        return AgentResponse("molecular", f"Could not parse the structure for '{name}'.", ok=False)

    present = _client.chat.completions.create(
        model=PRESENT_MODEL,
        messages=[{
            "role": "user",
            "content": (
                "You are a cheminformatics assistant. Report these RDKit-computed "
                "properties clearly for a researcher: give the actual values for "
                "molecular weight, logP, H-bond donors, H-bond acceptors, TPSA, and "
                "rotatable bonds, then state the Lipinski Rule of Five drug-likeness "
                "verdict and briefly why. Use ONLY these numbers; add no outside facts.\n\n"
                f"Molecule: {name}\nProperties: {json.dumps(props)}"
            ),
        }],
        temperature=0.2,
    )
    return AgentResponse("molecular", present.choices[0].message.content, ok=True,
                         sources=[{"molecule": name, "properties": props}])


if __name__ == "__main__":
    for q in [
        "What are the molecular properties of aspirin?",
        "Is ibuprofen drug-like?",
        "Tell me about the molecule CC(=O)Nc1ccc(O)cc1",
        "What's the capital of France?",
    ]:
        print(f"\nQ: {q}")
        print(run(q).text)