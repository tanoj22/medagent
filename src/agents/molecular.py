"""Molecular small-molecule agent: compute drug-discovery properties via RDKit."""
import json
import os

from dotenv import load_dotenv
from groq import Groq
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski

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


def _mol_block_3d(smiles: str) -> str | None:
    """Generate 3D atom coordinates and return them as a MOL block (text the viewer can read)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)                          # add hydrogens for a realistic 3D shape
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
        return None                                # embedding failed
    try:
        AllChem.MMFFOptimizeMolecule(mol)          # relax the geometry
    except Exception:
        pass                                       # optimization is best-effort
    return Chem.MolToMolBlock(mol)


def _extract_molecule(query: str) -> str | None:
    """Cheap LLM call to pull the molecule name or SMILES out of the question."""
    prompt = (
        "Extract the single chemical the user is asking about and output ONLY its "
        "name as one or two words (e.g. aspirin), or the exact SMILES the user gave. "
        "No explanation, no formula, no other words. If there is no chemical, output "
        "exactly NONE.\n\n"
        f"User: {query}\n"
        "Chemical:"
    )
    resp = _client.chat.completions.create(
        model=EXTRACT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=20,          # can't ramble into a full answer
    )
    answer = resp.choices[0].message.content.strip().strip('."\'').split("\n")[0].strip()
    return None if not answer or answer.upper() == "NONE" else answer


def _resolve_to_smiles(ref: str):
    key = ref.lower().strip()
    if key in KNOWN_MOLECULES:
        return key, KNOWN_MOLECULES[key]
    if Chem.MolFromSmiles(ref) is not None:
        return ref, ref
    return None


def run(query: str) -> AgentResponse:
    """Agent entry point: question -> RDKit-computed properties + 3D coords + grounded summary."""
    ref = _extract_molecule(query)
    if ref is None:
        return AgentResponse("molecular", "No molecule identified in the question.", ok=False)

    resolved = _resolve_to_smiles(ref)
    if resolved is None:
        return AgentResponse("molecular", f"'{ref}' isn't in my known set; provide its SMILES.", ok=False)

    name, smiles = resolved
    props = compute_properties(smiles)
    if props is None:
        return AgentResponse("molecular", f"Could not parse the structure for '{name}'.", ok=False)

    molblock = _mol_block_3d(smiles)               # 3D coordinates for the viewer (may be None)

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
    r = run("Is aspirin drug-like?")
    print(r.text)
    mb = r.sources[0].get("molblock") if r.sources else None
    print("\n3D MOL block generated:", "yes" if mb else "no", f"({len(mb)} chars)" if mb else "")