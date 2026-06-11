"""Molecular property computation via RDKit (the small-molecule tool)."""
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski


def compute_properties(smiles: str) -> dict | None:
    """Compute drug-discovery properties for a molecule given its SMILES.
    Returns None if the SMILES can't be parsed."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)

    # Lipinski's Rule of Five: a drug-like molecule has at most 1 violation.
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


if __name__ == "__main__":
    tests = {
        "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
        "caffeine": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
        "acetaminophen": "CC(=O)Nc1ccc(O)cc1",
    }
    for name, smiles in tests.items():
        p = compute_properties(smiles)
        print(f"\n{name}  ({p['canonical_smiles']})")
        print(f"  MW={p['molecular_weight']}  logP={p['logp']}  "
              f"HBD={p['h_bond_donors']}  HBA={p['h_bond_acceptors']}  "
              f"TPSA={p['tpsa']}  rot={p['rotatable_bonds']}")
        print(f"  Lipinski violations={p['lipinski_violations']}  "
              f"drug-like={p['drug_like']}")