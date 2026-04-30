"""
Core molecular editing operations using RDKit's RWMol API.

Each operation takes an RWMol and parameters, returns the modified RWMol.
Each has a forward (corruption) and inverse (correction) form.
"""

import copy
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops


BOND_TYPE_MAP = {
    "SINGLE": Chem.BondType.SINGLE,
    "DOUBLE": Chem.BondType.DOUBLE,
    "TRIPLE": Chem.BondType.TRIPLE,
    "AROMATIC": Chem.BondType.AROMATIC,
}

BOND_TYPE_NAME = {v: k for k, v in BOND_TYPE_MAP.items()}

ELEM_TO_NUM = {
    "C": 6, "N": 7, "O": 8, "S": 16, "P": 15, "F": 9,
    "Cl": 17, "Br": 35, "I": 53, "B": 5, "Si": 14, "Se": 34,
}

NUM_TO_ELEM = {v: k for k, v in ELEM_TO_NUM.items()}

FUNCTIONAL_GROUPS = {
    "OH":   "[OH]",
    "NH2":  "[NH2]",
    "F":    "[F]",
    "Cl":   "[Cl]",
    "Br":   "[Br]",
    "CH3":  "[CH3]",
    "NO2":  "[N+](=O)[O-]",
    "COOH": "C(=O)[OH]",
    "CHO":  "[CH]=O",
}


def validate_mol(mol):
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        return False


def safe_apply(func):
    """Decorator: returns None if the operation produces an invalid molecule."""
    def wrapper(mol, **kwargs):
        rw = copy.deepcopy(mol)
        try:
            result = func(rw, **kwargs)
            if result is None:
                return None
            if not validate_mol(result):
                return None
            return result
        except Exception:
            return None
    return wrapper


# ---------------------------------------------------------------------------
# 1. change_atom_element
# ---------------------------------------------------------------------------
@safe_apply
def change_atom_element(mol, atom_idx, to_elem):
    atom = mol.GetAtomWithIdx(atom_idx)
    atom.SetAtomicNum(ELEM_TO_NUM[to_elem])
    atom.SetNoImplicit(False)
    atom.UpdatePropertyCache(strict=False)
    return mol


# ---------------------------------------------------------------------------
# 2. add_atom — adds a new atom bonded to anchor
# ---------------------------------------------------------------------------
@safe_apply
def add_atom(mol, anchor_idx, new_elem, bond_type="SINGLE"):
    new_idx = mol.AddAtom(Chem.Atom(ELEM_TO_NUM[new_elem]))
    mol.AddBond(anchor_idx, new_idx, BOND_TYPE_MAP[bond_type])
    return mol


# ---------------------------------------------------------------------------
# 3. remove_atom
# ---------------------------------------------------------------------------
@safe_apply
def remove_atom(mol, atom_idx):
    mol.RemoveAtom(atom_idx)
    return mol


# ---------------------------------------------------------------------------
# 4. change_charge
# ---------------------------------------------------------------------------
@safe_apply
def change_charge(mol, atom_idx, to_charge):
    atom = mol.GetAtomWithIdx(atom_idx)
    atom.SetFormalCharge(to_charge)
    atom.UpdatePropertyCache(strict=False)
    return mol


# ---------------------------------------------------------------------------
# 5. change_bond_order
# ---------------------------------------------------------------------------
@safe_apply
def change_bond_order(mol, atom_idx1, atom_idx2, to_order):
    bond = mol.GetBondBetweenAtoms(atom_idx1, atom_idx2)
    if bond is None:
        return None
    bond.SetBondType(BOND_TYPE_MAP[to_order])
    return mol


# ---------------------------------------------------------------------------
# 6. add_bond
# ---------------------------------------------------------------------------
@safe_apply
def add_bond(mol, atom_idx1, atom_idx2, bond_type="SINGLE"):
    existing = mol.GetBondBetweenAtoms(atom_idx1, atom_idx2)
    if existing is not None:
        return None
    mol.AddBond(atom_idx1, atom_idx2, BOND_TYPE_MAP[bond_type])
    return mol


# ---------------------------------------------------------------------------
# 7. remove_bond
# ---------------------------------------------------------------------------
@safe_apply
def remove_bond(mol, atom_idx1, atom_idx2):
    bond = mol.GetBondBetweenAtoms(atom_idx1, atom_idx2)
    if bond is None:
        return None
    mol.RemoveBond(atom_idx1, atom_idx2)
    return mol


# ---------------------------------------------------------------------------
# 8. add_functional_group
# ---------------------------------------------------------------------------
@safe_apply
def add_functional_group(mol, anchor_idx, group_name):
    group_smi = FUNCTIONAL_GROUPS.get(group_name)
    if group_smi is None:
        return None
    group_mol = Chem.MolFromSmiles(group_smi)
    if group_mol is None:
        return None

    combo = Chem.RWMol(rdmolops.CombineMols(mol, group_mol))
    old_n = mol.GetNumAtoms()
    anchor = combo.GetAtomWithIdx(anchor_idx)
    anchor.SetNumExplicitHs(0)
    anchor.SetNoImplicit(False)
    anchor.SetNumRadicalElectrons(0)
    combo.AddBond(anchor_idx, old_n, Chem.BondType.SINGLE)
    return combo


# ---------------------------------------------------------------------------
# 9. remove_functional_group (removes a terminal substituent at anchor)
# ---------------------------------------------------------------------------
def remove_functional_group(mol, anchor_idx, group_name):
    """Remove atoms belonging to a terminal functional group attached at anchor_idx.
    Returns (new_mol, removed_atom_indices) or (None, [])."""
    rw = copy.deepcopy(mol)
    anchor = rw.GetAtomWithIdx(anchor_idx)
    neighbors = [n for n in anchor.GetNeighbors()]

    if group_name in ("F", "Cl", "Br"):
        target_num = ELEM_TO_NUM.get(group_name)
        for n in neighbors:
            if n.GetAtomicNum() == target_num and n.GetDegree() == 1:
                idx = n.GetIdx()
                rw.RemoveAtom(idx)
                try:
                    Chem.SanitizeMol(rw)
                    return rw, [idx]
                except Exception:
                    return None, []
        return None, []

    if group_name == "OH":
        for n in neighbors:
            if n.GetAtomicNum() == 8 and n.GetDegree() == 1:
                bond = rw.GetBondBetweenAtoms(anchor_idx, n.GetIdx())
                if bond and bond.GetBondType() == Chem.BondType.SINGLE:
                    idx = n.GetIdx()
                    rw.RemoveAtom(idx)
                    try:
                        Chem.SanitizeMol(rw)
                        return rw, [idx]
                    except Exception:
                        return None, []
        return None, []

    if group_name == "NH2":
        for n in neighbors:
            if n.GetAtomicNum() == 7 and n.GetDegree() == 1:
                idx = n.GetIdx()
                rw.RemoveAtom(idx)
                try:
                    Chem.SanitizeMol(rw)
                    return rw, [idx]
                except Exception:
                    return None, []
        return None, []

    if group_name == "CH3":
        for n in neighbors:
            if n.GetAtomicNum() == 6 and n.GetDegree() == 1 and n.GetTotalNumHs() == 3:
                idx = n.GetIdx()
                rw.RemoveAtom(idx)
                try:
                    Chem.SanitizeMol(rw)
                    return rw, [idx]
                except Exception:
                    return None, []
        return None, []

    return None, []


# ---------------------------------------------------------------------------
# 10. flip_chirality
# ---------------------------------------------------------------------------
@safe_apply
def flip_chirality(mol, atom_idx):
    atom = mol.GetAtomWithIdx(atom_idx)
    chi = atom.GetChiralTag()
    if chi == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
        atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    elif chi == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
        atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    else:
        return None
    return mol


# ---------------------------------------------------------------------------
# 11. flip_ez — flip E/Z via SMILES-level slash inversion
# ---------------------------------------------------------------------------
def flip_ez(smiles):
    """Flip E/Z by inverting the first /-slash in the SMILES.
    Works at the SMILES string level since RDKit mol-level stereo
    doesn't round-trip reliably through MolToSmiles."""
    if "/" not in smiles and "\\" not in smiles:
        return None
    chars = list(smiles)
    for i, c in enumerate(chars):
        if c == "/":
            chars[i] = "\\"
            break
        elif c == "\\":
            chars[i] = "/"
            break
    new_smi = "".join(chars)
    mol = Chem.MolFromSmiles(new_smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


# ---------------------------------------------------------------------------
# Registry: maps operation type name → apply function
# ---------------------------------------------------------------------------
OP_REGISTRY = {
    "change_atom_element": change_atom_element,
    "add_atom":            add_atom,
    "remove_atom":         remove_atom,
    "change_charge":       change_charge,
    "change_bond_order":   change_bond_order,
    "add_bond":            add_bond,
    "remove_bond":         remove_bond,
    "add_functional_group":    add_functional_group,
    "remove_functional_group": lambda mol, **kw: remove_functional_group(mol, **kw)[0],
    "flip_chirality":      flip_chirality,
    "flip_ez":             flip_ez,
}
