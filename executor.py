"""
Executor: applies a correction operation to a wrong_smiles_mapped string,
returning the restored canonical SMILES.

The correction params use atom_map / anchor_map / atom1_map / atom2_map
(map_num integers that survive SMILES round-trips) instead of raw atom_idx.
"""

import copy
from rdkit import Chem
from mol_ops import (
    change_atom_element, add_atom, remove_atom, change_charge,
    change_bond_order, add_bond, remove_bond,
    add_functional_group, remove_functional_group,
    flip_chirality, flip_ez,
)


def _map_to_idx(mol):
    """Return {map_num: current_atom_idx} for all mapped atoms."""
    return {a.GetAtomMapNum(): a.GetIdx()
            for a in mol.GetAtoms() if a.GetAtomMapNum() > 0}


def _strip_maps(mol):
    rw = Chem.RWMol(copy.deepcopy(mol))
    for a in rw.GetAtoms():
        a.SetAtomMapNum(0)
    return rw


def _translate_params(params, m2i):
    """Convert map_num params to atom_idx params; drop description-only keys."""
    # Keys that are metadata only — not passed to op functions
    SKIP = {"from_elem", "from_charge", "from_order", "elem"}
    out = {}
    for k, v in params.items():
        if k in SKIP:
            continue
        if k == "atom_map":
            out["atom_idx"] = m2i[v]
        elif k == "anchor_map":
            out["anchor_idx"] = m2i[v]
        elif k == "atom1_map":
            out["atom_idx1"] = m2i[v]
        elif k == "atom2_map":
            out["atom_idx2"] = m2i[v]
        else:
            out[k] = v
    return out


def apply_correction(wrong_smiles_mapped, correction):
    """
    Apply a correction operation to wrong_smiles_mapped.

    Args:
        wrong_smiles_mapped: SMILES string with :[n] atom map tags
        correction: dict with keys 'type' and 'params' (map_num-based)

    Returns:
        Restored canonical SMILES string, or None if operation failed.
    """
    op_type = correction["type"]
    params  = correction.get("params", {})

    # flip_ez works at SMILES level — strip maps first
    if op_type == "flip_ez":
        mol = Chem.MolFromSmiles(wrong_smiles_mapped)
        if mol is None:
            return None
        rw = _strip_maps(mol)
        smi = Chem.MolToSmiles(rw, isomericSmiles=True)
        return flip_ez(smi)

    mol = Chem.MolFromSmiles(wrong_smiles_mapped)
    if mol is None:
        return None

    m2i = _map_to_idx(mol)
    try:
        idx_params = _translate_params(params, m2i)
    except KeyError as e:
        return None  # map_num not found in this molecule

    rw = Chem.RWMol(mol)

    if op_type == "remove_functional_group":
        result, _ = remove_functional_group(rw, **idx_params)
    elif op_type == "change_atom_element":
        result = change_atom_element(rw, **idx_params)
    elif op_type == "add_atom":
        result = add_atom(rw, **idx_params)
    elif op_type == "remove_atom":
        result = remove_atom(rw, **idx_params)
    elif op_type == "change_charge":
        result = change_charge(rw, **idx_params)
    elif op_type == "change_bond_order":
        result = change_bond_order(rw, **idx_params)
    elif op_type == "add_bond":
        result = add_bond(rw, **idx_params)
    elif op_type == "remove_bond":
        result = remove_bond(rw, **idx_params)
    elif op_type == "add_functional_group":
        result = add_functional_group(rw, **idx_params)
    elif op_type == "flip_chirality":
        result = flip_chirality(rw, **idx_params)
    else:
        return None

    if result is None:
        return None

    # Strip atom map numbers and return canonical SMILES
    result = _strip_maps(result)
    return Chem.MolToSmiles(result)
