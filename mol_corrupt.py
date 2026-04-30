"""
Corruption engine using atom map numbers for stable atom identification.

Each corrupt_* function returns (wrong_rw, raw_correction_with_wrong_rw_indices).
corrupt_molecule() assigns atom map nums to wrong_rw, translates indices → map_nums,
and returns (wrong_smiles, wrong_smiles_mapped, correction_with_maps).

wrong_smiles        : clean SMILES without maps  (for display / training input)
wrong_smiles_mapped : SMILES with :[n] map tags  (for executor to locate atoms)
correction          : operation params use atom_map/anchor_map/atom1_map/atom2_map
                      instead of raw atom_idx — these survive SMILES round-trips.
"""

import copy
import random
from rdkit import Chem
from rdkit.Chem import AllChem

from mol_ops import (
    BOND_TYPE_MAP, BOND_TYPE_NAME, ELEM_TO_NUM, NUM_TO_ELEM,
    FUNCTIONAL_GROUPS,
    change_atom_element, add_atom, remove_atom, change_charge,
    change_bond_order, add_bond, remove_bond,
    add_functional_group, remove_functional_group,
    flip_chirality, flip_ez,
    validate_mol,
)

CORRUPTION_TYPES = [
    "change_atom_element",
    "add_atom",
    "remove_atom",
    "change_charge",
    "change_bond_order",
    "add_bond",
    "remove_bond",
    "add_functional_group",
    "remove_functional_group",
    "flip_chirality",
    "flip_ez",
    "move_substituent",
    "swap_substituents",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_other_elem(current_elem):
    candidates = [e for e in ("C", "N", "O", "S") if e != current_elem]
    return random.choice(candidates)


def _has_free_valence(atom):
    try:
        return atom.GetNumImplicitHs() > 0
    except Exception:
        return False


def _assign_atom_maps(rw):
    """Assign map_num = idx+1 to every atom in rw (in-place)."""
    for atom in rw.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)


def _strip_atom_maps(rw):
    """Return a copy of rw with all atom maps removed."""
    c = copy.deepcopy(rw)
    for atom in c.GetAtoms():
        atom.SetAtomMapNum(0)
    return c


def _translate_correction(wrong_rw, raw):
    """
    Translate a raw correction (atom_idx → map_num) for stable cross-SMILES use.
    raw = {'type': ..., 'params': {atom_idx: X, ...}, 'description': ...}
    Returns new correction dict with map_num-based params.
    """
    params = {}
    for k, v in raw["params"].items():
        if k == "atom_idx":
            params["atom_map"] = wrong_rw.GetAtomWithIdx(v).GetAtomMapNum()
        elif k == "anchor_idx":
            params["anchor_map"] = wrong_rw.GetAtomWithIdx(v).GetAtomMapNum()
        elif k == "atom_idx1":
            params["atom1_map"] = wrong_rw.GetAtomWithIdx(v).GetAtomMapNum()
        elif k == "atom_idx2":
            params["atom2_map"] = wrong_rw.GetAtomWithIdx(v).GetAtomMapNum()
        elif k == "substituent_idx":
            params["substituent_map"] = wrong_rw.GetAtomWithIdx(v).GetAtomMapNum()
        elif k == "from_idx":
            params["from_map"] = wrong_rw.GetAtomWithIdx(v).GetAtomMapNum()
        elif k == "to_idx":
            params["to_map"] = wrong_rw.GetAtomWithIdx(v).GetAtomMapNum()
        elif k == "ring_idx1":
            params["ring1_map"] = wrong_rw.GetAtomWithIdx(v).GetAtomMapNum()
        elif k == "ring_idx2":
            params["ring2_map"] = wrong_rw.GetAtomWithIdx(v).GetAtomMapNum()
        else:
            params[k] = v
    return {"type": raw["type"], "params": params, "description": raw["description"]}


# ---------------------------------------------------------------------------
# Individual corruption functions
# Each returns (wrong_rw, raw_correction) or None
# raw_correction uses atom indices in wrong_rw space
# ---------------------------------------------------------------------------

def corrupt_change_atom_element(mol):
    rw = Chem.RWMol(mol)
    atoms = [a for a in rw.GetAtoms()
             if a.GetAtomicNum() > 1 and not a.GetIsAromatic()]
    if not atoms:
        return None
    atom = random.choice(atoms)
    idx = atom.GetIdx()
    old_elem = NUM_TO_ELEM.get(atom.GetAtomicNum(), "C")
    new_elem = _random_other_elem(old_elem)
    result = change_atom_element(rw, atom_idx=idx, to_elem=new_elem)
    if result is None:
        return None
    return result, {
        "type": "change_atom_element",
        "params": {"atom_idx": idx, "from_elem": new_elem, "to_elem": old_elem},
        "description": f"Change atom {idx} from {new_elem} back to {old_elem}",
    }


def corrupt_add_atom(mol):
    rw = Chem.RWMol(mol)
    atoms = [a for a in rw.GetAtoms() if _has_free_valence(a)]
    if not atoms:
        return None
    anchor = random.choice(atoms)
    new_elem = random.choice(["C", "N", "O"])
    result = add_atom(rw, anchor_idx=anchor.GetIdx(), new_elem=new_elem)
    if result is None:
        return None
    new_idx = result.GetNumAtoms() - 1
    return result, {
        "type": "remove_atom",
        "params": {"atom_idx": new_idx, "elem": new_elem},
        "description": f"Remove extra {new_elem} atom (added by corruption)",
    }


def corrupt_remove_atom(mol):
    rw = Chem.RWMol(mol)
    terminals = [a for a in rw.GetAtoms()
                 if a.GetDegree() == 1 and not a.IsInRing() and a.GetAtomicNum() > 1]
    if not terminals:
        return None
    atom = random.choice(terminals)
    idx = atom.GetIdx()
    elem = NUM_TO_ELEM.get(atom.GetAtomicNum(), "C")
    neighbor = atom.GetNeighbors()[0]
    anchor_idx = neighbor.GetIdx()
    bond = rw.GetBondBetweenAtoms(idx, anchor_idx)
    bond_type = BOND_TYPE_NAME.get(bond.GetBondType(), "SINGLE")

    result = remove_atom(rw, atom_idx=idx)
    if result is None:
        return None
    # Atoms after idx shift down by 1
    adj_anchor = anchor_idx if anchor_idx < idx else anchor_idx - 1
    return result, {
        "type": "add_atom",
        "params": {"anchor_idx": adj_anchor, "new_elem": elem, "bond_type": bond_type},
        "description": f"Add missing {elem} atom back to anchor",
    }


def corrupt_change_charge(mol):
    rw = Chem.RWMol(mol)
    # Only corrupt atoms that currently have charge 0 (reversible: 0→±1, then ±1→0 is safe)
    candidates = [a for a in rw.GetAtoms()
                  if a.GetAtomicNum() in (7, 8) and not a.GetIsAromatic()
                  and a.GetFormalCharge() == 0 and a.GetNumImplicitHs() == 0]
    if not candidates:
        return None
    atom = random.choice(candidates)
    idx = atom.GetIdx()
    old_charge = 0
    new_charge = random.choice([-1, 1])
    result = change_charge(rw, atom_idx=idx, to_charge=new_charge)
    if result is None:
        return None
    return result, {
        "type": "change_charge",
        "params": {"atom_idx": idx, "from_charge": new_charge, "to_charge": old_charge},
        "description": f"Change charge at atom back from {new_charge:+d} to {old_charge:+d}",
    }


def corrupt_change_bond_order(mol):
    rw = Chem.RWMol(mol)
    bonds = [b for b in rw.GetBonds()
             if b.GetBondType() in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE)
             and not b.GetIsAromatic()]
    if not bonds:
        return None
    bond = random.choice(bonds)
    idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    old_order = BOND_TYPE_NAME[bond.GetBondType()]
    new_order = "DOUBLE" if old_order == "SINGLE" else "SINGLE"
    result = change_bond_order(rw, atom_idx1=idx1, atom_idx2=idx2, to_order=new_order)
    if result is None:
        return None
    return result, {
        "type": "change_bond_order",
        "params": {"atom_idx1": idx1, "atom_idx2": idx2,
                   "from_order": new_order, "to_order": old_order},
        "description": f"Change bond ({idx1}-{idx2}) from {new_order} back to {old_order}",
    }


def corrupt_add_bond(mol):
    rw = Chem.RWMol(mol)
    n = rw.GetNumAtoms()
    if n < 4:
        return None
    pairs = [(i, j) for i in range(n) for j in range(i + 2, n)
             if rw.GetBondBetweenAtoms(i, j) is None]
    if not pairs:
        return None
    i, j = random.choice(pairs)
    result = add_bond(rw, atom_idx1=i, atom_idx2=j, bond_type="SINGLE")
    if result is None:
        return None
    return result, {
        "type": "remove_bond",
        "params": {"atom_idx1": i, "atom_idx2": j},
        "description": f"Remove spurious bond between atoms {i} and {j}",
    }


def corrupt_remove_bond(mol):
    rw = Chem.RWMol(mol)
    bonds = [b for b in rw.GetBonds() if not b.IsInRing()]
    if not bonds:
        return None
    bond = random.choice(bonds)
    idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    order = BOND_TYPE_NAME.get(bond.GetBondType(), "SINGLE")
    result = remove_bond(rw, atom_idx1=idx1, atom_idx2=idx2)
    if result is None:
        return None
    return result, {
        "type": "add_bond",
        "params": {"atom_idx1": idx1, "atom_idx2": idx2, "bond_type": order},
        "description": f"Add back {order} bond between atoms {idx1} and {idx2}",
    }


def corrupt_add_functional_group(mol):
    rw = Chem.RWMol(mol)
    atoms = [a for a in rw.GetAtoms() if a.GetAtomicNum() == 6]
    if not atoms:
        return None
    anchor = random.choice(atoms)
    group = random.choice(["OH", "NH2", "F", "Cl", "Br", "CH3"])
    result = add_functional_group(rw, anchor_idx=anchor.GetIdx(), group_name=group)
    if result is None:
        return None
    # anchor idx unchanged after adding group atoms at the end
    return result, {
        "type": "remove_functional_group",
        "params": {"anchor_idx": anchor.GetIdx(), "group_name": group},
        "description": f"Remove extra -{group} group",
    }


def corrupt_remove_functional_group(mol):
    rw = Chem.RWMol(mol)
    for group in ("OH", "NH2", "F", "Cl", "Br", "CH3"):
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            test = Chem.RWMol(mol)
            result, removed = remove_functional_group(test, atom.GetIdx(), group)
            if result is None:
                continue
            removed_idx = removed[0]
            anchor_in_mol = atom.GetIdx()
            # After removing atom at removed_idx, anchor shifts if removed_idx < anchor
            adj_anchor = anchor_in_mol if removed_idx > anchor_in_mol else anchor_in_mol - 1
            return result, {
                "type": "add_functional_group",
                "params": {"anchor_idx": adj_anchor, "group_name": group},
                "description": f"Add -{group} back to anchor atom",
            }
    return None


def corrupt_flip_chirality(mol):
    rw = Chem.RWMol(mol)
    Chem.AssignStereochemistry(rw, cleanIt=True, force=True)
    chiral = [a for a in rw.GetAtoms()
              if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED]
    if not chiral:
        return None
    atom = random.choice(chiral)
    idx = atom.GetIdx()
    result = flip_chirality(rw, atom_idx=idx)
    if result is None:
        return None
    return result, {
        "type": "flip_chirality",
        "params": {"atom_idx": idx},
        "description": f"Flip chirality back at atom {idx}",
    }


def corrupt_flip_ez(mol):
    """SMILES-level; returns (wrong_smi, raw_correction) directly."""
    smi = Chem.MolToSmiles(mol, isomericSmiles=True)
    if "/" not in smi and "\\" not in smi:
        return None
    wrong_smi = flip_ez(smi)
    if wrong_smi is None or wrong_smi == smi:
        return None
    return wrong_smi, {
        "type": "flip_ez",
        "params": {},
        "description": "Flip E/Z configuration back",
    }


def corrupt_move_substituent(mol):
    """
    Move a non-ring substituent from one ring atom to a different ring atom
    in the same ring (e.g. ortho→meta shift on benzene).
    Operates in Kekulé form to avoid aromatic bond issues.
    Correction: MoveSubstituent back.
    """
    rw = Chem.RWMol(mol)
    try:
        Chem.Kekulize(rw, clearAromaticFlags=True)
    except Exception:
        return None

    ri = rw.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings() if len(r) >= 5]
    if not rings:
        return None

    # Build all (from_idx, sub_idx, to_idx) triples and shuffle
    triples = []
    for ring in rings:
        ring_list = list(ring)
        for from_idx in ring_list:
            atom = rw.GetAtomWithIdx(from_idx)
            subs = [nb for nb in atom.GetNeighbors()
                    if nb.GetIdx() not in ring and nb.GetAtomicNum() > 1]
            for sub in subs:
                sub_idx = sub.GetIdx()
                for to_idx in ring_list:
                    if to_idx != from_idx and rw.GetBondBetweenAtoms(to_idx, sub_idx) is None:
                        triples.append((from_idx, sub_idx, to_idx))
    random.shuffle(triples)

    for from_idx, sub_idx, to_idx in triples:
        rw2 = Chem.RWMol(copy.deepcopy(rw))
        rw2.RemoveBond(from_idx, sub_idx)
        rw2.AddBond(to_idx, sub_idx, Chem.BondType.SINGLE)
        if not validate_mol(rw2):
            continue
        # Check it actually changed the molecule
        wrong_smi = Chem.MolToSmiles(rw2)
        orig_smi = Chem.MolToSmiles(mol)
        if wrong_smi == orig_smi:
            continue
        return rw2, {
            "type": "move_substituent",
            "params": {
                "substituent_idx": sub_idx,
                "from_idx": to_idx,    # correction reverses: from=to, to=from
                "to_idx":   from_idx,
            },
            "description": f"Move substituent {sub_idx} back from ring atom {to_idx} to {from_idx}",
        }
    return None


def corrupt_swap_substituents(mol):
    """
    Swap the non-ring substituents of two ring atoms in the same ring.
    Operates in Kekulé form to avoid aromatic bond issues.
    Correction: SwapSubstituents back (same action).
    """
    rw = Chem.RWMol(mol)
    try:
        Chem.Kekulize(rw, clearAromaticFlags=True)
    except Exception:
        return None

    ri = rw.GetRingInfo()
    rings = [list(r) for r in ri.AtomRings() if len(r) >= 5]
    if not rings:
        return None

    random.shuffle(rings)
    for ring in rings:
        ring_set = set(ring)
        substituted = []
        for idx in ring:
            subs = [nb.GetIdx() for nb in rw.GetAtomWithIdx(idx).GetNeighbors()
                    if nb.GetIdx() not in ring_set and nb.GetAtomicNum() > 1]
            if subs:
                substituted.append((idx, subs))
        if len(substituted) < 2:
            continue

        random.shuffle(substituted)
        (idx1, subs1), (idx2, subs2) = substituted[0], substituted[1]

        def bonds_to(center, sub_list):
            return [(s, rw.GetBondBetweenAtoms(center, s).GetBondType()) for s in sub_list]

        b1 = bonds_to(idx1, subs1)
        b2 = bonds_to(idx2, subs2)

        if any(rw.GetBondBetweenAtoms(idx2, s) is not None for s, _ in b1):
            continue
        if any(rw.GetBondBetweenAtoms(idx1, s) is not None for s, _ in b2):
            continue

        rw2 = Chem.RWMol(copy.deepcopy(rw))
        for s, _ in b1:
            rw2.RemoveBond(idx1, s)
        for s, _ in b2:
            rw2.RemoveBond(idx2, s)
        for s, bt in b1:
            rw2.AddBond(idx2, s, Chem.BondType.SINGLE)
        for s, bt in b2:
            rw2.AddBond(idx1, s, Chem.BondType.SINGLE)

        if not validate_mol(rw2):
            continue
        return rw2, {
            "type": "swap_substituents",
            "params": {
                "ring_idx1": idx1,
                "ring_idx2": idx2,
            },
            "description": f"Swap substituents back between ring atoms {idx1} and {idx2}",
        }
    return None


CORRUPT_FUNCS = {
    "change_atom_element":     corrupt_change_atom_element,
    "add_atom":                corrupt_add_atom,
    "remove_atom":             corrupt_remove_atom,
    "change_charge":           corrupt_change_charge,
    "change_bond_order":       corrupt_change_bond_order,
    "add_bond":                corrupt_add_bond,
    "remove_bond":             corrupt_remove_bond,
    "add_functional_group":    corrupt_add_functional_group,
    "remove_functional_group": corrupt_remove_functional_group,
    "flip_chirality":          corrupt_flip_chirality,
    "flip_ez":                 corrupt_flip_ez,
    "move_substituent":        corrupt_move_substituent,
    "swap_substituents":       corrupt_swap_substituents,
}


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def corrupt_molecule(smiles, op_type=None, max_retries=20):
    """
    Returns (wrong_smiles, wrong_smiles_mapped, correction) or None.

    wrong_smiles        — clean SMILES (no atom maps), the erroneous input
    wrong_smiles_mapped — SMILES with :[n] atom map tags for executor use
    correction          — dict with type, params (map_num-based), description
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    for _ in range(max_retries):
        chosen_type = op_type or random.choice(CORRUPTION_TYPES)
        func = CORRUPT_FUNCS[chosen_type]
        raw = func(mol)
        if raw is None:
            continue

        # flip_ez is SMILES-level — handle separately
        if chosen_type == "flip_ez":
            wrong_smi, raw_correction = raw
            if wrong_smi == smiles:
                continue
            wrong_mol = Chem.MolFromSmiles(wrong_smi)
            if wrong_mol is None:
                continue
            wrong_rw = Chem.RWMol(wrong_mol)
            _assign_atom_maps(wrong_rw)
            wrong_smiles_mapped = Chem.MolToSmiles(wrong_rw)
            correction = {"type": "flip_ez", "params": {},
                          "description": raw_correction["description"]}
            return wrong_smi, wrong_smiles_mapped, correction

        wrong_rw, raw_correction = raw
        if not validate_mol(wrong_rw):
            continue

        # Assign atom map numbers to wrong_rw (map_num = idx + 1)
        _assign_atom_maps(wrong_rw)

        # Translate index-based params → map-num-based params
        try:
            correction = _translate_correction(wrong_rw, raw_correction)
        except (IndexError, RuntimeError):
            continue

        # Generate clean wrong SMILES (no maps)
        wrong_smiles = Chem.MolToSmiles(_strip_atom_maps(wrong_rw))
        if wrong_smiles == smiles:
            continue

        wrong_smiles_mapped = Chem.MolToSmiles(wrong_rw)

        # ── Verify round-trip before accepting ──────────────────────────────
        from action_executor import ActionExecutor as _AE
        from validate_with_executor import translate_operation
        try:
            action = translate_operation(correction)
            restored = _AE().execute(wrong_smiles_mapped, action)
        except Exception:
            continue
        restored_mol = Chem.MolFromSmiles(restored) if restored else None
        if restored_mol is None or Chem.MolToSmiles(restored_mol) != smiles:
            continue  # bad sample — discard and retry

        return wrong_smiles, wrong_smiles_mapped, correction

    return None
