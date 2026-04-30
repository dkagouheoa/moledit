"""
ActionExecutor — general-purpose molecular editing via a clean JSON Action protocol.

Design:
  - Graph-level: ChangeAtom, AddAtom, RemoveAtom, ChangeBond, AddBond, RemoveBond
  - Shorthand:   AddGroup, RemoveGroup
  - Stereo:      FlipChirality, FlipEZ
  - Semantic:    MoveSubstituent, SwapSubstituents
  - Composite:   Batch

AtomRef (how to select an atom in the input SMILES):
    {"map_num": 5}                        atom map number  (preferred)
    {"smarts":  "[NH2]", "match_idx": 0}  SMARTS pattern   (most flexible)
    {"idx":     3}                        raw atom index   (fragile)

BondRef:
    {"atom1": <AtomRef>, "atom2": <AtomRef>}

Usage:
    ex = ActionExecutor()
    new_smiles = ex.execute("CCO", {"type": "ChangeAtom", "atom": {"idx": 2}, "element": "N"})
    # → "CCN"
"""

import copy
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from mol_ops import (
    BOND_TYPE_MAP, ELEM_TO_NUM,
    add_functional_group, remove_functional_group, flip_ez,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ActionError(Exception):
    """Base class for all executor errors."""

class AtomNotFoundError(ActionError):
    """Raised when an AtomRef cannot be resolved."""

class BondNotFoundError(ActionError):
    """Raised when a BondRef cannot be resolved."""

class InvalidMolError(ActionError):
    """Raised when an operation produces an invalid molecule."""

class UnknownActionError(ActionError):
    """Raised for unsupported action types."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOND_STEREO_MAP = {
    "E":   Chem.BondStereo.STEREOE,
    "Z":   Chem.BondStereo.STEREOZ,
    "CIS": Chem.BondStereo.STEREOZ,
    "TRANS": Chem.BondStereo.STEREOE,
    "NONE": Chem.BondStereo.STEREONONE,
}


def _sanitize(rw, context=""):
    try:
        Chem.SanitizeMol(rw)
    except Exception as e:
        raise InvalidMolError(f"Sanitization failed{' (' + context + ')' if context else ''}: {e}")


def _reset_atom_hs(atom):
    """Clear explicit Hs and radical state so SanitizeMol recomputes implicit Hs."""
    atom.SetNumExplicitHs(0)
    atom.SetNoImplicit(False)
    atom.SetNumRadicalElectrons(0)


def _kekulize_rw(rw: Chem.RWMol) -> Chem.RWMol:
    """Return a copy with aromatic bonds replaced by explicit single/double (Kekulé form)."""
    rw2 = Chem.RWMol(copy.deepcopy(rw))
    Chem.Kekulize(rw2, clearAromaticFlags=True)
    return rw2


def _to_canonical(mol):
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def _shared_rings(mol, idx1, idx2):
    """Return set of ring-atom-index sets that contain both idx1 and idx2."""
    ri = mol.GetRingInfo()
    return [r for r in ri.AtomRings() if idx1 in r and idx2 in r]


# ---------------------------------------------------------------------------
# ActionExecutor
# ---------------------------------------------------------------------------

class ActionExecutor:
    """
    Apply a JSON Action to a SMILES string and return the modified SMILES.

    Public API:
        execute(smiles, action)          → str
        execute_batch(smiles, actions)   → str
    """

    # ── Atom / Bond resolution ───────────────────────────────────────────────

    def _resolve_atom(self, mol, ref: dict) -> int:
        """Resolve an AtomRef dict to a concrete atom index in mol."""
        if "map_num" in ref:
            mn = ref["map_num"]
            for a in mol.GetAtoms():
                if a.GetAtomMapNum() == mn:
                    return a.GetIdx()
            raise AtomNotFoundError(f"No atom with map_num={mn}")

        if "smarts" in ref:
            patt = Chem.MolFromSmarts(ref["smarts"])
            if patt is None:
                raise AtomNotFoundError(f"Invalid SMARTS: {ref['smarts']!r}")
            matches = mol.GetSubstructMatches(patt)
            mi = ref.get("match_idx", 0)
            if not matches or mi >= len(matches):
                raise AtomNotFoundError(
                    f"SMARTS {ref['smarts']!r} match_idx={mi}: "
                    f"found {len(matches)} match(es)"
                )
            return matches[mi][0]

        if "idx" in ref:
            idx = ref["idx"]
            if idx < 0 or idx >= mol.GetNumAtoms():
                raise AtomNotFoundError(f"idx={idx} out of range (mol has {mol.GetNumAtoms()} atoms)")
            return idx

        raise AtomNotFoundError(f"AtomRef must have 'map_num', 'smarts', or 'idx': got {ref}")

    def _resolve_bond(self, mol, bond_ref: dict):
        """Resolve a BondRef to (idx1, idx2, Bond)."""
        i1 = self._resolve_atom(mol, bond_ref["atom1"])
        i2 = self._resolve_atom(mol, bond_ref["atom2"])
        bond = mol.GetBondBetweenAtoms(i1, i2)
        if bond is None:
            raise BondNotFoundError(f"No bond between atom idx {i1} and {i2}")
        return i1, i2, bond

    # ── Action handlers ──────────────────────────────────────────────────────

    def _change_atom(self, rw: Chem.RWMol, action: dict):
        idx = self._resolve_atom(rw, action["atom"])
        atom = rw.GetAtomWithIdx(idx)

        if "element" in action:
            n = ELEM_TO_NUM.get(action["element"])
            if n is None:
                raise ActionError(f"Unknown element: {action['element']!r}")
            atom.SetAtomicNum(n)
            atom.SetNoImplicit(False)

        if "charge" in action:
            atom.SetFormalCharge(int(action["charge"]))
            _reset_atom_hs(atom)

        if "isotope" in action and action["isotope"] is not None:
            atom.SetIsotope(int(action["isotope"]))

        if "num_hs" in action and action["num_hs"] is not None:
            atom.SetNumExplicitHs(int(action["num_hs"]))
            atom.SetNoImplicit(True)

        atom.UpdatePropertyCache(strict=False)
        _sanitize(rw, "ChangeAtom")
        return rw

    def _add_atom(self, rw: Chem.RWMol, action: dict):
        anchor_idx = self._resolve_atom(rw, action["anchor"])
        elem = action["element"]
        n = ELEM_TO_NUM.get(elem)
        if n is None:
            raise ActionError(f"Unknown element: {elem!r}")

        order = action.get("bond_order", "SINGLE").upper()
        bond_type = BOND_TYPE_MAP.get(order)
        if bond_type is None:
            raise ActionError(f"Unknown bond order: {order!r}")

        new_idx = rw.AddAtom(Chem.Atom(n))
        new_atom = rw.GetAtomWithIdx(new_idx)

        if "charge" in action:
            new_atom.SetFormalCharge(int(action["charge"]))
        if "num_hs" in action and action["num_hs"] is not None:
            new_atom.SetNumExplicitHs(int(action["num_hs"]))

        _reset_atom_hs(rw.GetAtomWithIdx(anchor_idx))

        rw.AddBond(anchor_idx, new_idx, bond_type)
        _sanitize(rw, "AddAtom")
        return rw

    def _remove_atom(self, rw: Chem.RWMol, action: dict):
        idx = self._resolve_atom(rw, action["atom"])
        neighbors = [n.GetIdx() for n in rw.GetAtomWithIdx(idx).GetNeighbors()]
        rw.RemoveAtom(idx)
        for nidx in neighbors:
            adj = nidx if nidx < idx else nidx - 1
            if adj < rw.GetNumAtoms():
                _reset_atom_hs(rw.GetAtomWithIdx(adj))
        _sanitize(rw, "RemoveAtom")
        return rw

    def _change_bond(self, rw: Chem.RWMol, action: dict):
        i1, i2, bond = self._resolve_bond(rw, action["bond"])
        order = action["order"].upper()
        bond_type = BOND_TYPE_MAP.get(order)
        if bond_type is None:
            raise ActionError(f"Unknown bond order: {order!r}")
        bond.SetBondType(bond_type)

        if "stereo" in action and action["stereo"]:
            stereo = BOND_STEREO_MAP.get(action["stereo"].upper())
            if stereo is not None:
                bond.SetStereo(stereo)

        for idx in (i1, i2):
            _reset_atom_hs(rw.GetAtomWithIdx(idx))
        _sanitize(rw, "ChangeBond")
        return rw

    def _add_bond(self, rw: Chem.RWMol, action: dict):
        i1 = self._resolve_atom(rw, action["bond"]["atom1"])
        i2 = self._resolve_atom(rw, action["bond"]["atom2"])
        if rw.GetBondBetweenAtoms(i1, i2) is not None:
            raise ActionError(f"Bond already exists between atoms {i1} and {i2}")
        order = action.get("order", "SINGLE").upper()
        bond_type = BOND_TYPE_MAP.get(order)
        if bond_type is None:
            raise ActionError(f"Unknown bond order: {order!r}")
        rw.AddBond(i1, i2, bond_type)
        for idx in (i1, i2):
            _reset_atom_hs(rw.GetAtomWithIdx(idx))
        _sanitize(rw, "AddBond")
        return rw

    def _remove_bond(self, rw: Chem.RWMol, action: dict):
        i1, i2, _ = self._resolve_bond(rw, action["bond"])
        rw.RemoveBond(i1, i2)
        for idx in (i1, i2):
            _reset_atom_hs(rw.GetAtomWithIdx(idx))
        _sanitize(rw, "RemoveBond")
        return rw

    def _add_group(self, rw: Chem.RWMol, action: dict):
        anchor = self._resolve_atom(rw, action["anchor"])
        group  = action["group"]
        result = add_functional_group(rw, anchor_idx=anchor, group_name=group)
        if result is None:
            raise InvalidMolError(f"AddGroup failed: group={group!r} anchor={anchor}")
        return Chem.RWMol(result)

    def _remove_group(self, rw: Chem.RWMol, action: dict):
        anchor = self._resolve_atom(rw, action["anchor"])
        group  = action["group"]
        result, removed = remove_functional_group(rw, anchor_idx=anchor, group_name=group)
        if result is None:
            raise InvalidMolError(
                f"RemoveGroup failed: group={group!r} not found at anchor={anchor}"
            )
        rw2 = Chem.RWMol(result)
        _reset_atom_hs(rw2.GetAtomWithIdx(anchor))
        _sanitize(rw2, "RemoveGroup")
        return rw2

    def _flip_chirality(self, rw: Chem.RWMol, action: dict):
        idx = self._resolve_atom(rw, action["atom"])
        atom = rw.GetAtomWithIdx(idx)
        chi = atom.GetChiralTag()
        if chi == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        elif chi == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
        else:
            raise ActionError(f"Atom idx={idx} has no specified chirality to flip")
        _sanitize(rw, "FlipChirality")
        return rw

    def _flip_ez(self, smiles: str, action: dict) -> str:
        """
        FlipEZ: invert E/Z at SMILES level.
        If 'bond' is provided, validates it's a double bond first.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise InvalidMolError(f"Cannot parse SMILES: {smiles!r}")

        if "bond" in action:
            i1, i2, bond = self._resolve_bond(mol, action["bond"])
            if bond.GetBondType() != Chem.BondType.DOUBLE:
                raise ActionError(
                    f"FlipEZ: bond between atoms {i1} and {i2} is not a double bond"
                )

        smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        if "/" not in smi and "\\" not in smi:
            raise ActionError("FlipEZ: SMILES has no E/Z notation to flip")

        result = flip_ez(smi)
        if result is None or result == smi:
            raise ActionError("FlipEZ: flip produced no change or invalid molecule")
        # Strip atom map numbers and return canonical form
        mol_r = Chem.MolFromSmiles(result)
        if mol_r is None:
            raise InvalidMolError(f"FlipEZ produced invalid SMILES: {result!r}")
        return _to_canonical(mol_r)

    def _move_substituent(self, rw: Chem.RWMol, action: dict):
        """
        Detach `substituent` from `from_atom` and reattach it to `to_atom`.
        Works on aromatic rings by operating in Kekulé form.
        """
        sub_idx  = self._resolve_atom(rw, action["substituent"])
        from_idx = self._resolve_atom(rw, action["from_atom"])
        to_idx   = self._resolve_atom(rw, action["to_atom"])

        bond = rw.GetBondBetweenAtoms(from_idx, sub_idx)
        if bond is None:
            raise BondNotFoundError(
                f"MoveSubstituent: no bond between from_atom={from_idx} and substituent={sub_idx}"
            )
        bond_type = bond.GetBondType()
        # Use SINGLE for substituent attachment (aromatic bond → substituent is always single)
        if bond_type == Chem.BondType.AROMATIC:
            bond_type = Chem.BondType.SINGLE

        if rw.GetBondBetweenAtoms(to_idx, sub_idx) is not None:
            raise ActionError(
                f"MoveSubstituent: bond already exists between to_atom={to_idx} and substituent={sub_idx}"
            )

        rw = _kekulize_rw(rw)
        rw.RemoveBond(from_idx, sub_idx)
        # Reset explicit Hs on both endpoints so valence is recomputed by sanitize
        for i in (from_idx, to_idx):
            _reset_atom_hs(rw.GetAtomWithIdx(i))
        rw.AddBond(to_idx, sub_idx, bond_type)
        _sanitize(rw, "MoveSubstituent")
        return rw

    def _swap_substituents(self, rw: Chem.RWMol, action: dict):
        """
        Swap all non-ring substituents between two atoms (typically ring atoms).
        Works on aromatic rings by operating in Kekulé form.
        """
        idx1 = self._resolve_atom(rw, action["atom1"])
        idx2 = self._resolve_atom(rw, action["atom2"])

        if idx1 == idx2:
            raise ActionError("SwapSubstituents: atom1 and atom2 are the same atom")

        shared_ring_atoms = set()
        for ring in _shared_rings(rw, idx1, idx2):
            shared_ring_atoms.update(ring)

        def get_substituents(center, other_center):
            subs = []
            for nb in rw.GetAtomWithIdx(center).GetNeighbors():
                nidx = nb.GetIdx()
                if nidx in shared_ring_atoms and nidx != other_center:
                    continue
                if nidx == other_center:
                    continue
                subs.append((nidx, rw.GetBondBetweenAtoms(center, nidx).GetBondType()))
            return subs

        subs1 = get_substituents(idx1, idx2)
        subs2 = get_substituents(idx2, idx1)

        rw = _kekulize_rw(rw)

        for sub_idx, _ in subs1:
            rw.RemoveBond(idx1, sub_idx)
        for sub_idx, _ in subs2:
            rw.RemoveBond(idx2, sub_idx)

        # Reset explicit Hs on swap targets so valence is recomputed
        for center in (idx1, idx2):
            _reset_atom_hs(rw.GetAtomWithIdx(center))

        for sub_idx, btype in subs1:
            bt = Chem.BondType.SINGLE if btype == Chem.BondType.AROMATIC else btype
            rw.AddBond(idx2, sub_idx, bt)
        for sub_idx, btype in subs2:
            bt = Chem.BondType.SINGLE if btype == Chem.BondType.AROMATIC else btype
            rw.AddBond(idx1, sub_idx, bt)

        _sanitize(rw, "SwapSubstituents")
        return rw

    # ── Dispatch ─────────────────────────────────────────────────────────────

    _SMILES_LEVEL = {"FlipEZ", "Batch"}

    def _dispatch(self, smiles: str, action: dict) -> str:
        atype = action.get("type")
        if not atype:
            raise UnknownActionError("Action dict missing 'type' key")

        # Batch: apply actions sequentially at SMILES level
        if atype == "Batch":
            for sub in action["actions"]:
                smiles = self.execute(smiles, sub)
            return smiles

        # FlipEZ: SMILES-level operation
        if atype == "FlipEZ":
            return self._flip_ez(smiles, action)

        # All other ops: work on RWMol
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise InvalidMolError(f"Cannot parse SMILES: {smiles!r}")
        rw = Chem.RWMol(copy.deepcopy(mol))

        handlers = {
            "ChangeAtom":        self._change_atom,
            "AddAtom":           self._add_atom,
            "RemoveAtom":        self._remove_atom,
            "ChangeBond":        self._change_bond,
            "AddBond":           self._add_bond,
            "RemoveBond":        self._remove_bond,
            "AddGroup":          self._add_group,
            "RemoveGroup":       self._remove_group,
            "FlipChirality":     self._flip_chirality,
            "MoveSubstituent":   self._move_substituent,
            "SwapSubstituents":  self._swap_substituents,
        }

        handler = handlers.get(atype)
        if handler is None:
            raise UnknownActionError(
                f"Unknown action type: {atype!r}. "
                f"Valid types: {sorted(handlers) + ['FlipEZ', 'Batch']}"
            )

        result_rw = handler(rw, action)
        return _to_canonical(result_rw)

    # ── Public API ────────────────────────────────────────────────────────────

    def execute(self, smiles: str, action: dict) -> str:
        """
        Apply a single Action to smiles.

        Args:
            smiles: input SMILES string (with or without atom map numbers)
            action: Action dict with 'type' key and type-specific fields

        Returns:
            Canonical SMILES after applying the action.

        Raises:
            AtomNotFoundError, BondNotFoundError, InvalidMolError, UnknownActionError
        """
        return self._dispatch(smiles, action)

    def execute_batch(self, smiles: str, actions: list) -> str:
        """
        Apply a list of Actions sequentially.

        Equivalent to: execute(smiles, {"type": "Batch", "actions": actions})
        """
        return self._dispatch(smiles, {"type": "Batch", "actions": actions})


# ---------------------------------------------------------------------------
# Inline tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ex = ActionExecutor()
    passed = failed = 0

    def test(name, smiles, action, expected):
        global passed, failed
        try:
            result = ex.execute(smiles, action)
            exp_canon = Chem.MolToSmiles(Chem.MolFromSmiles(expected))
            if result == exp_canon:
                print(f"  PASS  {name}")
                passed += 1
            else:
                print(f"  FAIL  {name}")
                print(f"        input:    {smiles}")
                print(f"        expected: {exp_canon}")
                print(f"        got:      {result}")
                failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            failed += 1

    print("=" * 60)
    print("ActionExecutor — inline tests")
    print("=" * 60)

    # 1. ChangeAtom — element
    test("ChangeAtom/element",
         "CCCO",
         {"type": "ChangeAtom", "atom": {"idx": 2}, "element": "N"},
         "CCNO")

    # 2. ChangeAtom — charge
    test("ChangeAtom/charge",
         "CC(=O)[O-]",
         {"type": "ChangeAtom", "atom": {"smarts": "[O-]"}, "charge": 0},
         "CC([O])=O")  # canonical RDKit form of acetic acid

    # 3. AddAtom — via idx
    try:
        r = ex.execute("CCO", {"type": "AddAtom", "anchor": {"idx": 1}, "element": "C"})
        assert Chem.MolFromSmiles(r) is not None
        print("  PASS  AddAtom")
        passed += 1
    except Exception as e:
        print(f"  FAIL  AddAtom: {e}")
        failed += 1

    # 4. RemoveAtom — via smarts
    test("RemoveAtom",
         "CCO",
         {"type": "RemoveAtom", "atom": {"smarts": "[OH]"}},
         "CC")

    # 5. ChangeBond — DOUBLE→SINGLE (C=O → C-O in acetaldehyde)
    test("ChangeBond",
         "CC=O",
         {"type": "ChangeBond",
          "bond": {"atom1": {"idx": 1}, "atom2": {"idx": 2}},
          "order": "SINGLE"},
         "CCO")

    # 6. AddBond
    test("AddBond",
         "CCCC",
         {"type": "AddBond",
          "bond": {"atom1": {"idx": 0}, "atom2": {"idx": 3}},
          "order": "SINGLE"},
         "C1CCC1")

    # 7. RemoveBond
    test("RemoveBond",
         "C1CCC1",
         {"type": "RemoveBond",
          "bond": {"atom1": {"smarts": "[CH2]", "match_idx": 0},
                   "atom2": {"smarts": "[CH2]", "match_idx": 3}}},
         "CCCC")

    # 8. AddGroup
    test("AddGroup",
         "c1ccccc1",
         {"type": "AddGroup", "anchor": {"idx": 0}, "group": "OH"},
         "Oc1ccccc1")

    # 9. RemoveGroup
    test("RemoveGroup",
         "Oc1ccccc1",
         {"type": "RemoveGroup", "anchor": {"smarts": "[c;r6]", "match_idx": 0}, "group": "OH"},
         "c1ccccc1")

    # 10. FlipChirality — via smarts
    test("FlipChirality",
         "C[C@@H](O)F",
         {"type": "FlipChirality", "atom": {"smarts": "[C@@H]"}},
         "C[C@H](O)F")

    # 11. FlipEZ
    test("FlipEZ",
         "C/C=C/C",
         {"type": "FlipEZ",
          "bond": {"atom1": {"idx": 1}, "atom2": {"idx": 2}}},
         "C/C=C\\C")

    # 12. MoveSubstituent — move Cl from position 1 to position 3 on pyridine
    # pyridine with Cl at C2: Clc1ccccn1 → move Cl to C4: c1cc(Cl)ccn1
    test("MoveSubstituent",
         "Clc1ccccn1",
         {"type": "MoveSubstituent",
          "substituent": {"smarts": "[Cl]"},
          "from_atom":   {"smarts": "[c;r6]", "match_idx": 0},
          "to_atom":     {"smarts": "[c;r6]", "match_idx": 2}},
         "c1cc(Cl)ccn1")

    # 13. SwapSubstituents — on o-chlorotoluene, swap Cl and CH3
    # Cc1ccccc1Cl → Clc1ccccc1C  (same thing, test canonical form)
    try:
        mol_in  = Chem.MolFromSmiles("Cc1ccccc1Cl")
        # assign map nums for stable ref
        rw = Chem.RWMol(mol_in)
        for a in rw.GetAtoms():
            a.SetAtomMapNum(a.GetIdx() + 1)
        mapped = Chem.MolToSmiles(rw)

        # find the two substituted ring carbons
        c_idx = next(a.GetIdx() for a in rw.GetAtoms()
                     if a.GetAtomicNum() == 6 and not a.GetIsAromatic()
                     and any(nb.GetIsAromatic() for nb in a.GetNeighbors()))
        cl_nb = next(a for a in rw.GetAtoms() if a.GetAtomicNum() == 17)
        cl_ring = next(nb.GetIdx() for nb in cl_nb.GetNeighbors())
        ch3_ring = next(nb.GetIdx() for nb in rw.GetAtomWithIdx(c_idx).GetNeighbors()
                        if nb.GetIsAromatic())

        r = ex.execute(mapped, {
            "type": "SwapSubstituents",
            "atom1": {"map_num": ch3_ring + 1},
            "atom2": {"map_num": cl_ring  + 1},
        })
        mol_r = Chem.MolFromSmiles(r)
        assert mol_r is not None
        print(f"  PASS  SwapSubstituents  ({Chem.MolToSmiles(mol_in)} → {r})")
        passed += 1
    except Exception as e:
        print(f"  FAIL  SwapSubstituents: {e}")
        failed += 1

    # 14. Batch — two actions
    test("Batch",
         "CCO",
         {"type": "Batch", "actions": [
             {"type": "ChangeAtom", "atom": {"idx": 2}, "element": "N"},
             {"type": "AddGroup",   "anchor": {"idx": 2}, "group": "CH3"},
         ]},
         "CCN(C)")

    # 15. map_num AtomRef (with pre-mapped SMILES)
    test("AtomRef/map_num",
         "[CH3:1][CH2:2][OH:3]",
         {"type": "ChangeAtom", "atom": {"map_num": 3}, "element": "N"},
         "CCN")

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed+failed} tests")
    print("=" * 60)
