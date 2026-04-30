"""
Validate pubchem_dataset records using ActionExecutor.

Translates old-format operation dicts (raw atom_idx params) to the new
ActionExecutor JSON protocol, then checks that executing the action on
wrong_smiles reproduces correct_smiles.

Usage:
    python validate_with_executor.py [--dataset ./data/pubchem_dataset] [--verbose]
"""

import os, sys, json, argparse
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(__file__))

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

from action_executor import ActionExecutor, ActionError


# ---------------------------------------------------------------------------
# Old-format → ActionExecutor JSON translation
# ---------------------------------------------------------------------------

def translate_operation(op: dict) -> dict:
    """
    Convert a mol_corrupt.py correction dict to an ActionExecutor action dict.

    New format (atom_map / anchor_map / atom1_map / atom2_map) uses map_num refs.
    Old format (atom_idx / anchor_idx / atom_idx1 / atom_idx2) uses idx refs.
    Both are handled transparently.
    """
    old_type = op["type"]
    params   = op.get("params", {})

    # Determine ref style: map-num (new) or raw idx (old)
    has_maps = any(k in params for k in ("atom_map", "anchor_map", "atom1_map", "atom2_map"))

    def atom_ref(map_key, idx_key):
        if has_maps and map_key in params:
            return {"map_num": params[map_key]}
        return {"idx": params[idx_key]}

    def bond_ref(map1, idx1, map2, idx2):
        return {"atom1": atom_ref(map1, idx1), "atom2": atom_ref(map2, idx2)}

    if old_type == "change_atom_element":
        return {
            "type":    "ChangeAtom",
            "atom":    atom_ref("atom_map", "atom_idx"),
            "element": params["to_elem"],
        }

    if old_type == "change_charge":
        return {
            "type":   "ChangeAtom",
            "atom":   atom_ref("atom_map", "atom_idx"),
            "charge": params["to_charge"],
        }

    if old_type == "add_atom":
        return {
            "type":       "AddAtom",
            "anchor":     atom_ref("anchor_map", "anchor_idx"),
            "element":    params["new_elem"],
            "bond_order": params.get("bond_type", "SINGLE"),
        }

    if old_type == "remove_atom":
        return {
            "type": "RemoveAtom",
            "atom": atom_ref("atom_map", "atom_idx"),
        }

    if old_type == "change_bond_order":
        return {
            "type":  "ChangeBond",
            "bond":  bond_ref("atom1_map", "atom_idx1", "atom2_map", "atom_idx2"),
            "order": params["to_order"],
        }

    if old_type == "add_bond":
        return {
            "type":  "AddBond",
            "bond":  bond_ref("atom1_map", "atom_idx1", "atom2_map", "atom_idx2"),
            "order": params.get("bond_type", "SINGLE"),
        }

    if old_type == "remove_bond":
        return {
            "type": "RemoveBond",
            "bond": bond_ref("atom1_map", "atom_idx1", "atom2_map", "atom_idx2"),
        }

    if old_type == "add_functional_group":
        return {
            "type":   "AddGroup",
            "anchor": atom_ref("anchor_map", "anchor_idx"),
            "group":  params["group_name"],
        }

    if old_type == "remove_functional_group":
        return {
            "type":   "RemoveGroup",
            "anchor": atom_ref("anchor_map", "anchor_idx"),
            "group":  params["group_name"],
        }

    if old_type == "flip_chirality":
        return {
            "type": "FlipChirality",
            "atom": atom_ref("atom_map", "atom_idx"),
        }

    if old_type == "flip_ez":
        # SMILES-level op — find the double bond automatically
        return {"type": "FlipEZ"}

    if old_type == "move_substituent":
        return {
            "type":        "MoveSubstituent",
            "substituent": {"map_num": params["substituent_map"]},
            "from_atom":   {"map_num": params["from_map"]},
            "to_atom":     {"map_num": params["to_map"]},
        }

    if old_type == "swap_substituents":
        return {
            "type":  "SwapSubstituents",
            "atom1": {"map_num": params["ring1_map"]},
            "atom2": {"map_num": params["ring2_map"]},
        }

    raise ValueError(f"Unhandled old type: {old_type!r}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(records, verbose=False):
    ex = ActionExecutor()
    results = defaultdict(lambda: {"pass": 0, "fail": 0, "error": 0})
    failures = []

    for r in records:
        op_type  = r["operation"]["type"]
        # Prefer mapped SMILES (new format) so map_num refs work correctly
        wrong    = r.get("wrong_smiles_mapped") or r.get("wrong_smiles", "")
        correct  = r["correct_smiles"]

        try:
            action = translate_operation(r["operation"])
        except Exception as e:
            results[op_type]["error"] += 1
            failures.append({"id": r.get("id"), "op": op_type, "kind": "translate_error", "msg": str(e)})
            continue

        try:
            restored = ex.execute(wrong, action)
        except ActionError as e:
            results[op_type]["error"] += 1
            failures.append({
                "id": r.get("id"), "op": op_type, "kind": "executor_error",
                "msg": str(e), "wrong": wrong, "action": action,
            })
            continue
        except Exception as e:
            results[op_type]["error"] += 1
            failures.append({
                "id": r.get("id"), "op": op_type, "kind": "unexpected_error",
                "msg": str(e), "wrong": wrong,
            })
            continue

        # Canonicalize both for fair comparison
        mol_r = Chem.MolFromSmiles(restored) if restored else None
        mol_c = Chem.MolFromSmiles(correct)
        canon_r = Chem.MolToSmiles(mol_r) if mol_r else None
        canon_c = Chem.MolToSmiles(mol_c) if mol_c else None

        if canon_r == canon_c:
            results[op_type]["pass"] += 1
        else:
            results[op_type]["fail"] += 1
            failures.append({
                "id": r.get("id"), "op": op_type, "kind": "mismatch",
                "wrong": wrong, "expected": correct, "got": restored,
                "action": action,
            })

    return results, failures


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(results, failures, verbose=False):
    print("=" * 76)
    print(f"{'Op Type':<35} {'Total':>6} {'Pass':>5} {'Fail':>5} {'Error':>6} {'Rate':>6}")
    print("-" * 76)

    total_p = total_f = total_e = 0
    for op in sorted(results):
        r = results[op]
        p, f, e = r["pass"], r["fail"], r["error"]
        n = p + f + e
        rate = p / n * 100 if n else 0
        mark = "" if (f + e) == 0 else " ✗"
        print(f"{op:<35} {n:>6} {p:>5} {f:>5} {e:>6} {rate:>5.0f}%{mark}")
        total_p += p; total_f += f; total_e += e

    total = total_p + total_f + total_e
    print("-" * 76)
    rate = total_p / total * 100 if total else 0
    print(f"{'TOTAL':<35} {total:>6} {total_p:>5} {total_f:>5} {total_e:>6} {rate:>5.0f}%")
    print("=" * 76)

    if failures:
        show = failures if verbose else failures[:5]
        print(f"\n--- {len(failures)} failure(s){' (showing first 5)' if not verbose and len(failures) > 5 else ''} ---")
        for fl in show:
            print(f"\n  [{fl['op']}] id={fl.get('id')}  kind={fl['kind']}")
            if fl["kind"] == "mismatch":
                print(f"    wrong    : {fl['wrong']}")
                print(f"    action   : {fl['action']}")
                print(f"    expected : {fl['expected']}")
                print(f"    got      : {fl['got']}")
            elif fl["kind"] in ("executor_error", "translate_error", "unexpected_error"):
                print(f"    wrong    : {fl.get('wrong', '')}")
                print(f"    action   : {fl.get('action', '')}")
                print(f"    error    : {fl['msg']}")
    else:
        print("\n✓ All records pass validation.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./data/pubchem_dataset")
    parser.add_argument("--verbose", action="store_true", help="Show all failures")
    args = parser.parse_args()

    for name in ("dataset.json", "roundtrip_test.json"):
        p = os.path.join(args.dataset, name)
        if os.path.exists(p):
            json_path = p
            break
    else:
        print(f"No dataset JSON found in {args.dataset}")
        sys.exit(1)

    with open(json_path) as f:
        records = json.load(f)
    print(f"Loaded {len(records)} records from {json_path}\n")

    results, failures = validate(records, verbose=args.verbose)
    report(results, failures, verbose=args.verbose)


if __name__ == "__main__":
    main()
