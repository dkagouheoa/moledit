"""
Demo: generates a small sample dataset showcasing all operation types.
Run: python demo.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))

from rdkit import Chem
from mol_corrupt import corrupt_molecule, CORRUPTION_TYPES
from dataset_builder import build_dataset, render_molecule

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")

DEMO_MOLECULES = {
    "change_atom_element":     ("CCCO",                "propanol — change atom element"),
    "add_atom":                ("CC(=O)O",             "acetic acid — extra atom added"),
    "remove_atom":             ("CCO",                 "ethanol — atom removed"),
    "change_charge":           ("CC(=O)[O-]",          "acetate ion — charge changed"),
    "change_bond_order":       ("CC=O",                "acetaldehyde — bond order changed"),
    "add_bond":                ("CCCCCC",              "hexane — spurious bond added"),
    "remove_bond":             ("CC(=O)OC",            "methyl acetate — bond removed"),
    "add_functional_group":    ("c1ccccc1",            "benzene — extra group added"),
    "remove_functional_group": ("c1ccc(O)cc1",         "phenol — group removed"),
    "flip_chirality":          ("C[C@@H](O)F",         "chiral center flipped"),
    "flip_ez":                 ("C/C=C/C",             "trans-2-butene — E/Z flipped"),
}


def run_demo():
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)

    records = []
    idx = 0

    print("=" * 70)
    print("Molecular Editing Tool — Demo")
    print("=" * 70)

    for op_type in CORRUPTION_TYPES:
        smi, desc = DEMO_MOLECULES[op_type]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"[SKIP] Cannot parse: {smi}")
            continue
        canonical = Chem.MolToSmiles(mol)

        result = corrupt_molecule(canonical, op_type=op_type, max_retries=20)
        if result is None:
            print(f"[SKIP] {op_type:30s} | {canonical:20s} | corruption failed")
            continue

        wrong_smi, correction = result
        item_id = f"demo_{idx:03d}"
        img_path = os.path.join("images", f"{item_id}.png")
        abs_img = os.path.join(OUTPUT_DIR, img_path)

        render_molecule(canonical, abs_img)

        record = {
            "id": item_id,
            "correct_smiles": canonical,
            "wrong_smiles": wrong_smi,
            "image_path": img_path,
            "operation": correction,
        }
        records.append(record)

        print(f"\n[{op_type}] {desc}")
        print(f"  Correct SMILES : {canonical}")
        print(f"  Wrong SMILES   : {wrong_smi}")
        print(f"  Correction     : {correction['description']}")
        print(f"  Image          : {img_path}")

        idx += 1

    json_path = os.path.join(OUTPUT_DIR, "demo_dataset.json")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"Generated {len(records)} demo samples")
    print(f"  Images  : {OUTPUT_DIR}/images/")
    print(f"  Metadata: {json_path}")
    print("=" * 70)

    print("\n--- Sample JSON record ---")
    if records:
        print(json.dumps(records[0], indent=2))

    return records


if __name__ == "__main__":
    run_demo()
