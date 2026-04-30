"""
Dataset builder: reads source SMILES, generates (image, wrong_smiles, operation)
tuples, and saves images as PNG + metadata as JSON.
"""

import os
import json
import uuid
from rdkit import Chem
from rdkit.Chem import Draw, AllChem

from mol_corrupt import corrupt_molecule, CORRUPTION_TYPES


DEFAULT_IMG_SIZE = (400, 300)


def render_molecule(smiles, path, size=DEFAULT_IMG_SIZE):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    AllChem.Compute2DCoords(mol)
    Draw.MolToFile(mol, path, size=size)
    return True


def build_dataset(
    smiles_list,
    output_dir,
    ops_per_mol=1,
    op_types=None,
    img_size=DEFAULT_IMG_SIZE,
):
    """Build the (image, wrong_smiles, operation) dataset.

    Args:
        smiles_list: list of correct SMILES strings
        output_dir: root output directory
        ops_per_mol: number of corruptions per molecule
        op_types: list of allowed operation types, or None for all
        img_size: (width, height) for molecule images

    Returns:
        list of dataset records (also saved to dataset.json)
    """
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    allowed_ops = op_types or CORRUPTION_TYPES
    records = []
    idx = 0

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"[SKIP] Invalid SMILES: {smi}")
            continue
        canonical = Chem.MolToSmiles(mol)

        for _ in range(ops_per_mol):
            for op_type in allowed_ops:
                result = corrupt_molecule(canonical, op_type=op_type)
                if result is None:
                    continue

                wrong_smi, correction = result
                item_id = f"{idx:05d}"
                img_path = os.path.join("images", f"{item_id}.png")
                abs_img_path = os.path.join(output_dir, img_path)

                if not render_molecule(canonical, abs_img_path, size=img_size):
                    continue

                record = {
                    "id": item_id,
                    "correct_smiles": canonical,
                    "wrong_smiles": wrong_smi,
                    "image_path": img_path,
                    "operation": correction,
                }
                records.append(record)
                idx += 1

    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(records)} samples -> {json_path}")
    return records


def build_dataset_random(
    smiles_list,
    output_dir,
    ops_per_mol=3,
    img_size=DEFAULT_IMG_SIZE,
):
    """Build dataset with random corruption types (not all types per mol)."""
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    records = []
    idx = 0

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol)

        for _ in range(ops_per_mol):
            result = corrupt_molecule(canonical)
            if result is None:
                continue

            wrong_smi, correction = result
            item_id = f"{idx:05d}"
            img_path = os.path.join("images", f"{item_id}.png")
            abs_img_path = os.path.join(output_dir, img_path)

            if not render_molecule(canonical, abs_img_path, size=img_size):
                continue

            record = {
                "id": item_id,
                "correct_smiles": canonical,
                "wrong_smiles": wrong_smi,
                "image_path": img_path,
                "operation": correction,
            }
            records.append(record)
            idx += 1

    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(records)} random samples -> {json_path}")
    return records
