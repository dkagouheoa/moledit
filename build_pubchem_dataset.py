"""
Build (image, wrong_smiles, operation) dataset from PubChem CID-SMILES file.

Usage:
    python build_pubchem_dataset.py \
        --input /home/dataset-assist-0/usr/lh/mzm/data/pubchem/CID-SMILES.gz.1 \
        --output ./data/pubchem_dataset \
        --n_samples 1000 \
        --ops_per_mol 2
"""

import os
import sys
import json
import gzip
import random
import argparse
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit import RDLogger

sys.path.insert(0, os.path.dirname(__file__))
from mol_corrupt import corrupt_molecule, CORRUPTION_TYPES

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Molecule filtering
# ---------------------------------------------------------------------------
def is_valid_for_dataset(mol, smiles):
    """Keep only drug-like, single-fragment molecules."""
    if mol is None:
        return False
    if "." in smiles:
        return False
    n = mol.GetNumAtoms()
    if n < 5 or n > 60:
        return False
    # Must have at least one carbon
    if not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
        return False
    return True


# ---------------------------------------------------------------------------
# Sampling from large gzip file
# ---------------------------------------------------------------------------
def reservoir_sample(filepath, n, seed=42):
    """Sample n records from start of a gzip file using a streaming zcat subprocess."""
    import subprocess, signal
    rng = random.Random(seed)
    pool_size = n * 10
    pool = []

    proc = subprocess.Popen(
        ["zcat", filepath],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        preexec_fn=lambda: signal.signal(signal.SIGPIPE, signal.SIG_DFL),
    )

    for raw_line in proc.stdout:
        line = raw_line.decode("ascii", errors="ignore").strip()
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].isdigit():
            pool.append((parts[0], parts[1]))
        if len(pool) >= pool_size:
            break

    proc.kill()
    proc.wait()

    rng.shuffle(pool)
    return pool[:n]


def render_molecule(smiles, path, size=(400, 300)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    AllChem.Compute2DCoords(mol)
    Draw.MolToFile(mol, path, size=size)
    return True


# ---------------------------------------------------------------------------
# Main dataset builder
# ---------------------------------------------------------------------------
def build(input_path, output_dir, n_samples=1000, ops_per_mol=1, seed=42):
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    print(f"Sampling {n_samples * 4} candidates from {input_path} ...")
    candidates = reservoir_sample(input_path, n_samples * 4, seed=seed)
    print(f"  Got {len(candidates)} candidates, filtering ...")

    valid = []
    for cid, smi in candidates:
        mol = Chem.MolFromSmiles(smi)
        if is_valid_for_dataset(mol, smi):
            valid.append((cid, Chem.MolToSmiles(mol), mol))
        if len(valid) >= n_samples:
            break
    print(f"  {len(valid)} molecules passed filtering")

    # Target per op type — cap at 20 so rare types don't cause infinite looping
    # Cap at 20 per op so rare types don't cause infinite looping
    target_per_op = min(20, max(1, n_samples // len(CORRUPTION_TYPES)))
    collected = {op: 0 for op in CORRUPTION_TYPES}
    records = []
    idx = 0
    skipped = 0

    # Single pass — no repeating the pool for rare op types
    for cid, canonical, mol in valid:
        for op_type in CORRUPTION_TYPES:
            if collected[op_type] >= target_per_op:
                continue

            result = corrupt_molecule(canonical, op_type=op_type, max_retries=10)
            if result is None:
                skipped += 1
                continue

            wrong_smi, wrong_smi_mapped, correction = result
            item_id = f"{idx:06d}"
            img_path = os.path.join("images", f"{item_id}.png")
            abs_img = os.path.join(output_dir, img_path)

            if not render_molecule(canonical, abs_img):
                skipped += 1
                continue

            records.append({
                "id":                  item_id,
                "cid":                 cid,
                "correct_smiles":      canonical,
                "wrong_smiles":        wrong_smi,
                "wrong_smiles_mapped": wrong_smi_mapped,
                "image_path":          img_path,
                "operation":           correction,
            })
            collected[op_type] += 1
            idx += 1

        if idx % 100 == 0 and idx > 0:
            print(f"  {idx} samples ...")
        if all(collected[op] >= target_per_op for op in CORRUPTION_TYPES):
            break

    print("\nOp type coverage:")
    for op in CORRUPTION_TYPES:
        print(f"  {op:<35} {collected[op]:>4}")

    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\nDone: {len(records)} samples → {json_path}")
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="/home/dataset-assist-0/usr/lh/mzm/data/pubchem/CID-SMILES.gz.1")
    parser.add_argument("--output", default="./data/pubchem_dataset")
    parser.add_argument("--n_samples",  type=int, default=1000)
    parser.add_argument("--ops_per_mol", type=int, default=2)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    build(
        input_path=args.input,
        output_dir=args.output,
        n_samples=args.n_samples,
        ops_per_mol=args.ops_per_mol,
        seed=args.seed,
    )
