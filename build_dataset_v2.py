"""
Build MolEdit dataset v2 — balanced, 20 samples per op type (13 types = 260 total).

Changes from v1:
  - target_per_op is a fixed CLI arg (default 20), not derived from n_samples
  - Uses a much larger candidate pool to handle rare op types
  - Multi-pass: exhausts full pool before giving up on any op type
  - Saves to a new output directory (default: ./data/pubchem_dataset_v2)

Usage:
    python build_dataset_v2.py \
        --input  /home/dataset-assist-0/usr/lh/mzm/data/pubchem/CID-SMILES.gz.1 \
        --output ./data/pubchem_dataset_v2 \
        --target_per_op 20 \
        --pool   20000 \
        --seed   42
"""

import os, sys, json, random, argparse, subprocess, signal
sys.path.insert(0, os.path.dirname(__file__))

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from mol_corrupt import corrupt_molecule, CORRUPTION_TYPES


# ---------------------------------------------------------------------------
# Molecule filtering
# ---------------------------------------------------------------------------

def is_valid(mol, smiles):
    if mol is None or "." in smiles:
        return False
    n = mol.GetNumAtoms()
    if n < 5 or n > 60:
        return False
    if not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
        return False
    return True


# ---------------------------------------------------------------------------
# Streaming sample from gzip file
# ---------------------------------------------------------------------------

def stream_sample(filepath, pool_size, seed=42):
    rng = random.Random(seed)
    pool = []
    proc = subprocess.Popen(
        ["zcat", filepath],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        preexec_fn=lambda: signal.signal(signal.SIGPIPE, signal.SIG_DFL),
    )
    for raw in proc.stdout:
        line = raw.decode("ascii", errors="ignore").strip()
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].isdigit():
            pool.append((parts[0], parts[1]))
        if len(pool) >= pool_size:
            break
    proc.kill(); proc.wait()
    rng.shuffle(pool)
    return pool


def render(smiles, path, size=(400, 300)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    AllChem.Compute2DCoords(mol)
    Draw.MolToFile(mol, path, size=size)
    return True


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build(input_path, output_dir, target_per_op=20, pool_size=20000, seed=42):
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    print(f"Sampling {pool_size} candidates from {input_path} ...")
    raw_pool = stream_sample(input_path, pool_size, seed=seed)
    print(f"  Got {len(raw_pool)} candidates, filtering ...")

    valid = []
    for cid, smi in raw_pool:
        mol = Chem.MolFromSmiles(smi)
        if is_valid(mol, smi):
            valid.append((cid, Chem.MolToSmiles(mol), mol))
    print(f"  {len(valid)} molecules passed filtering")

    collected = {op: 0 for op in CORRUPTION_TYPES}
    records = []
    idx = 0
    skipped = 0

    # Pass through the pool repeatedly until all op types hit target
    # or we've done max_passes full sweeps (to avoid infinite loop)
    max_passes = 5
    for pass_num in range(1, max_passes + 1):
        remaining_ops = [op for op in CORRUPTION_TYPES if collected[op] < target_per_op]
        if not remaining_ops:
            break
        print(f"\nPass {pass_num}: {len(remaining_ops)} op types still need samples")

        random.seed(seed + pass_num)
        random.shuffle(valid)

        for cid, canonical, mol in valid:
            if not remaining_ops:
                break

            for op_type in list(remaining_ops):
                if collected[op_type] >= target_per_op:
                    remaining_ops.remove(op_type)
                    continue

                result = corrupt_molecule(canonical, op_type=op_type, max_retries=15)
                if result is None:
                    skipped += 1
                    continue

                wrong_smi, wrong_smi_mapped, correction = result
                item_id = f"{idx:06d}"
                img_rel = os.path.join("images", f"{item_id}.png")
                abs_img = os.path.join(output_dir, img_rel)

                if not render(canonical, abs_img):
                    skipped += 1
                    continue

                records.append({
                    "id":                  item_id,
                    "cid":                 cid,
                    "correct_smiles":      canonical,
                    "wrong_smiles":        wrong_smi,
                    "wrong_smiles_mapped": wrong_smi_mapped,
                    "image_path":          img_rel,
                    "operation":           correction,
                })
                collected[op_type] += 1
                idx += 1

                if idx % 50 == 0:
                    print(f"  {idx} samples collected ...")

        remaining_ops = [op for op in CORRUPTION_TYPES if collected[op] < target_per_op]

    print("\nOp type coverage:")
    for op in CORRUPTION_TYPES:
        mark = "✓" if collected[op] >= target_per_op else f"✗ (only {collected[op]})"
        print(f"  {op:<35} {collected[op]:>4}  {mark}")

    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\nDone: {len(records)} samples → {json_path}  (skipped {skipped})")
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",         default="/home/dataset-assist-0/usr/lh/mzm/data/pubchem/CID-SMILES.gz.1")
    parser.add_argument("--output",        default="./data/pubchem_dataset_v2")
    parser.add_argument("--target_per_op", type=int, default=20,    help="Samples per op type")
    parser.add_argument("--pool",          type=int, default=20000,  help="Candidate pool size from input file")
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    build(
        input_path=args.input,
        output_dir=args.output,
        target_per_op=args.target_per_op,
        pool_size=args.pool,
        seed=args.seed,
    )
