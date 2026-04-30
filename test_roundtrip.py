"""
Generate 20 verified (image, wrong_smiles, operation) samples covering all
operation types from PubChem, and test that executor can restore each one.

Usage:
    python test_roundtrip.py
"""

import os, sys, json, random
sys.path.insert(0, os.path.dirname(__file__))

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, AllChem
RDLogger.DisableLog("rdApp.*")

from mol_corrupt import corrupt_molecule, CORRUPTION_TYPES
from executor import apply_correction
from build_pubchem_dataset import reservoir_sample, is_valid_for_dataset

PUBCHEM = "/home/dataset-assist-0/usr/lh/mzm/data/pubchem/CID-SMILES.gz.1"
OUT_DIR  = "./data/roundtrip_test"
TARGET_PER_OP = 2   # aim for 2 per op type → ~20 total
POOL_SIZE = 5000
SEED = 99


def render(smiles, path):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    AllChem.Compute2DCoords(mol)
    Draw.MolToFile(mol, path, size=(400, 300))
    return True


def main():
    os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)
    random.seed(SEED)

    # ── 1. Load molecule pool ────────────────────────────────────────────────
    print(f"Loading {POOL_SIZE} candidates from PubChem …")
    raw = reservoir_sample(PUBCHEM, POOL_SIZE, seed=SEED)
    mols = []
    for cid, smi in raw:
        mol = Chem.MolFromSmiles(smi)
        if is_valid_for_dataset(mol, smi):
            mols.append((cid, Chem.MolToSmiles(mol)))
    random.shuffle(mols)
    print(f"  {len(mols)} valid molecules after filtering\n")

    # ── 2. Collect samples — at least TARGET_PER_OP per op type ─────────────
    collected   = {op: [] for op in CORRUPTION_TYPES}
    records     = []
    idx         = 0
 
    mol_iter = iter(mols * 5)   # repeat pool if needed
    while any(len(v) < TARGET_PER_OP for v in collected.values()):
        try:
            cid, canonical = next(mol_iter)
        except StopIteration:
            break

        # Try each under-represented op type
        for op_type in CORRUPTION_TYPES:
            if len(collected[op_type]) >= TARGET_PER_OP:
                continue

            result = corrupt_molecule(canonical, op_type=op_type, max_retries=15)
            if result is None:
                continue
            wrong_smi, wrong_mapped, correction = result

            # ── 3. Verify round-trip BEFORE keeping ──────────────────────────
            restored = apply_correction(wrong_mapped, correction)
            ok = (restored == canonical)

            item_id  = f"{idx:04d}"
            img_path = os.path.join("images", f"{item_id}.png")
            render(canonical, os.path.join(OUT_DIR, img_path))

            record = {
                "id":                item_id,
                "cid":               cid,
                "op_type":           op_type,
                "correct_smiles":    canonical,
                "wrong_smiles":      wrong_smi,
                "wrong_smiles_mapped": wrong_mapped,
                "image_path":        img_path,
                "operation":         correction,
                "executor_restored": restored,
                "roundtrip_ok":      ok,
            }
            records.append(record)
            collected[op_type].append(record)
            idx += 1

    # ── 4. Report ─────────────────────────────────────────────────────────────
    print("=" * 72)
    print(f"{'Op Type':<35} {'Samples':>7} {'Pass':>5} {'Fail':>5} {'Rate':>6}")
    print("-" * 72)
    total_ok = total_fail = 0
    for op in CORRUPTION_TYPES:
        recs = collected[op]
        ok   = sum(1 for r in recs if r["roundtrip_ok"])
        fail = len(recs) - ok
        rate = ok / len(recs) * 100 if recs else 0
        mark = "" if fail == 0 else " ✗"
        print(f"{op:<35} {len(recs):>7} {ok:>5} {fail:>5} {rate:>5.0f}%{mark}")
        total_ok   += ok
        total_fail += fail
    print("-" * 72)
    n = total_ok + total_fail
    print(f"{'TOTAL':<35} {n:>7} {total_ok:>5} {total_fail:>5} "
          f"{total_ok/n*100 if n else 0:>5.0f}%")
    print("=" * 72)

    # ── 5. Show failing examples ──────────────────────────────────────────────
    failures = [r for r in records if not r["roundtrip_ok"]]
    if failures:
        print(f"\n--- {len(failures)} Failing example(s) ---")
        for r in failures[:4]:
            print(f"\n  [{r['op_type']}]")
            print(f"  correct  : {r['correct_smiles']}")
            print(f"  wrong    : {r['wrong_smiles']}")
            print(f"  mapped   : {r['wrong_smiles_mapped']}")
            print(f"  op params: {r['operation']['params']}")
            print(f"  restored : {r['executor_restored']}")
    else:
        print("\n✓ All samples pass round-trip verification.")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    out_json = os.path.join(OUT_DIR, "roundtrip_test.json")
    with open(out_json, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\n{len(records)} records → {out_json}")
    print(f"Images        → {OUT_DIR}/images/")


if __name__ == "__main__":
    main()
