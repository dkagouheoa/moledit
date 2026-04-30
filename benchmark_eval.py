"""
Benchmark evaluator for MolEdit dataset.

Sends (wrong_smiles_mapped + correct_image) to a vision LLM, asks it to predict
the correction action in ActionExecutor JSON format, then verifies execution.

Usage:
    # Dry run — prints prompt for first sample, no API call
    python benchmark_eval.py --dataset ./data/pubchem_dataset --dry_run

    # Real run
    python benchmark_eval.py \\
        --dataset ./data/pubchem_dataset \\
        --model gpt-4o \\
        --api_key sk-... \\
        --api_base https://api.openai.com/v1 \\
        --out ./data/pubchem_dataset/benchmark_results.json \\
        --n 20

    # Filter to one op type
    python benchmark_eval.py ... --op change_atom_element
"""

import os, sys, json, re, time, base64, argparse, random, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(__file__))

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

from action_executor import ActionExecutor, ActionError

# ---------------------------------------------------------------------------
# Op type name mapping: dataset format → ActionExecutor type
# ---------------------------------------------------------------------------

OLD_TO_NEW_TYPE = {
    "change_atom_element":     "ChangeAtom",
    "change_charge":           "ChangeAtom",
    "add_atom":                "AddAtom",
    "remove_atom":             "RemoveAtom",
    "change_bond_order":       "ChangeBond",
    "add_bond":                "AddBond",
    "remove_bond":             "RemoveBond",
    "add_functional_group":    "AddGroup",
    "remove_functional_group": "RemoveGroup",
    "flip_chirality":          "FlipChirality",
    "flip_ez":                 "FlipEZ",
    "move_substituent":        "MoveSubstituent",
    "swap_substituents":       "SwapSubstituents",
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a molecular structure correction assistant.

You will be shown:
  1. An image of the CORRECT molecule
  2. A SMILES string with atom map numbers (:[n] tags) representing the WRONG molecule

Your task: identify what is wrong and output a single JSON action that corrects the wrong SMILES to match the image.

## Action types and schemas

  ChangeAtom:       {"type": "ChangeAtom",  "atom": AtomRef, "element": str, "charge": int}
  AddAtom:          {"type": "AddAtom",     "anchor": AtomRef, "element": str, "bond_order": str}
  RemoveAtom:       {"type": "RemoveAtom",  "atom": AtomRef}
  ChangeBond:       {"type": "ChangeBond",  "bond": BondRef, "order": str}
  AddBond:          {"type": "AddBond",     "bond": BondRef, "order": str}
  RemoveBond:       {"type": "RemoveBond",  "bond": BondRef}
  AddGroup:         {"type": "AddGroup",    "anchor": AtomRef, "group": str}
  RemoveGroup:      {"type": "RemoveGroup", "anchor": AtomRef, "group": str}
  FlipChirality:    {"type": "FlipChirality", "atom": AtomRef}
  FlipEZ:           {"type": "FlipEZ"}
  MoveSubstituent:  {"type": "MoveSubstituent", "substituent": AtomRef, "from_atom": AtomRef, "to_atom": AtomRef}
  SwapSubstituents: {"type": "SwapSubstituents", "atom1": AtomRef, "atom2": AtomRef}

## AtomRef and BondRef

  AtomRef:  {"map_num": <integer>}   ← the number after : in the SMILES, e.g. :5 → map_num 5
  BondRef:  {"atom1": AtomRef, "atom2": AtomRef}

## Valid values

  bond_order: "SINGLE" | "DOUBLE" | "TRIPLE"
  group:      "OH" | "NH2" | "F" | "Cl" | "Br" | "CH3" | "COOH" | "CHO" | "NO2"
  element:    standard element symbol, e.g. "C", "N", "O", "S", "F", "Cl", "Br"

## Output format

Output ONLY a valid JSON object with no additional text or explanation.
Example: {"type": "ChangeAtom", "atom": {"map_num": 3}, "element": "N"}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _img_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _extract_json(raw: str):
    """Parse JSON from LLM output, handling markdown fences."""
    raw = raw.strip()
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Strip markdown fences
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Find first {...} block
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _canon(smiles):
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None


# ---------------------------------------------------------------------------
# BenchmarkEvaluator
# ---------------------------------------------------------------------------

class BenchmarkEvaluator:
    def __init__(self, model, api_key, api_base, dataset_dir, delay=0.5, workers=4):
        self.model       = model
        self.dataset_dir = dataset_dir
        self.delay       = delay
        self.workers     = workers
        self.executor    = ActionExecutor()
        self._lock       = threading.Lock()

        if api_key and model:
            import openai
            self.client = openai.OpenAI(api_key=api_key, base_url=api_base)
        else:
            self.client = None

    # ── Data loading ─────────────────────────────────────────────────────────

    def _load_dataset(self, n=None, op_filter=None, seed=0):
        for fname in ("dataset.json", "roundtrip_test.json"):
            p = os.path.join(self.dataset_dir, fname)
            if os.path.exists(p):
                with open(p) as f:
                    records = json.load(f)
                break
        else:
            raise FileNotFoundError(f"No dataset JSON in {self.dataset_dir}")

        if op_filter:
            records = [r for r in records if r["operation"]["type"] == op_filter]

        if n and n < len(records):
            random.seed(seed)
            records = random.sample(records, n)

        return records

    # ── LLM call ─────────────────────────────────────────────────────────────

    def _call_llm(self, record):
        img_path = os.path.join(self.dataset_dir, record["image_path"])
        b64      = _img_to_b64(img_path)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Wrong SMILES (with atom map numbers): {record['wrong_smiles_mapped']}\n\nThe image below shows the CORRECT molecule. Output the JSON action to fix the wrong SMILES.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            },
        ]

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096,
            temperature=0,
        )
        return resp.choices[0].message.content

    # ── Evaluate one record ───────────────────────────────────────────────────

    def _evaluate_one(self, record, raw_output):
        op_type    = record["operation"]["type"]
        gt_ae_type = OLD_TO_NEW_TYPE.get(op_type, op_type)
        result = {
            "id":                record.get("id"),
            "op_type":           op_type,
            "image_path":        record.get("image_path"),
            "wrong_smiles_mapped": record["wrong_smiles_mapped"],
            "correct_smiles":    record["correct_smiles"],
            "predicted_raw":     raw_output,
            "predicted_action":  None,
            "restored_smiles":   None,
            "exec_match":        False,
            "type_match":        False,
            "parse_error":       False,
            "exec_error":        None,
        }

        # 1. Parse
        action = _extract_json(raw_output)
        if action is None:
            result["parse_error"] = True
            return result
        result["predicted_action"] = action

        # 2. Type match
        result["type_match"] = (action.get("type") == gt_ae_type)

        # 3. Execute
        try:
            restored = self.executor.execute(record["wrong_smiles_mapped"], action)
            result["restored_smiles"] = restored
        except ActionError as e:
            result["exec_error"] = str(e)
            return result
        except Exception as e:
            result["exec_error"] = f"unexpected: {e}"
            return result

        # 4. Compare (canonical)
        result["exec_match"] = (_canon(restored) == _canon(record["correct_smiles"]))
        return result

    # ── Report ────────────────────────────────────────────────────────────────

    def _report(self, results, out_path=None):
        by_op = defaultdict(list)
        for r in results:
            by_op[r["op_type"]].append(r)

        total = len(results)
        total_exec = sum(1 for r in results if r["exec_match"])
        total_type = sum(1 for r in results if r["type_match"])
        total_parse = sum(1 for r in results if r["parse_error"])
        total_err   = sum(1 for r in results if r["exec_error"])

        w = 78
        print("=" * w)
        print(f"  Model: {self.model or 'dry_run'}   N={total}")
        print("=" * w)
        print(f"{'Op Type':<32} {'N':>4}  {'ExecMatch':>10}  {'TypeMatch':>10}  {'ParseErr':>8}  {'ExecErr':>7}")
        print("-" * w)

        for op in sorted(by_op):
            recs   = by_op[op]
            n      = len(recs)
            em     = sum(1 for r in recs if r["exec_match"])
            tm     = sum(1 for r in recs if r["type_match"])
            pe     = sum(1 for r in recs if r["parse_error"])
            ee     = sum(1 for r in recs if r["exec_error"])
            mark   = "" if em == n else " ✗"
            print(f"{op:<32} {n:>4}  {em:>4}/{n:<4}{em/n*100:>4.0f}%  "
                  f"{tm:>4}/{n:<4}{tm/n*100:>4.0f}%  {pe:>8}  {ee:>7}{mark}")

        print("-" * w)
        print(f"{'TOTAL':<32} {total:>4}  {total_exec:>4}/{total:<4}{total_exec/total*100:>4.0f}%  "
              f"{total_type:>4}/{total:<4}{total_type/total*100:>4.0f}%  "
              f"{total_parse:>8}  {total_err:>7}")
        print("=" * w)
        if out_path:
            print(f"Results → {out_path}")

    # ── Main evaluate loop ────────────────────────────────────────────────────

    def evaluate(self, n=None, op_filter=None, out_path=None, seed=0):
        records = self._load_dataset(n=n, op_filter=op_filter, seed=seed)
        print(f"Loaded {len(records)} records from {self.dataset_dir}")

        # Resume: load existing results
        done_ids = set()
        results  = []
        if out_path and os.path.exists(out_path):
            with open(out_path) as f:
                results = json.load(f)
            done_ids = {r["id"] for r in results}
            print(f"Resuming: {len(done_ids)} already evaluated")

        todo = [r for r in records if r.get("id") not in done_ids]
        total = len(records)

        def _process(record):
            raw = self._call_llm(record)
            if self.delay > 0:
                time.sleep(self.delay)
            return self._evaluate_one(record, raw)

        completed = len(done_ids)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(_process, r): r for r in todo}
            for fut in as_completed(futures):
                result = fut.result()
                status = "✓" if result["exec_match"] else ("?" if result["parse_error"] else "✗")
                with self._lock:
                    results.append(result)
                    completed += 1
                    print(f"  [{completed}/{total}] id={result['id']} op={result['op_type']} {status}", flush=True)
                    if out_path:
                        with open(out_path, "w") as f:
                            json.dump(results, f, indent=2)

        self._report(results, out_path)
        return results

    def dry_run(self, op_filter=None):
        """Print the prompt for the first matching record, no API call."""
        records = self._load_dataset(n=None, op_filter=op_filter)
        if not records:
            print("No records found.")
            return
        r = records[0]
        img_path = os.path.join(self.dataset_dir, r["image_path"])
        print("=" * 60)
        print("SYSTEM PROMPT:")
        print("=" * 60)
        print(SYSTEM_PROMPT)
        print("=" * 60)
        print("USER MESSAGE:")
        print("=" * 60)
        print(f"Wrong SMILES (mapped): {r['wrong_smiles_mapped']}")
        print(f"[image: {img_path}  ({os.path.getsize(img_path)//1024} KB)]")
        print()
        print(f"GT operation: {r['operation']}")
        print(f"GT correct:   {r['correct_smiles']}")
        print("=" * 60)
        print("Image exists:", os.path.exists(img_path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MolEdit benchmark evaluator")
    parser.add_argument("--dataset",  default="./data/pubchem_dataset")
    parser.add_argument("--model",    default=None,  help="e.g. gpt-4o")
    parser.add_argument("--api_base", default="https://api.openai.com/v1")
    parser.add_argument("--out",      default=None,  help="Output JSON path")
    parser.add_argument("--n",        type=int, default=None, help="Sample size")
    parser.add_argument("--op",       default=None, help="Filter to one op type")
    parser.add_argument("--delay",    type=float, default=0.5, help="Seconds between API calls")
    parser.add_argument("--workers",  type=int, default=4, help="Concurrent API workers")
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--dry_run",  action="store_true", help="Print prompt, no API call")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")

    ev = BenchmarkEvaluator(
        model=args.model,
        api_key=api_key,
        api_base=args.api_base,
        dataset_dir=args.dataset,
        delay=args.delay,
        workers=args.workers,
    )

    if args.dry_run:
        ev.dry_run(op_filter=args.op)
        return

    if not args.model or not api_key:
        parser.error("--model is required and OPENAI_API_KEY env var must be set (or use --dry_run)")

    out = args.out or os.path.join(args.dataset, "benchmark_results.json")
    ev.evaluate(n=args.n, op_filter=args.op, out_path=out, seed=args.seed)


if __name__ == "__main__":
    main()
