"""
Visualise a sample of the generated dataset as an HTML gallery.

Usage:
    python show_dataset.py --dataset ./data/pubchem_dataset --n 20
Opens  ./data/pubchem_dataset/gallery.html  in the terminal output.
"""

import os
import sys
import json
import base64
import argparse
import random

sys.path.insert(0, os.path.dirname(__file__))


CARD_CSS = """
body { font-family: monospace; background: #1a1a2e; color: #eee; margin: 20px; }
h1   { color: #e94560; }
.grid{ display: flex; flex-wrap: wrap; gap: 16px; }
.card{
  background: #16213e; border: 1px solid #0f3460;
  border-radius: 8px; padding: 12px; width: 340px;
}
.card img { width: 100%; border-radius: 4px; background: white; }
.op-type { color: #e94560; font-weight: bold; font-size: 0.85em; }
.smiles  { font-size: 0.75em; word-break: break-all; margin: 4px 0; }
.correct { color: #4ecca3; }
.wrong   { color: #ff6b6b; }
.desc    { color: #ffd460; font-size: 0.78em; margin-top: 6px; }
.cid     { color: #888; font-size: 0.7em; }
"""


def img_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def make_gallery(records, output_dir, n=30):
    sample = random.sample(records, min(n, len(records)))
    cards_html = []

    for r in sample:
        abs_img = os.path.join(output_dir, r["image_path"])
        if not os.path.exists(abs_img):
            continue

        b64 = img_to_b64(abs_img)
        op = r["operation"]
        params_str = ", ".join(f"{k}={v}" for k, v in op.get("params", {}).items())

        cards_html.append(f"""
<div class="card">
  <img src="data:image/png;base64,{b64}" alt="molecule"/>
  <div class="cid">PubChem CID: {r.get('cid', '?')}</div>
  <div class="op-type">&#9654; {op['type']}({params_str})</div>
  <div class="smiles correct">&#10003; {r['correct_smiles']}</div>
  <div class="smiles wrong">&#10007; {r['wrong_smiles']}</div>
  <div class="desc">{op['description']}</div>
</div>""")

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>MolEdit Dataset — PubChem Sample</title>
  <style>{CARD_CSS}</style>
</head>
<body>
  <h1>MolEdit Dataset — PubChem Sample ({len(cards_html)} molecules)</h1>
  <p style="color:#888">Green = correct SMILES &nbsp;|&nbsp; Red = wrong SMILES &nbsp;|&nbsp;
     Yellow = correction operation needed &nbsp;|&nbsp; Image shows the <em>correct</em> molecule.</p>
  <div class="grid">{''.join(cards_html)}</div>
</body>
</html>"""

    out_path = os.path.join(output_dir, "gallery.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Gallery written to: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./data/pubchem_dataset")
    parser.add_argument("--n",       type=int, default=30)
    parser.add_argument("--seed",    type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    json_path = os.path.join(args.dataset, "dataset.json")
    with open(json_path) as f:
        records = json.load(f)

    print(f"Loaded {len(records)} records from {json_path}")

    # Print text summary per operation type
    from collections import Counter
    counts = Counter(r["operation"]["type"] for r in records)
    print("\nOperation distribution:")
    for op, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {op:<35} {cnt:>5}")

    make_gallery(records, args.dataset, n=args.n)
