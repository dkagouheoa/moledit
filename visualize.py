"""
Visualize a MolEdit dataset as a self-contained HTML gallery.

Each card shows:
  - correct molecule image  (rendered from correct_smiles)
  - wrong molecule image    (rendered from wrong_smiles)
  - operation badge + params
  - both SMILES strings
  - round-trip status badge

Usage:
    python visualize.py --dataset ./data/roundtrip_test --n 20
    python visualize.py --dataset ./data/pubchem_dataset --n 30 --seed 7
"""

import os, sys, json, base64, argparse, random, tempfile
sys.path.insert(0, os.path.dirname(__file__))

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, AllChem
RDLogger.DisableLog("rdApp.*")

OP_COLORS = {
    "change_atom_element":    "#e63946",
    "add_atom":               "#2a9d8f",
    "remove_atom":            "#e9c46a",
    "change_charge":          "#f4a261",
    "change_bond_order":      "#a8dadc",
    "add_bond":               "#457b9d",
    "remove_bond":            "#6d6875",
    "add_functional_group":   "#b5e48c",
    "remove_functional_group":"#99d98c",
    "flip_chirality":         "#f72585",
    "flip_ez":                "#7209b7",
}

HTML_STYLE = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Courier New', monospace;
  background: #0f0f1a;
  color: #cdd6f4;
  padding: 24px;
}
h1 { color: #cba6f7; font-size: 1.4rem; margin-bottom: 6px; }
.subtitle { color: #6c7086; font-size: 0.82rem; margin-bottom: 24px; }
.stats { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:24px; }
.stat-chip {
  background: #1e1e2e; border: 1px solid #313244;
  border-radius: 20px; padding: 4px 14px;
  font-size: 0.78rem; color: #a6adc8;
}
.stat-chip span { color: #cba6f7; font-weight: bold; }

/* Grid */
.grid { display: flex; flex-wrap: wrap; gap: 20px; }

/* Card */
.card {
  background: #1e1e2e;
  border: 1px solid #313244;
  border-radius: 12px;
  padding: 16px;
  width: 520px;
  transition: transform .15s, box-shadow .15s;
}
.card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px #00000080; }

/* Op badge */
.op-badge {
  display: inline-block;
  border-radius: 6px;
  padding: 3px 10px;
  font-size: 0.72rem;
  font-weight: bold;
  letter-spacing: .04em;
  margin-bottom: 10px;
  color: #0f0f1a;
}
.status-ok   { color: #a6e3a1; font-size: 0.72rem; font-weight: bold; }
.status-fail { color: #f38ba8; font-size: 0.72rem; font-weight: bold; }
.card-header { display:flex; justify-content:space-between; align-items:center; }

/* Mol images side by side */
.mol-row {
  display: flex;
  gap: 10px;
  margin: 10px 0;
}
.mol-box {
  flex: 1;
  background: #181825;
  border: 1px solid #313244;
  border-radius: 8px;
  overflow: hidden;
  text-align: center;
}
.mol-box img { width: 100%; display: block; background: white; }
.mol-label {
  font-size: 0.68rem;
  padding: 3px 0 4px;
  letter-spacing: .05em;
  font-weight: bold;
}
.label-correct { color: #a6e3a1; }
.label-wrong   { color: #f38ba8; }

/* SMILES lines */
.smiles-block { margin: 8px 0 4px; }
.smiles-row {
  display: flex;
  align-items: baseline;
  gap: 6px;
  margin: 3px 0;
  font-size: 0.72rem;
  line-height: 1.4;
}
.smiles-tag {
  flex-shrink: 0;
  font-size: 0.65rem;
  font-weight: bold;
  letter-spacing: .04em;
  padding: 1px 5px;
  border-radius: 4px;
}
.tag-correct { background:#1c3a2a; color:#a6e3a1; }
.tag-wrong   { background:#3a1c1e; color:#f38ba8; }
.smiles-val  { color:#cdd6f4; word-break:break-all; }

/* Op params */
.params {
  margin-top: 8px;
  background: #181825;
  border-radius: 6px;
  padding: 7px 10px;
  font-size: 0.70rem;
  color: #a6adc8;
}
.param-key   { color: #89b4fa; }
.param-val   { color: #f9e2af; }
.desc-line   { color: #6c7086; font-size: 0.68rem; margin-top: 5px; font-style: italic; }
.cid-line    { color: #45475a; font-size: 0.66rem; margin-top: 6px; }

.divider { border: none; border-top: 1px solid #313244; margin: 8px 0; }
"""


def mol_to_b64(smiles, size=(240, 180)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    AllChem.Compute2DCoords(mol)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    try:
        Draw.MolToFile(mol, tmp, size=size)
        with open(tmp, "rb") as f:
            return base64.b64encode(f.read()).decode()
    finally:
        os.unlink(tmp)


def params_html(params):
    parts = []
    for k, v in params.items():
        parts.append(
            f'<span class="param-key">{k}</span>'
            f'<span style="color:#6c7086">: </span>'
            f'<span class="param-val">{v}</span>'
        )
    return " &nbsp; ".join(parts)


def build_card(r):
    op_type = r.get("op_type") or r["operation"]["type"]
    color   = OP_COLORS.get(op_type, "#888")
    ok      = r.get("roundtrip_ok", None)

    correct_b64 = mol_to_b64(r["correct_smiles"])
    wrong_b64   = mol_to_b64(r["wrong_smiles"])

    status_html = ""
    if ok is True:
        status_html = '<span class="status-ok">✓ round-trip OK</span>'
    elif ok is False:
        status_html = '<span class="status-fail">✗ round-trip FAIL</span>'

    cid_html = ""
    if r.get("cid"):
        cid_html = f'<div class="cid-line">PubChem CID: {r["cid"]}</div>'

    desc = r["operation"].get("description", "")
    params = r["operation"].get("params", {})

    return f"""
<div class="card">
  <div class="card-header">
    <span class="op-badge" style="background:{color}">{op_type}</span>
    {status_html}
  </div>

  <div class="mol-row">
    <div class="mol-box">
      <div class="mol-label label-correct">✓ CORRECT</div>
      <img src="data:image/png;base64,{correct_b64}" alt="correct"/>
    </div>
    <div class="mol-box">
      <div class="mol-label label-wrong">✗ WRONG (input)</div>
      <img src="data:image/png;base64,{wrong_b64}" alt="wrong"/>
    </div>
  </div>

  <div class="smiles-block">
    <div class="smiles-row">
      <span class="smiles-tag tag-correct">CORRECT</span>
      <span class="smiles-val">{r["correct_smiles"]}</span>
    </div>
    <div class="smiles-row">
      <span class="smiles-tag tag-wrong">WRONG</span>
      <span class="smiles-val">{r["wrong_smiles"]}</span>
    </div>
  </div>

  <hr class="divider"/>

  <div class="params">
    <div><b style="color:#cdd6f4">Operation to apply:</b></div>
    <div style="margin-top:4px">{params_html(params)}</div>
    <div class="desc-line">{desc}</div>
  </div>

  {cid_html}
</div>"""


def build_html(records, title="MolEdit Dataset Gallery"):
    total  = len(records)
    ok_n   = sum(1 for r in records if r.get("roundtrip_ok") is True)
    fail_n = sum(1 for r in records if r.get("roundtrip_ok") is False)

    from collections import Counter
    op_counts = Counter(r.get("op_type") or r["operation"]["type"] for r in records)

    chips = "".join(
        f'<div class="stat-chip">{op} <span>{cnt}</span></div>'
        for op, cnt in sorted(op_counts.items(), key=lambda x: -x[1])
    )
    if ok_n or fail_n:
        chips += (f'<div class="stat-chip">✓ pass <span style="color:#a6e3a1">{ok_n}</span></div>'
                  f'<div class="stat-chip">✗ fail <span style="color:#f38ba8">{fail_n}</span></div>')

    cards = "\n".join(build_card(r) for r in records)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{title}</title>
  <style>{HTML_STYLE}</style>
</head>
<body>
  <h1>{title}</h1>
  <p class="subtitle">
    Showing {total} samples &nbsp;|&nbsp;
    Left image = <span style="color:#a6e3a1">correct molecule</span> &nbsp;|&nbsp;
    Right image = <span style="color:#f38ba8">wrong SMILES input</span> &nbsp;|&nbsp;
    Badge = correction operation needed
  </p>
  <div class="stats">{chips}</div>
  <div class="grid">{cards}</div>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./data/roundtrip_test",
                        help="Directory containing dataset.json or roundtrip_test.json")
    parser.add_argument("--n",    type=int, default=20, help="Max cards to show")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out",  default=None, help="Output HTML path (default: <dataset>/gallery.html)")
    parser.add_argument("--title", default="MolEdit Dataset Gallery")
    args = parser.parse_args()

    random.seed(args.seed)

    # Find JSON file
    for name in ("roundtrip_test.json", "dataset.json"):
        p = os.path.join(args.dataset, name)
        if os.path.exists(p):
            json_path = p
            break
    else:
        print(f"No dataset JSON found in {args.dataset}")
        sys.exit(1)

    with open(json_path) as f:
        records = json.load(f)
    print(f"Loaded {len(records)} records from {json_path}")

    sample = random.sample(records, min(args.n, len(records)))
    # Sort by op_type for visual grouping
    sample.sort(key=lambda r: r.get("op_type") or r["operation"]["type"])

    print(f"Rendering {len(sample)} cards …")
    html = build_html(sample, title=args.title)

    out_path = args.out or os.path.join(args.dataset, "gallery.html")
    with open(out_path, "w") as f:
        f.write(html)

    size_kb = os.path.getsize(out_path) // 1024
    print(f"Gallery written → {out_path}  ({size_kb} KB)")
    print(f"\nTo view: python -m http.server 8080  then open")
    print(f"  http://61.172.170.106:8080/{os.path.relpath(out_path)}")


if __name__ == "__main__":
    main()
