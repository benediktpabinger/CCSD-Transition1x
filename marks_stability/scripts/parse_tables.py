"""Parse the four per-reaction result tables out of the Marks et al. LaTeX source.

Emits one row per (table, reaction, model, workflow) with the DFT gradient count,
whether the run succeeded (non-italic = success) and the failure-mode letter.
"""
import csv
import re
import sys

TEX = open(sys.argv[1], encoding="utf-8", errors="replace").read()
OUT = sys.argv[2]

MODELS = ["GFN2-xTB", "AIMNet2", "eSEN-S", "UMA-S", "UMA-M", "MACE-OMol25"]
TABLES = [
    ("FSM", "baker", r"\label{tab:fsm_baker}"),
    ("FSM", "sharada", r"\label{tab:FSM_sharada}"),
    ("CI-NEB", "baker", r"\label{tab:NEB_baker}"),
    ("CI-NEB", "sharada", r"\label{tab:NEB_sharada}"),
]


def clean_name(s):
    s = re.sub(r"\$\\rightarrow\$", "->", s)
    s = re.sub(r"\$\\leftrightarrow\$", "<->", s)
    s = re.sub(r"\\textit\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\$_\{?(\w+)\}?\$", r"\1", s)
    s = re.sub(r"\$\^\{?(\w+)\}?\$", r"\1", s)
    s = s.replace("$", "").replace("\\", "").replace("{", "").replace("}", "")
    s = s.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", s).strip()


def parse_cell(c):
    """-> (gradients or None, success bool, failure-mode letter or '')"""
    c = c.strip()
    mode = ""
    m = re.search(r"\$\^\{?([a-z])(?:,\w+)?\}?\$", c)
    if m:
        mode = m.group(1)
    # Mode (c) -- "converges to correct TS with spurious imaginary frequency,
    # additional cost to eliminate it was added" -- is a success. Two such cells
    # are italicised anyway (NEB/baker AIMNet2 rxn 19, NEB/sharada AIMNet2 rxn 8);
    # the paper's own success-rate and mean-cost footers count both as successes,
    # so the superscript wins over the italics.
    failed = ("\\textit{" in c or "\\emph{" in c) and mode != "c"
    body = re.sub(r"\$\^[^$]*\$", "", c)
    body = re.sub(r"\\textit\{([^}]*)\}", r"\1", body)
    body = body.replace("{", "").replace("}", "").replace("$", "").strip()
    n = int(body) if re.fullmatch(r"\d+", body) else None
    return n, (not failed), mode


rows = []
for algo, setname, label in TABLES:
    i = TEX.index(label)
    block = TEX[i:TEX.index(r"\bottomrule", i)]
    body = block[block.index(r"\midrule") + len(r"\midrule"):]
    body = body[:body.index(r"\midrule")]  # stop before the summary block
    # Drop LaTeX comments line-by-line first: a row may carry a trailing comment
    # whose text would otherwise mask the row that follows it, and two rows in
    # tab:NEB_baker are commented-out earlier variants.
    body = "\n".join(re.sub(r"(?<!\\)%.*", "", ln) for ln in body.split("\n"))
    for line in body.split("\\\\"):
        line = re.sub(r"\s+", " ", line).strip()
        if not line or "&" not in line:
            continue
        cells = line.split("&")
        m = re.match(r"\s*(\d+)\.\s*(.*)", cells[0])
        if not m:
            continue
        rid, name = int(m.group(1)), clean_name(m.group(2))
        vals = cells[1:]
        if len(vals) != 12:
            print(f"WARN {algo}/{setname} rxn {rid}: {len(vals)} cells", file=sys.stderr)
            continue
        for k, model in enumerate(MODELS):
            for j, wf in enumerate(("native", "low-level refined")):
                n, ok, mode = parse_cell(vals[2 * k + j])
                rows.append({
                    "algorithm": algo, "set": setname, "reaction_id": rid,
                    "name": name, "model": model, "workflow": wf,
                    "gradients": "" if n is None else n,
                    "success": int(ok), "failure_mode": mode,
                })

with open(OUT, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

print(f"wrote {len(rows)} rows to {OUT}")
for algo, setname, _ in TABLES:
    sub = [r for r in rows if r["algorithm"] == algo and r["set"] == setname]
    nr = len({r["reaction_id"] for r in sub})
    sr = 100.0 * sum(r["success"] for r in sub) / len(sub)
    print(f"  {algo:7} {setname:8} {nr:2d} reactions, {len(sub):3d} runs, "
          f"success {sr:.1f}%")
