"""Where do the OMol25-trained models fail, and does it coincide with the one
RKS-unstable reference transition state?

  python omol_failures.py <marks_stability.csv> <marks_per_reaction.csv>
"""
import csv
import sys
from collections import defaultdict

OMOL = ["eSEN-S", "UMA-S", "UMA-M", "MACE-OMol25"]
OTHER = ["GFN2-xTB", "AIMNet2"]

stab, name = {}, {}
for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")):
    k = (r["set"], int(r["reaction_id"]))
    stab[k] = "unstable" if r["stable"] == "0" else \
              "stable" if r["stable"] == "1" else "open-shell"
    name[k] = r["name"]

runs = list(csv.DictReader(open(sys.argv[2], encoding="utf-8")))
for r in runs:
    r["k"] = (r["set"], int(r["reaction_id"]))
    r["ok"] = r["success"] == "1"

# Deduplicate: Sharada repeats five Baker reference structures. Rank over
# distinct reactions, keeping the Baker row where they coincide.
DUP_SHARADA = {1, 3, 4, 5, 6}
def keep(k):
    return not (k[0] == "sharada" and k[1] in DUP_SHARADA)

print("=== OMol25 failures per reaction (out of 8 runs: 4 models x 2 workflows,"
      " per algorithm) ===\n")
rows = []
for k in sorted(stab, key=lambda x: (x[0], x[1])):
    if not keep(k):
        continue
    rec = {"k": k, "name": name[k], "cls": stab[k]}
    for algo in ("FSM", "CI-NEB"):
        sub = [r for r in runs if r["k"] == k and r["algorithm"] == algo]
        rec[algo] = sum(1 for r in sub if r["model"] in OMOL and not r["ok"])
        rec[algo + "_oth"] = sum(1 for r in sub if r["model"] in OTHER and not r["ok"])
    rec["tot"] = rec["FSM"] + rec["CI-NEB"]
    rows.append(rec)

rows.sort(key=lambda r: -r["tot"])
print(f"{'set':8} {'#':>2} {'reaction':46} {'class':10} "
      f"{'FSM':>4} {'NEB':>4} {'tot':>4} {'other':>6}")
print("-" * 92)
for r in rows:
    if r["tot"] == 0:
        continue
    print(f"{r['k'][0]:8} {r['k'][1]:>2} {r['name'][:46]:46} {r['cls']:10} "
          f"{r['FSM']:>4} {r['CI-NEB']:>4} {r['tot']:>4} "
          f"{r['FSM_oth'] + r['CI-NEB_oth']:>6}")
n0 = sum(1 for r in rows if r["tot"] == 0)
print(f"\n({n0} of {len(rows)} distinct reactions have zero OMol25 failures)")

print("\n=== the one unstable reaction, run by run ===\n")
for k in [("baker", 6)]:
    for algo in ("FSM", "CI-NEB"):
        for wf in ("native", "low-level refined"):
            bits = []
            for m in OMOL + OTHER:
                r = next(r for r in runs if r["k"] == k and r["algorithm"] == algo
                         and r["model"] == m and r["workflow"] == wf)
                tag = "ok " if r["ok"] else "FAIL"
                bits.append(f"{m}={r['gradients'] or '-'}{r['failure_mode']}[{tag.strip()}]")
            print(f"  {algo:7} {wf:18} " + "  ".join(bits))

print("\n=== aggregate by stability class, OMol25 models only ===\n")
agg = defaultdict(lambda: [0, 0])
for r in runs:
    if r["model"] not in OMOL or not keep(r["k"]):
        continue
    a = agg[(r["algorithm"], stab[r["k"]])]
    a[0] += 1
    a[1] += 0 if r["ok"] else 1
for algo in ("FSM", "CI-NEB"):
    for c in ("stable", "unstable", "open-shell"):
        n, f = agg[(algo, c)]
        if n:
            print(f"  {algo:7} {c:11} {f:3d} failures / {n:3d} runs = {100 * f / n:5.1f}%")
