"""Check the parsed table CSV against the success-rate / mean-cost footer rows
printed in the paper. Any mismatch means the parse is wrong.
"""
import csv
import re
import sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
TEX = open(sys.argv[2], encoding="utf-8", errors="replace").read()

MODELS = ["GFN2-xTB", "AIMNet2", "eSEN-S", "UMA-S", "UMA-M", "MACE-OMol25"]
TABLES = [("FSM", "baker", r"\label{tab:fsm_baker}"),
          ("FSM", "sharada", r"\label{tab:FSM_sharada}"),
          ("CI-NEB", "baker", r"\label{tab:NEB_baker}"),
          ("CI-NEB", "sharada", r"\label{tab:NEB_sharada}")]

bad = 0
for algo, setname, label in TABLES:
    blk = TEX[TEX.index(label):TEX.index(r"\bottomrule", TEX.index(label))]
    sr = re.search(r"Success Rate(.*?)\\\\", blk, re.S)
    srv = [float(x) for x in re.findall(r"([\d.]+)\\%", sr.group(1))]
    mc = re.search(r"Mean Success Cost(.*?)\\\\", blk, re.S)
    mcv = [float(x) for x in re.findall(r"(\d+\.\d+)", mc.group(1))]

    agg = defaultdict(list)
    for r in rows:
        if r["algorithm"] == algo and r["set"] == setname:
            agg[(r["model"], r["workflow"])].append(r)

    print(f"\n=== {algo} / {setname}")
    k = 0
    for m in MODELS:
        for wf in ("native", "low-level refined"):
            sub = agg[(m, wf)]
            got_sr = 100.0 * sum(int(x["success"]) for x in sub) / len(sub)
            costs = [int(x["gradients"]) for x in sub
                     if int(x["success"]) and x["gradients"] != ""]
            got_mc = sum(costs) / len(costs)
            ok_sr = abs(got_sr - srv[k]) < 0.1
            ok_mc = abs(got_mc - mcv[k]) < 0.06
            bad += (not ok_sr) + (not ok_mc)
            flag = "" if (ok_sr and ok_mc) else "   <-- MISMATCH"
            print(f"  {m:12} {wf:18} SR {got_sr:5.1f} vs {srv[k]:5.1f} | "
                  f"cost {got_mc:5.2f} vs {mcv[k]:5.2f}{flag}")
            k += 1

print(f"\n{'ALL COLUMNS MATCH' if bad == 0 else str(bad) + ' MISMATCHES'}")
sys.exit(1 if bad else 0)
