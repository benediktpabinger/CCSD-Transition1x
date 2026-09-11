"""Join the stability classification onto the paper's per-reaction MLIP results.

  python crosstab.py <marks_stability.csv> <marks_per_reaction.csv> <out_md>

Counts only. With one unstable reaction in the set there is nothing to test and
no test is applied.
"""
import csv
import sys
from collections import defaultdict

stab_csv, runs_csv, out_md = sys.argv[1], sys.argv[2], sys.argv[3]

cls, name = {}, {}
for r in csv.DictReader(open(stab_csv, encoding="utf-8")):
    key = (r["set"], int(r["reaction_id"]))
    s = r["stable"]
    cls[key] = "unstable" if s == "0" else "stable" if s == "1" else "open-shell"
    name[key] = r["name"]

runs = list(csv.DictReader(open(runs_csv, encoding="utf-8")))
for r in runs:
    r["key"] = (r["set"], int(r["reaction_id"]))
    r["ok"] = r["success"] == "1"
    r["g"] = int(r["gradients"]) if r["gradients"] else None

MODELS = ["GFN2-xTB", "AIMNet2", "eSEN-S", "UMA-S", "UMA-M", "MACE-OMol25"]
ALGOS = ["FSM", "CI-NEB"]
CLASSES = ["stable", "unstable", "open-shell"]


def stats(rows):
    n = len(rows)
    if not n:
        return None
    fails = sum(1 for r in rows if not r["ok"])
    costs = [r["g"] for r in rows if r["ok"] and r["g"] is not None]
    mean = sum(costs) / len(costs) if costs else float("nan")
    return n, fails, 100.0 * fails / n, mean


L = []
L.append("# Stability class vs. MLIP transition-state search outcome\n")
L.append("Reference transition states of the Marks benchmark (arXiv:2604.00405),")
L.append("classified RKS-stable / RKS-unstable at that paper's own level,")
L.append("wB97X-V/def2-TZVP, joined onto its per-reaction results.\n")

# class sizes over the paper's own table rows
sizes = defaultdict(int)
for k, c in cls.items():
    sizes[c] += 1
L.append("## Class sizes (paper table rows: 24 Baker + 9 Sharada = 33)\n")
L.append("| class | rows |")
L.append("|---|---|")
for c in CLASSES:
    L.append(f"| {c} | {sizes[c]} |")
L.append("")
L.append("The Sharada set repeats five Baker reference structures, so the 33 rows")
L.append("cover 28 distinct structures. Counts below are over table rows, i.e. over")
L.append("the runs the paper actually reports.\n")

for algo in ALGOS:
    L.append(f"## {algo}\n")
    L.append("### Aggregated over all six models\n")
    L.append("| class | runs | failures | failure rate | mean gradients (successes) |")
    L.append("|---|---|---|---|---|")
    for c in CLASSES:
        rows = [r for r in runs if r["algorithm"] == algo and cls.get(r["key"]) == c]
        s = stats(rows)
        if s:
            L.append(f"| {c} | {s[0]} | {s[1]} | {s[2]:.1f}% | {s[3]:.2f} |")
    L.append("")
    L.append("### Per model\n")
    L.append("| model | class | runs | failures | failure rate | mean gradients |")
    L.append("|---|---|---|---|---|---|")
    for m in MODELS:
        for c in CLASSES:
            rows = [r for r in runs if r["algorithm"] == algo
                    and r["model"] == m and cls.get(r["key"]) == c]
            s = stats(rows)
            if s:
                L.append(f"| {m} | {c} | {s[0]} | {s[1]} | {s[2]:.1f}% | {s[3]:.2f} |")
    L.append("")

L.append("## Raw list\n")
L.append("Failure modes: (a) alternate first-order saddle, (b) local minimum,")
L.append("(c) spurious imaginary frequency [counted a success by the paper],")
L.append("(d) no P-RFO convergence in 250 cycles, (e) SCF error,")
L.append("(f) multiple strong imaginary frequencies, (g) P-RFO step failure,")
L.append("(h) internal-coordinate back-transformation failure.\n")
L.append("| set | # | reaction | class | algorithm | failures / 12 runs | modes |")
L.append("|---|---|---|---|---|---|---|")
for key in sorted(cls, key=lambda k: (k[0], k[1])):
    for algo in ALGOS:
        rows = [r for r in runs if r["key"] == key and r["algorithm"] == algo]
        if not rows:
            continue
        f = sum(1 for r in rows if not r["ok"])
        modes = sorted({r["failure_mode"] for r in rows
                        if not r["ok"] and r["failure_mode"]})
        L.append(f"| {key[0]} | {key[1]} | {name[key]} | {cls[key]} | {algo} | "
                 f"{f} | {','.join(modes) or '-'} |")

open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
print(f"wrote {out_md}")
print("\nclass sizes:", dict(sizes))
for algo in ALGOS:
    print(f"\n{algo}:")
    for c in CLASSES:
        rows = [r for r in runs if r["algorithm"] == algo and cls.get(r["key"]) == c]
        s = stats(rows)
        if s:
            print(f"  {c:11} runs {s[0]:3d}  failures {s[1]:3d} ({s[2]:5.1f}%)  "
                  f"mean gradients {s[3]:.2f}")
