"""Find which Sharada reference TS geometries duplicate Baker ones.

Compares raw coordinates (same atom order) and reports max |dx| per pair so an
exact duplicate is distinguishable from a re-optimised near-copy.
"""
import os
import sys

ROOT = sys.argv[1]


def read(path):
    lines = [l for l in open(path).read().split("\n") if l.strip()]
    n = int(lines[0].split()[0])
    syms, xyz = [], []
    for ln in lines[2:2 + n]:
        p = ln.split()
        syms.append(p[0])
        xyz.append(tuple(float(v) for v in p[1:4]))
    return syms, xyz


def load(setname):
    base = os.path.join(ROOT, "data", setname)
    out = {}
    for d in sorted(os.listdir(base)):
        out[d] = read(os.path.join(base, d, "ts.xyz"))
    return out


baker = load("baker")
sharada = load("sharada")

print(f"{'sharada':28} {'closest baker':28} {'max|dx| (A)':>12}  verdict")
print("-" * 84)
for sd, (ss, sx) in sharada.items():
    best, bestd = None, None
    for bd, (bs, bx) in baker.items():
        if bs != ss:
            continue
        m = max(abs(a - b) for pa, pb in zip(sx, bx) for a, b in zip(pa, pb))
        if bestd is None or m < bestd:
            best, bestd = bd, m
    if best is None:
        print(f"{sd:28} {'(no atom-order match)':28} {'-':>12}  UNIQUE")
    else:
        v = ("EXACT DUPLICATE" if bestd < 1e-8 else
             "near-copy (re-optimised)" if bestd < 0.2 else "UNIQUE")
        print(f"{sd:28} {best:28} {bestd:12.6f}  {v}")
