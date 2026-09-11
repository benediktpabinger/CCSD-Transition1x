"""Inventory the reference TS geometries in thegomeslab/fsm.

For every Baker / Sharada reaction: charge, multiplicity, atom count, formula,
electron count, and whether the RKS-stability question applies (closed-shell
singlet with an even electron count).
"""
import csv
import os
import sys
from collections import Counter

Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "Si": 14, "P": 15, "S": 16, "Cl": 17}

ROOT = sys.argv[1]
OUT = sys.argv[2]

# Paper table row -> repo directory. Baker dir 07 does not exist; the paper's
# 24 rows map onto the 24 directories in ascending order.
BAKER_NAMES = {
    "01_hcn": "HCN -> HNC",
    "02_hcch": "HCCH -> CCH2",
    "03_h2co": "H2CO -> H2 + CO",
    "04_ch3o": "CH3O -> CH2OH",
    "05_cyclopropyl": "cyclopropyl ring opening",
    "06_bicyclobutane": "bicyclo[1.1.0]butane -> trans-butadiene",
    "08_formyloxyethyl": "formyloxyethyl 1,2-migration",
    "09_parentdielsalder": "parent Diels-Alder cycloaddition",
    "10_tetrazine": "s-tetrazine -> 2HCN + N2",
    "11_trans_butadiene": "trans-butadiene -> cis-butadiene",
    "12_ethane_h2_abstraction": "CH3CH3 -> CH2CH2 + H2",
    "13_hf_abstraction": "CH3CH2F -> CH2CH2 + HF",
    "14_vinyl_alcohol": "acetaldehyde keto-enol tautomerism",
    "15_hocl": "HCOCl -> HCl + CO",
    "16_h2po4_anion": "H2O + PO3- -> H2PO4-",
    "17_claisen": "CH2CHCH2CH2CHO Claisen rearrangement",
    "18_silylene_insertion": "SiH2 + CH3CH3 -> SiH3CH2CH3",
    "19_hnccs": "HNCCS -> HNC + CS",
    "20_hconh3_cation": "HCONH3+ -> NH4+ + CO",
    "21_acrolein_rot": "acrolein rotational TS",
    "22_hconhoh": "HCONHOH -> HCOHNHO",
    "23_hcn_h2": "HNC + H2 -> H2CNH",
    "24_h2cnh": "H2CNH -> HCNH2",
    "25_hcnh2": "HCNH2 -> HCN + H2",
}
SHARADA_NAMES = {
    "01_formaldehyde": "H2CO -> H2 + CO",
    "02_silane": "SiH2 + H2 -> SiH4",
    "03_ethanal": "CH2CHOH <-> CH3CHO",
    "04_ethane_dehydrogenation": "CH3CH3 -> CH2CH2 + H2",
    "05_bicyclobutane": "bicyclo[1.1.0]butane -> trans-butadiene",
    "06_diels_alder": "parent Diels-Alder cycloaddition",
    "07_hexadiene": "cis,cis-2,4-hexadiene <-> 3,4-dimethylcyclobutene",
    "08_alanine": "alanine dipeptide C5 <-> C7AX",
    "09_icr": "silyl ketene acetal -> silyl ester Ireland-Claisen",
}


def read_xyz(path):
    lines = open(path).read().split("\n")
    n = int(lines[0].split()[0])
    syms = [ln.split()[0] for ln in lines[2:2 + n] if ln.strip()]
    assert len(syms) == n, path
    return syms


def formula(syms):
    c = Counter(syms)
    order = ["C", "H"] + sorted(k for k in c if k not in ("C", "H"))
    return "".join(f"{e}{c[e] if c[e] > 1 else ''}" for e in order if e in c)


rows = []
for setname, names in (("baker", BAKER_NAMES), ("sharada", SHARADA_NAMES)):
    for i, (d, label) in enumerate(sorted(names.items()), start=1):
        base = os.path.join(ROOT, "data", setname, d)
        chg = int(open(os.path.join(base, "chg")).read().split()[0])
        mult = int(open(os.path.join(base, "mult")).read().split()[0])
        syms = read_xyz(os.path.join(base, "ts.xyz"))
        nel = sum(Z[s] for s in syms) - chg
        applies = (mult == 1 and nel % 2 == 0)
        rows.append({
            "set": setname,
            "reaction_id": i,
            "dir": d,
            "name": label,
            "natoms": len(syms),
            "formula": formula(syms),
            "charge": chg,
            "mult": mult,
            "n_electrons": nel,
            "rks_question_applies": int(applies),
        })

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

hdr = f"{'set':8} {'#':>2} {'dir':26} {'formula':10} {'nat':>3} {'chg':>3} {'mult':>4} {'nel':>4} {'RKS?':>4}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['set']:8} {r['reaction_id']:>2} {r['dir']:26} {r['formula']:10} "
          f"{r['natoms']:>3} {r['charge']:>3} {r['mult']:>4} {r['n_electrons']:>4} "
          f"{'yes' if r['rks_question_applies'] else 'NO':>4}")

n_app = sum(r["rks_question_applies"] for r in rows)
print(f"\ntotal rows {len(rows)}; RKS question applies to {n_app}; "
      f"open-shell/excluded {len(rows) - n_app}")
