"""Collect the ORCA stability runs into marks_stability.csv.

  python collect_results.py <orca_outputs_dir> <out_csv>

One row per paper table row -- all 24 Baker and all 9 Sharada -- so the five
Sharada reactions whose reference TS is the same structure as a Baker one carry
that structure's numbers plus a note saying so.

Nothing is inferred from absence. A verdict is only written when ORCA printed
"Stability Analysis indicates a {stable,UNSTABLE} HF/KS wave function"; anything
else becomes a note and an empty verdict. An earlier version read a missing
UNSTABLE line as "stable", which turned a batch of failed runs into 14 apparent
results.
"""
import csv
import os
import re
import sys

HARTREE_TO_MEV = 27211.386

BAKER = ["01_hcn", "02_hcch", "03_h2co", "04_ch3o", "05_cyclopropyl",
         "06_bicyclobutane", "08_formyloxyethyl", "09_parentdielsalder",
         "10_tetrazine", "11_trans_butadiene", "12_ethane_h2_abstraction",
         "13_hf_abstraction", "14_vinyl_alcohol", "15_hocl", "16_h2po4_anion",
         "17_claisen", "18_silylene_insertion", "19_hnccs", "20_hconh3_cation",
         "21_acrolein_rot", "22_hconhoh", "23_hcn_h2", "24_h2cnh", "25_hcnh2"]

# Sharada row -> structure actually computed, and why when it is a Baker one.
SHARADA_MAP = [
    ("01_formaldehyde", "baker_03_h2co",
     "same reference TS as Baker 3 (max |dx| 0.0014 A)"),
    ("02_silane", "sharada_02_silane", ""),
    ("03_ethanal", "baker_14_vinyl_alcohol",
     "identical reference TS to Baker 13"),
    ("04_ethane_dehydrogenation", "baker_12_ethane_h2_abstraction",
     "identical reference TS to Baker 11; repo chg/mult swapped (1/0), corrected to 0/1"),
    ("05_bicyclobutane", "baker_06_bicyclobutane",
     "same reference TS as Baker 6 (max |dx| 0.00004 A)"),
    ("06_diels_alder", "baker_09_parentdielsalder",
     "identical reference TS to Baker 8"),
    ("07_hexadiene", "sharada_07_hexadiene", ""),
    ("08_alanine", "sharada_08_alanine", ""),
    ("09_icr", "sharada_09_icr", ""),
]

STABLE_RE = re.compile(r"Stability Analysis indicates a stable HF/KS wave function")
UNSTABLE_RE = re.compile(r"Stability Analysis indicates an UNSTABLE HF/KS wave function")
ROOT0_RE = re.compile(r"Root\s+Eigenvalue \(au\)\s*\n\s*0\s+(-?\d+\.\d+)")
S2_RE = re.compile(r"Expectation value of <S\*\*2>\s*:\s*(-?\d+\.\d+)")
E_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")


def read(p):
    return open(p, errors="replace").read() if os.path.exists(p) else ""


def max_grad(path):
    if not os.path.exists(path):
        return None
    ls = [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]
    if len(ls) < 2:
        return None
    nat = int(ls[0])
    v = [float(x) for x in ls[2:2 + 3 * nat]]
    return max(abs(x) for x in v) if len(v) == 3 * nat else None


def analyse(base, rid, mult):
    d = os.path.join(base, rid)
    ref, stab = read(os.path.join(d, "ref.out")), read(os.path.join(d, "stab.out"))
    r = {"E_RKS_Ha": "", "stable": "", "E_BS_Ha": "", "depth_meV": "",
         "S2": "", "lambda_min": "", "max_grad": "", "notes": []}

    if not ref:
        r["notes"].append("no reference output")
        return r
    if "ORCA TERMINATED NORMALLY" not in ref:
        r["notes"].append("reference run did not terminate normally")
    if "SCF NOT CONVERGED" in ref:
        r["notes"].append("reference SCF not converged")
    e = E_RE.findall(ref)
    if not e:
        r["notes"].append("reference run produced no energy")
        return r
    r["E_RKS_Ha"] = float(e[-1])

    g = max_grad(os.path.join(d, "ref.engrad"))
    if g is not None:
        r["max_grad"] = g
        if g > 1e-3:
            r["notes"].append(f"max |grad| {g:.2e} Eh/bohr: not a tight stationary point")

    if mult != 1:
        s2 = S2_RE.findall(ref)
        r["S2"] = float(s2[-1]) if s2 else ""
        r["notes"].insert(0, "open-shell, RKS question not applicable")
        return r

    if not stab:
        r["notes"].append("no stability output")
        return r
    if "ORCA TERMINATED NORMALLY" not in stab:
        r["notes"].append("stability run did not terminate normally")
    if "SCF NOT CONVERGED" in stab:
        r["notes"].append("stability SCF not converged")

    m = ROOT0_RE.search(stab)
    if m:
        r["lambda_min"] = float(m.group(1))

    s2 = S2_RE.findall(stab)
    s2v = float(s2[-1]) if s2 else None
    es = E_RE.findall(stab)

    is_stable, is_unstable = bool(STABLE_RE.search(stab)), bool(UNSTABLE_RE.search(stab))
    if is_stable == is_unstable:
        r["notes"].append("no unambiguous stability verdict in output")
        return r

    if is_unstable:
        r["stable"] = 0
        if not es:
            r["notes"].append("instability reported but no broken-symmetry energy")
            return r
        r["E_BS_Ha"] = float(es[-1])
        r["S2"] = s2v if s2v is not None else ""
        r["depth_meV"] = (r["E_RKS_Ha"] - r["E_BS_Ha"]) * HARTREE_TO_MEV
        if r["depth_meV"] < -1e-3:
            r["notes"].append("broken-symmetry solution lies ABOVE the restricted one")
        # E_RKS and E_BS come from different code paths, and every stable case
        # shows a small positive RKS->UKS offset that grows with system size.
        # Where a plain UKS single point exists, quote the same-path depth too.
        uks = ""
        for cand in sorted(os.listdir(base)):
            if cand.startswith("uks_ref_") and cand[len("uks_ref_"):] in rid:
                uks = read(os.path.join(base, cand, "uks.out"))
                break
        if uks and "ORCA TERMINATED NORMALLY" in uks:
            eu = E_RE.findall(uks)
            s2u = S2_RE.findall(uks)
            if eu and s2u and abs(float(s2u[-1])) < 1e-6:
                off = (float(eu[-1]) - r["E_RKS_Ha"]) * HARTREE_TO_MEV
                same = (float(eu[-1]) - r["E_BS_Ha"]) * HARTREE_TO_MEV
                r["notes"].append(
                    f"RKS->UKS numerical offset {off:+.2f} meV; depth against a "
                    f"same-path UKS reference (<S**2>=0) is {same:.2f} meV")
    else:
        r["stable"] = 1
        r["S2"] = s2v if s2v is not None else ""
        # QC (b): a stable closed-shell singlet must have <S^2> exactly 0.
        if s2v is None:
            r["notes"].append("stable but <S**2> not reported")
        elif abs(s2v) > 1e-6:
            r["notes"].append(f"stable but <S**2> = {s2v:.6f} != 0; rerun")
        # the unrestricted run must land on the same solution as the restricted one
        if es and abs(float(es[-1]) - r["E_RKS_Ha"]) > 1e-6:
            r["notes"].append(f"stable but UKS energy differs from RKS by "
                              f"{(float(es[-1]) - r['E_RKS_Ha']) * HARTREE_TO_MEV:.2f} meV")
        if r["lambda_min"] != "" and r["lambda_min"] < 0:
            r["notes"].append("stable verdict but lowest eigenvalue is negative")
    return r


def main():
    base, out = sys.argv[1], sys.argv[2]
    inv = {}
    invcsv = os.path.join(os.path.dirname(os.path.abspath(out)), "inventory.csv")
    if os.path.exists(invcsv):
        for row in csv.DictReader(open(invcsv, encoding="utf-8")):
            inv[(row["set"], row["dir"])] = row

    cache = {}

    def get(rid, mult):
        if rid not in cache:
            cache[rid] = analyse(base, rid, mult)
        return cache[rid]

    rows = []
    for i, d in enumerate(BAKER, start=1):
        meta = inv.get(("baker", d), {})
        mult = int(meta.get("mult", 1) or 1)
        res = get(f"baker_{d}", mult)
        rows.append(("baker", i, meta.get("name", d), f"baker_{d}", res,
                     list(res["notes"])))
    for i, (d, sid, note) in enumerate(SHARADA_MAP, start=1):
        meta = inv.get(("sharada", d), {})
        smeta = next((v for (s, dd), v in inv.items() if f"{s}_{dd}" == sid), {})
        mult = int(smeta.get("mult", 1) or 1)
        res = get(sid, mult)
        rows.append(("sharada", i, meta.get("name", d), sid, res,
                     ([note] if note else []) + list(res["notes"])))

    def f(v, n):
        return "" if v == "" or v is None else f"{v:.{n}f}"

    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "reaction_id", "name", "E_RKS_Ha", "stable", "E_BS_Ha",
                    "depth_meV", "S2_BS", "notes", "structure",
                    "lambda_min_au", "max_grad_Eh_bohr"])
        for setn, rid_, name, sid, res, notes in rows:
            w.writerow([setn, rid_, name, f(res["E_RKS_Ha"], 9), res["stable"],
                        f(res["E_BS_Ha"], 9), f(res["depth_meV"], 2),
                        f(res["S2"], 6), "; ".join(notes), sid,
                        f(res["lambda_min"], 6), f(res["max_grad"], 8)])

    uniq = {sid: res for _, _, _, sid, res, _ in rows}
    ns = sum(1 for r in uniq.values() if r["stable"] == 1)
    nu = sum(1 for r in uniq.values() if r["stable"] == 0)
    no = sum(1 for r in uniq.values() if r["stable"] == "")
    print(f"wrote {out}")
    print(f"{len(rows)} table rows over {len(uniq)} distinct structures: "
          f"{ns} stable, {nu} unstable, {no} open-shell/no verdict")
    mg = [r["max_grad"] for r in uniq.values() if r["max_grad"] != ""]
    if mg:
        print(f"max |gradient| over all structures: {max(mg):.2e} Eh/bohr "
              f"(median {sorted(mg)[len(mg) // 2]:.2e})")
    print("\nflags:")
    any_flag = False
    for setn, rid_, name, sid, res, notes in rows:
        real = [n for n in notes if not n.startswith(("same reference", "identical reference"))]
        if real:
            any_flag = True
            print(f"  {setn} {rid_:2d} {sid:32} {'; '.join(real)}")
    if not any_flag:
        print("  none")


if __name__ == "__main__":
    main()
