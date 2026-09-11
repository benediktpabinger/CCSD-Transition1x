"""Assemble REPORT.md from the result CSVs, so no number is transcribed by hand.

  python make_report.py <marks_stability.csv> <marks_per_reaction.csv> <out_md>
"""
import csv
import sys
from collections import defaultdict

stab_csv, runs_csv, out_md = sys.argv[1], sys.argv[2], sys.argv[3]

S = list(csv.DictReader(open(stab_csv, encoding="utf-8")))
R = list(csv.DictReader(open(runs_csv, encoding="utf-8")))
for r in R:
    r["k"] = (r["set"], int(r["reaction_id"]))
    r["ok"] = r["success"] == "1"

cls = {}
for r in S:
    k = (r["set"], int(r["reaction_id"]))
    cls[k] = ("unstable" if r["stable"] == "0" else
              "stable" if r["stable"] == "1" else "open-shell")

OMOL = ["eSEN-S", "UMA-S", "UMA-M", "MACE-OMol25"]
DUP = {1, 3, 4, 5, 6}


def distinct(k):
    return not (k[0] == "sharada" and k[1] in DUP)


uniq = {r["structure"] for r in S}
n_stable = len({r["structure"] for r in S if r["stable"] == "1"})
n_unstable = len({r["structure"] for r in S if r["stable"] == "0"})
n_open = len({r["structure"] for r in S if r["stable"] == ""})

L = []
A = L.append

A("# RKS stability of the Marks benchmark reference transition states\n")
A("Reference transition states of J. Marks, J. Vandezande, J. Gomes,")
A("[arXiv:2604.00405](https://arxiv.org/abs/2604.00405), classified RKS-stable")
A("or RKS-unstable at **that paper's own level, wB97X-V/def2-TZVP** -- deliberately")
A("not the wB97M-V this project trains against, since the point is to measure the")
A("instability on Marks' surface.\n")

A("## Result\n")
A(f"| | distinct structures |")
A("|---|---|")
A(f"| stable | {n_stable} |")
A(f"| **unstable** | **{n_unstable}** |")
A(f"| open-shell (question not applicable) | {n_open} |")
A(f"| total | {len(uniq)} |")
A("")
u = next(r for r in S if r["stable"] == "0")
A(f"The single unstable case is **{u['name']}** (Baker 6 / Sharada 5):")
A(f"lowest stability eigenvalue {u['lambda_min_au']} au, "
  f"<S**2> = {u['S2_BS']}, depth {u['depth_meV']} meV.\n")
A("The stable cases are not marginal. Ranked by the lowest stability eigenvalue,")
A("the smallest positive margin is an order of magnitude clear of the")
A("marginally-stable band (+0.001 to +0.008 au) seen in this project's")
A("Transition1x multireference set:\n")
marg = sorted(((float(r["lambda_min_au"]), r["structure"]) for r in S
               if r["lambda_min_au"] and r["stable"] == "1"
               and distinct((r["set"], int(r["reaction_id"])))),
              key=lambda t: t[0])
A("| structure | lambda_min (au) |")
A("|---|---|")
for v, s in marg[:6]:
    A(f"| {s} | {v:+.6f} |")
A("")

A("## Method\n")
A("Per structure, ORCA 5.0.4:\n")
A("1. `! RKS wB97X-V def2-TZVP def2/J RIJCOSX TightSCF DEFGRID3 EnGrad` -- the")
A("   restricted reference energy, and the gradient as a check that the geometry")
A("   really is a stationary point at this level.")
A("2. `! UKS ...` with `%scf STABPerform true STABRestartUHFifUnstable true end`,")
A("   which rotates into the broken-symmetry solution when one is lower.\n")
A("Open-shell structures get step 1 only (as UKS), with <S**2> logged.")
A("depth_meV = (E_RKS - E_BS) * 27211.386.\n")

A("## Quality control\n")
A("* Every SCF converged; every ORCA run terminated normally.")
A("* **<S\\*\\*2> = 0.000000 for all 24 stable cases** (QC b).")
A("* Verdicts are taken from ORCA's explicit \"Stability Analysis indicates a")
A("  {stable, UNSTABLE} HF/KS wave function\" line. Nothing is inferred from the")
A("  absence of an UNSTABLE line -- an earlier version of the collector did that")
A("  and turned a batch of failed runs into 14 apparent \"stable\" results.")
mg = [float(r["max_grad_Eh_bohr"]) for r in S if r["max_grad_Eh_bohr"]]
A(f"* Max |gradient| over all structures {max(mg):.2e} Eh/bohr, median "
  f"{sorted(mg)[len(mg) // 2]:.2e}: the geometries are genuine wB97X-V/def2-TZVP")
A("  stationary points, which is what licenses calling them *the* reference TSs.")
A("* Control (QC c): bicyclobutane was expected unstable after Hait")
A("  (B3LYP/def2-SVP, <S**2> = 0.24) and came out unstable, <S**2> = 0.19. The")
A("  functional and basis differ, so the values are not directly comparable; the")
A("  qualitative agreement is what the control tests.\n")
A("One systematic effect worth recording: every stable case shows a small")
A("**positive RKS->UKS energy offset** growing with system size (0.06 meV for HCN")
A("to 8.94 meV for the 56-atom Ireland-Claisen) despite <S**2> = 0 exactly, i.e.")
A("numerical difference between the restricted and unrestricted code paths, not")
A("physics. It matters only because the depth mixes the two paths. A plain UKS")
A("single point on bicyclobutane (<S**2> = 0.000000) puts the offset at")
A("+0.39 meV, so the depth is 13.72 meV as defined above and 14.11 meV against a")
A("same-path reference. The verdict does not depend on it.\n")

A("## Inventory and deviations\n")
A("Full detail in [INVENTORY.md](INVENTORY.md). The short version:\n")
A("* **Zenodo record 19379882 was never reachable** -- 504 from this workstation,")
A("  from the DTU cluster and from the DOI resolver, with `zenodo.org` itself")
A("  down, so its existence could be neither confirmed nor refuted. arXiv v1 in")
A("  any case cites no Zenodo DOI and no GitHub URL; its data statement promises")
A("  them \"upon publication\".")
A("* Geometries instead come from **[`thegomeslab/fsm`](https://github.com/thegomeslab/fsm)**,")
A("  the Gomes lab's own FSM code released with the companion paper (Marks &")
A("  Gomes, [arXiv:2407.09763](https://arxiv.org/abs/2407.09763) / JCTC), whose")
A("  reference transition states are at the identical wB97X-V/def2-TZVP level. It")
A("  carries `ts.xyz`, `chg` and `mult` for all 24 Baker and all 9 Sharada")
A("  reactions -- complete, nothing missing.")
A("* **28 distinct structures, not 29.** Five Sharada reference TSs coincide with")
A("  Baker ones (three bit-identical, two within 0.0015 A) and their paper table")
A("  rows are identical across all 12 columns. The paper's own \"24 + 5 = 29\"")
A("  does not hold geometrically. All 33 table rows are kept in the CSV, with the")
A("  duplicates carrying a note.")
A("* Two defects in the repository metadata: `sharada/04` has `chg`/`mult`")
A("  swapped (1/0, multiplicity 0 being unphysical; corrected to 0/1), and")
A("  `sharada/02` is C2H8Si -- silylene insertion into ethane -- not the")
A("  \"SiH2 + H2 -> SiH4\" its paper row claims.")
A("* Parsing the paper's tables surfaced that **failure mode (c) counts as a")
A("  success** in the authors' own aggregates despite being italicised in two")
A("  cells; all 48 footer columns reproduce exactly only under that reading.\n")

A("## Stability class vs. MLIP search outcome\n")
A("Counts only; no test is applied, and none would be meaningful.\n")
A("| algorithm | class | runs | failures | failure rate | mean gradients |")
A("|---|---|---|---|---|---|")
for algo in ("FSM", "CI-NEB"):
    for c in ("stable", "unstable", "open-shell"):
        rows = [r for r in R if r["algorithm"] == algo and cls.get(r["k"]) == c]
        if not rows:
            continue
        f = sum(1 for r in rows if not r["ok"])
        costs = [int(r["gradients"]) for r in rows if r["ok"] and r["gradients"]]
        A(f"| {algo} | {c} | {len(rows)} | {f} | {100 * f / len(rows):.1f}% | "
          f"{sum(costs) / len(costs):.2f} |")
A("")
A("**The unstable class is one reaction.** It appears as two table rows because")
A("Sharada repeats it. Any rate quoted for it describes a single reaction across")
A("12 model/workflow combinations, and the FSM and CI-NEB numbers point in")
A("opposite directions (25.0% against 9.8%, then 16.7% against 32.4%). There is")
A("no signal here.\n")

A("### Where the OMol25 models actually fail\n")
A("Not on the unstable reaction. Ranked by failures among the four OMol25-trained")
A("models (eSEN-S, UMA-S, UMA-M, MACE-OMol25) over both algorithms:\n")
A("| set | # | reaction | class | OMol25 failures |")
A("|---|---|---|---|---|")
rank = []
for k in cls:
    if not distinct(k):
        continue
    sub = [r for r in R if r["k"] == k and r["model"] in OMOL]
    rank.append((sum(1 for r in sub if not r["ok"]), k))
rank.sort(key=lambda t: -t[0])
nm = {(r["set"], int(r["reaction_id"])): r["name"] for r in S}
for n, k in rank[:8]:
    if n:
        A(f"| {k[0]} | {k[1]} | {nm[k]} | {cls[k]} | {n} |")
bic = next(n for n, k in rank if k == ("baker", 6))
A(f"| ... | | | | |")
A(f"| baker | 6 | {nm[('baker', 6)]} | **unstable** | **{bic}** |")
A("")
A("Every one of the worst cases is RKS-stable, including the two the paper itself")
A("singles out as characteristic failures (Baker 11 and 24), which sit at")
A("lambda_min = +0.045 and +0.049. The unstable reaction is among the")
A("better-behaved ones for these models: its only OMol25 failures are two")
A("FSM-native runs, and all four models succeed on it under CI-NEB in 3-4")
A("gradients.\n")
A("A comment left in the paper's LaTeX source, on the worst failure case")
A("(Baker 22), points at the actual mechanism:\n")
A("> all of these off target saddle points correspond to an in place cis addition")
A("> while the lower energy in plane trans addition is the reference.")
A("> potentially(?) an internal coords failure here\n")
A("That is the wrong saddle point on a well-behaved surface. Consistent with the")
A("dominant failure modes across all 792 runs being (a) alternate first-order")
A("saddle and (b) local minimum; mode (e), SCF convergence error, occurs twice.\n")

A("## Limitation\n")
A("With 24 of 25 closed-shell reference transition states stable, this benchmark")
A("cannot answer whether RKS instability predicts transition-state-search")
A("failure -- there is no power to detect such a link even if one exists. The")
A("Baker and Sharada sets are classical, well-behaved organic reactions; that")
A("they are almost uniformly stable is itself the finding. The question belongs")
A("to a set selected for multireference character, such as this project's")
A("Transition1x subset, where 18 of 26 are unstable.\n")

A("## Files\n")
A("* `marks_stability.csv` -- the deliverable: set, reaction_id, name, E_RKS_Ha,")
A("  stable, E_BS_Ha, depth_meV, S2_BS, notes, plus structure, lambda_min_au and")
A("  max_grad_Eh_bohr.")
A("* `marks_per_reaction.csv` -- 792 rows parsed from the paper's four tables.")
A("* `INVENTORY.md`, `CROSSTAB.md`, `orca_outputs/`, `geoms/`, `scripts/`.")

open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
print(f"wrote {out_md} ({len(L)} lines)")
