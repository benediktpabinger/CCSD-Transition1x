# MACE+delta NEB Benchmark

> Rewritten 2026-06-22 with the fw=2.0 delta head and the full delta-correction
> ablation investigation. Supersedes the earlier fw=1.0 preliminary version of
> this document.

---

## What this is

NEB run using the MACE+delta head as the ASE calculator, benchmarked against
the 30-reaction multireference benchmark (10 High/Mid/Low MR each, by FOD
rank). The calculator wraps frozen MACE (wB97X-D3/6-31G(d)) + a delta
correction head trained to predict δ = E(wB97M-V/def2-TZVP) −
E(wB97X-D3/6-31G(d)), giving a wB97M-V-level PES at MACE inference speed.

**The key question this document answers**: does the delta correction
actually improve NEB-derived transition-state geometries relative to bare
(uncorrected) MACE? The short answer is **no** — and the investigation below
explains why, mechanistically.

See [delta_head.md](delta_head.md) for full architecture/training history and
[delta_head_data_and_training_summary.md](delta_head_data_and_training_summary.md)
for a verified-against-logs summary of data selection and training
hyperparameters.

---

## Delta head used here

| Parameter | Value |
|-----------|-------|
| Backbone | `mace_t1x_p10_compiled.model` (frozen) |
| Head architecture | `NonLinearReadoutBlock`, MLP_irreps=64x0e, 65,600 params |
| Training data | 80,592 geoms from 4,997 Transition1x train reactions, stratified TS-centered sampling |
| Force weight | **2.0** (`delta_head_fw2.00.pt`) — selected from sweep 0.5/1.0/2.0 by validation force loss |
| Val energy / force loss (fw=2.0) | 0.0112 / 0.0037 |

---

## Benchmark setup

30 reactions from the Transition1x test set, split by multireference
character (FOD rank):

- **High MR** (n=10): rxn7949, rxn8832, rxn1320, rxn4113, rxn8885, rxn7945, rxn7937, rxn6196, rxn0346, rxn1150
- **Mid MR** (n=10): rxn0896, rxn1154, rxn5690, rxn4513, rxn7955, rxn4519, rxn4500, rxn2553, rxn8829, rxn1155
- **Low MR** (n=10): rxn9246, rxn4498, rxn1061, rxn4003, rxn4004, rxn4063, rxn4114, rxn4060, rxn1961, rxn1962

**NEB protocol** (all methods): 10-image starting band from Transition1x.h5
wB97X-D3 NEB (R + last 8 interior + P), BFGS endpoint relaxation (fmax=0.05
eV/Å), plain NEB → CI-NEB (improvedtangent), fmax=0.05 eV/Å, max 500 steps
each.

**Reference**: ORCA wB97M-V/def2-TZVP NEB (`pipeline/orca_neb.py`).

---

## Part 1 — Does the delta correction improve NEB geometry vs bare MACE?

To isolate the correction's effect, `mace_delta_neb.py` was extended with a
`--no-delta` flag (bare MACE only, no correction head loaded) and rerun on
all 30 reactions (`mace_bare_neb_results/`, job 10546176 — all 30/30
completed cleanly).

### RMSD vs ORCA wB97M-V NEB (Å), by MR category

| MR | bare MACE | MACE+delta fw2.0 | winner |
|---|---|---|---|
| High | **0.061** | 0.134 | bare MACE, 2.2× better |
| Mid | **0.092** | 0.116 | bare MACE |
| Low | **0.015** | 0.053 | bare MACE, 3.5× better |
| All | **0.056** | 0.101 | bare MACE |

Bare MACE wins on **28/30 individual reactions** — only rxn1320 and rxn1154
go the other way.

### Forward barrier MAE vs wB97M-V NEB (meV), by MR category

| MR | bare MACE | MACE+delta fw2.0 |
|---|---|---|
| High | **94** | 272 |
| Mid | 234 | **159** |
| Low | 166 | **89** |
| All | **165** | 174 |

Mixed at the barrier-energy level (delta helps Mid/Low, bare MACE wins
decisively on High-MR), but RMSD is the more fundamental metric — it tells
you whether the TS is even the right structure before any energy correction
is applied on top.

### RMSD vs CASSCF(AVAS)+NEVPT2 OptTS gold standard (Å), n=9 valid High-MR reactions

> **Superseded by Part 3** — this preliminary table covered only 9 of the 10
> High(orig) reactions. Part 3 expands to all 23 reactions with the
> reliability classification applied. Numbers below are left for reference.

| Method | Mean RMSD (n=9) |
|---|---|
| ORCA wB97M-V | 0.118 |
| **bare MACE** | **0.141** |
| UMA-S | 0.144 |
| UMA-M | 0.182 |
| MACE+delta fw2.0 | 0.206 |
| eSEN | 0.286 |

---

## Part 2 — Why does the correction hurt geometry despite being more accurate?

This is the counterintuitive part: the same fw=2.0 delta head, evaluated as a
**single point** on fixed, already-converged geometries, is *much* more
accurate than bare MACE:

### Single-point energy/force MAE vs wB97M-V ground truth (300 fixed ORCA-NEB geometries)

| | wB97X-D3 true | bare MACE | MACE+delta fw2.0 |
|---|---|---|---|
| Energy MAE (meV) | 94.9 | 107.6 | **63.7** |
| Force MAE (meV/Å) | 139.9 | 138.5 | **76.0** |

So the correction roughly **halves** both energy and force error at fixed
geometries, yet produces **worse** geometries when used to drive an actual
NEB search. Two hypotheses were tested:

### Hypothesis A: delta forces are non-conservative (inconsistent with delta energy)

Tested by comparing the calculator's analytic (autograd) forces against a
finite-difference estimate of its own energy function
(`pipeline/_check_force_conservativeness.py`), at increasing step sizes to
separate real inconsistency from float32 rounding noise:

| eps (Å) | bare MACE max\|F_analytic − F_fd\| | MACE+delta max\|F_analytic − F_fd\| |
|---|---|---|
| 0.005 | 0.082 | 0.082 |
| 0.010 | 0.043 | 0.043 |
| 0.020 | 0.020 | 0.020 |

Both calculators show **identical** residual disagreement at every step
size, shrinking as the step grows (consistent with float32 noise in the
absolute ~−8800 eV total energy dominating at small steps, not a real
inconsistency). **Hypothesis A is ruled out** — the delta head's forces are
just as conservative as MACE's own.

### Hypothesis B: the delta-corrected surface is rougher, making optimization harder

Tested by comparing NEB iteration counts needed to reach convergence, bare
MACE vs MACE+delta, across all 30 reactions
(`pipeline/_compare_neb_convergence.py`):

| Metric | Value |
|---|---|
| Reactions where delta needed *more* iterations | 20/30 |
| Reactions needing ≥5× more iterations | 8/30 |
| Reactions needing ≥10× more iterations | 6/30 |
| Most extreme case (rxn1962) | bare: 1 iteration → delta: **221 iterations** |
| Mean ratio (delta/bare) | 15.9× |
| Mean ratio by MR — High / Mid / Low | 4.8× / 3.5× / **39.5×** |

The effect is **most dramatic for Low-MR reactions** — the "easy" cases
bare MACE converges almost instantly (1–4 steps, the geometry was already
essentially right) but where the delta-corrected surface turns convergence
into a 58–472-step struggle. **This supports Hypothesis B.**

### Conclusion

The delta head doesn't break energy conservation, but it does measurably
roughen the global optimization landscape — plausibly because it's a small,
separately-trained MLP fit to **discretely sampled** points (20 stratified
geometries per reaction) rather than the densely-sampled continuous paths
MACE itself was trained on (~950 points/reaction). Between/around the
sampled points, the head's surface has no smoothness guarantee. NEB still
eventually satisfies its fmax convergence criterion on this bumpier surface,
but it's likely settling at a different — and per the RMSD data, usually
worse — nearby stationary point than the true smooth saddle, not the
accurate one the SP-level numbers would suggest.

---

## Part 3 — Full MR-Optimized benchmark (n=23, reliability-filtered)

> Expands the n=9 preliminary table in Part 1 to the complete 23-reaction
> CASSCF(AVAS)+NEVPT2 OptTS benchmark (10 High(orig) + 13 next-HIGH
> reactions). A reliability classification is applied; see
> `active_space_quality_analysis.md` for full details including NOON data,
> active-space sizes, and per-reaction remarks.

### Setup

> **Different reaction set from Part 1.** Part 1 uses 30 reactions (High/Mid/Low
> MR by FOD rank). Part 3 uses 23 reactions from FOD ranks 1–26 (top-10 + next-HIGH
> 11–26) — not the Mid or Low groups. Mid/Low were excluded because Step 4 showed
> CASSCF active spaces are unreliable there (0/10 Low-MR reliable, 4/10 Mid-MR
> reliable). See `multireference_screening.md` Step 5 for the full rationale.

26 reactions attempted; 23 converged (rxn5691, rxn1283, rxn0894 failed).
CASSCF+NEVPT2 settings: threshold=0.4, `--no-prune`, basis def2-TZVP,
AVAS AO targets `['C 2pz', 'N 2p', 'O 2pz', 'F 2pz']`.

The **TS-RMSD** metric is the Kabsch-aligned RMSD between each method's NEB
transition state and the CASSCF-optimized TS. Three reactions
(**excl-geo**: rxn4518, rxn0101, rxn4522, rxn10054) are excluded from
aggregate statistics because CASSCF found a wrong saddle point — RMSD
against such a reference would penalise methods for correctly finding the
DFT TS. For rxn10054, the evidence is the negative NEVPT2 barrier (−30 meV)
rather than a large RMSD (0.328 Å).

### Reliability summary

| Class | n | Reactions | Notes |
|-------|---|-----------|-------|
| reliable | 11 | rxn7949, rxn8832, rxn8885, rxn7945, rxn6196, rxn3107, rxn7936, rxn7957, rxn7937†, rxn0346†, rxn7060† | Intruder validated by CCSD(T) (Δ < 150 meV) or no intruder |
| reliable\* | 3 | rxn1320, rxn1147, rxn1150 | Geometry trustworthy; NEVPT2 barrier biased (0@R) — use CCSD(T) |
| caveat | 3 | rxn0896, rxn8827, rxn10005 | Intruder present, |Δ(CCSD(T)−NEVPT2)| > 200 meV |
| borderline | 1 | rxn8837 | Large ORCA vs OptTS RMSD (0.43 Å); CASSCF converged |
| excl-geo | 4 | rxn4518, rxn0101, rxn4522, rxn10054 | CASSCF found wrong saddle point — excluded |
| failed | 3 | rxn5691, rxn1283, rxn0894 | OptTS did not converge |

### TS-RMSD vs CASSCF OptTS (Å) — per reaction

| rxn | class | ORCA | T1x | MACE bare | MACE+delta | UMA-s | UMA-m | eSEN |
|-----|-------|------|-----|-----------|------------|-------|-------|------|
| rxn7949 | reliable | 0.072 | 0.074 | 0.184 | 0.128 | 0.169 | 0.211 | 0.206 |
| rxn8832 | reliable | 0.287 | 0.292 | 0.291 | 0.274 | 0.169 | 0.129 | 0.152 |
| rxn8885 | reliable | 0.151 | 0.153 | 0.153 | 0.152 | 0.415 | 0.104 | **1.365** |
| rxn7945 | reliable | 0.052 | 0.059 | 0.083 | 0.297 | 0.087 | 0.077 | 0.399 |
| rxn6196 | reliable | 0.079 | 0.081 | 0.095 | 0.273 | 0.123 | 0.129 | 0.129 |
| rxn3107 | reliable | 0.106 | 0.105 | 0.112 | 0.107 | 0.095 | 0.120 | 0.124 |
| rxn7936 | reliable | 0.070 | 0.071 | 0.099 | 0.090 | 0.070 | 0.071 | 0.070 |
| rxn7957 | reliable | 0.074 | 0.071 | 0.162 | 0.202 | 0.239 | 0.250 | 0.239 |
| rxn1320 | reliable\* | 0.195 | 0.194 | 0.160 | 0.184 | 0.291 | 0.292 | 0.299 |
| rxn1147 | reliable\* | 0.070 | 0.074 | 0.125 | **0.482** | 0.345 | 0.331 | 0.337 |
| rxn1150 | reliable\* | 0.161 | 0.147 | 0.154 | FRAG | 0.167 | 0.165 | 0.164 |
| rxn7937 | caveat | 0.048 | 0.054 | 0.099 | 0.200 | 0.079 | 0.069 | 0.078 |
| rxn0346 | caveat | 0.153 | 0.153 | 0.153 | 0.152 | 0.036 | 0.037 | 0.040 |
| rxn0896 | caveat | 0.230 | 0.229 | 0.181 | 0.141 | 0.228 | 0.235 | 0.229 |
| rxn8827 | caveat | 0.064 | 0.064 | 0.123 | 0.115 | 0.171 | 0.195 | 0.217 |
| rxn10005 | caveat | 0.247 | 0.247 | 0.274 | 0.277 | 0.246 | 0.248 | 0.248 |
| rxn7060 | reliable | 0.197 | 0.208 | 0.245 | 0.248 | 0.186 | 0.197 | 0.185 |
| rxn8837 | borderline | 0.426 | 0.430 | 0.277 | 0.294 | FRAG | FRAG | **1.724** |
| rxn10054 | excl-geo | *(0.328)* | *(0.329)* | *(0.330)* | *(0.332)* | *(0.342)* | *(0.339)* | *(0.339)* |
| rxn4518 | excl-geo | *(0.653)* | *(0.653)* | *(0.734)* | *(0.739)* | *(1.065)* | *(1.046)* | *(1.084)* |
| rxn0101 | excl-geo | *(0.711)* | *(0.714)* | *(0.652)* | *(0.680)* | *(0.659)* | *(0.672)* | *(0.671)* |
| rxn4522 | excl-geo | *(0.858)* | *(0.858)* | *(0.643)* | *(0.623)* | *(0.682)* | *(0.691)* | *(0.681)* |
| rxn4113 | excl-nevpt2 | *(0.056)* | *(0.060)* | *(0.061)* | *(0.065)* | *(0.050)* | *(0.718)* | *(0.046)* |

FRAG = NEB produced a fragmented/unphysical path. Values in italics are excluded from aggregate statistics.

### Aggregate TS-RMSD (Å)

| Filter | n | ORCA | T1x | MACE bare | MACE+delta | UMA-s | UMA-m | eSEN |
|--------|---|------|-----|-----------|------------|-------|-------|------|
| Reliable only | 11 | **0.117** | 0.120 | 0.152 | 0.193 | 0.152 | 0.127 | 0.271 |
| Reliable + reliable\* | 14 | **0.122** | 0.124 | 0.151 | 0.215 | 0.177 | 0.156 | 0.270 |
| Rel + rel\* + caveat | 17 | **0.133** | 0.134 | 0.158 | 0.208 | 0.183 | 0.168 | 0.264 |
| All excl-geo (19) | 19 | **0.144** | 0.146 | 0.160 | 0.205 | 0.176 | 0.199 | 0.329 |

On the **reliable** subset (cleanest reference, n=11), the geometry ranking is:

> ORCA (0.109) ≈ T1x (0.111) < UMA-m (0.120) < **MACE bare (0.143)** < UMA-s (0.148) < MACE+delta (0.188) ≪ eSEN (0.280)

UMA-m and MACE bare are the best MLIPs for geometry on the reliable set. On the broader 20-reaction set the order is similar (MACE bare 0.168, UMA-m 0.206 — MACE bare wins there).

eSEN's high RMSD is dominated by two catastrophic failures: rxn8885 (1.365 Å) and rxn8837 (1.724 Å). MACE+delta's elevated RMSD vs bare MACE is consistent with the Part 2 finding — the delta head roughens the optimization landscape and drives the NEB to a different nearby stationary point. The most extreme case is rxn1147 (MACE+delta RMSD = 0.482 Å vs bare MACE 0.125 Å).

### Barrier comparison vs NEVPT2(OptTS) (meV)

The **barrier MAE** is measured vs the NEVPT2(OptTS) forward barrier. Four
reactions are excluded because their NEVPT2 reference is unreliable
(BARRIER_UNRELIABLE):
rxn1150 (0@R → NEVPT2 1679 vs CCSD(T) 3460 meV, Δ = +1781),
rxn4113 (0@R → CCSD(T) 5346 meV),
rxn1147 (0@R → NEVPT2 2114 vs CCSD(T) 4023 meV, Δ = +1909),
rxn0896 (intruder → NEVPT2 2484 vs CCSD(T)@OptTS 4548 meV, Δ = +2064).
rxn10054 is already excluded as excl-geo (negative NEVPT2 barrier → wrong
saddle point). Rows marked ‡ appear for transparency but are excluded from
MAE aggregates.

| rxn | class | NEVPT2 ref | ORCA | MACE bare | MACE+delta | UMA-s | UMA-m | eSEN |
|-----|-------|-----------|------|-----------|------------|-------|-------|------|
| rxn7949 | reliable | 4812 | −856 | −955 | −1166 | −2015 | −2045 | −2060 |
| rxn8832 | reliable | 2340 | +866 | +585 | +403 | +152 | +134 | +187 |
| rxn8885 | reliable | 3709 | −102 | −50 | +24 | −466 | −137 | −446 |
| rxn7945 | reliable | 3920 | −20 | −82 | −504 | −51 | −36 | −533 |
| rxn6196 | reliable | 4346 | −91 | +12 | −249 | −122 | −135 | −125 |
| rxn3107 | reliable | 4743 | −603 | −500 | −592 | −654 | −650 | −648 |
| rxn7936 | reliable | 6127 | −314 | −827 | −516 | −335 | −332 | −329 |
| rxn7957 | reliable | 3023 | +930 | +694 | +521 | −107 | −106 | −76 |
| rxn1320 | reliable\* | 3872 | −465 | −511 | −699 | −1106 | −1101 | −1104 |
| rxn1147 | reliable\*‡ | 2114 | +2080 | +1955 | +1592 | +1762 | +1766 | +1775 |
| rxn7937 | reliable | 3808 | +20 | +41 | −113 | +5 | +6 | −5 |
| rxn0346 | reliable | 3237 | +314 | +428 | +200 | +88 | +72 | +94 |
| rxn0896 | caveat‡ | 2484 | +2737 | +2104 | +2129 | +2721 | +2720 | +2725 |
| rxn8827 | caveat | 4003 | −141 | −265 | −628 | −547 | −547 | −516 |
| rxn10005 | caveat | 3452 | +270 | +296 | +93 | +267 | +271 | +272 |
| rxn7060 | reliable | 3919 | +2245 | +1524 | +1225 | +2218 | +2231 | +2203 |
| rxn8837 | borderline | 3842 | +384 | −85 | −362 | FRAG | FRAG | +751 |

‡ NEVPT2 reference unreliable (large CCSD(T) discrepancy); excluded from MAE statistics.

#### Barrier MAE summary (meV)

| Filter | n | ORCA | MACE bare | MACE+delta | UMA-s | UMA-m | eSEN |
|--------|---|------|-----------|------------|-------|-------|------|
| Reliable only (11) | 11 | 578 | 518 | **501** | 565 | 535 | 610 |
| Reliable + reliable\* (12) | 12 | 569 | **518** | **518** | 610 | 582 | 651 |
| All valid (15) | 15 | 508 | **457** | 486 | 581 | 557 | 623 |

All methods show large MAEs relative to NEVPT2(OptTS), including ORCA DFT
(578 meV on the reliable subset). This is expected and meaningful: the
reactions were specifically selected for high multireference character, and
the NEVPT2(OptTS) reference uses a CASSCF-optimized geometry that may differ
from the DFT NEB saddle point. The error therefore conflates two
contributions — (1) electronic structure error (DFT vs CASSCF+NEVPT2) and
(2) geometry error (NEB TS vs CASSCF OptTS). For the methods with high RMSD
(notably MACE+delta and the UMA models on rxn1147; eSEN on rxn8885), the
geometry error is the dominant contribution.

Two reactions appear in the table for transparency but are excluded from all
MAE aggregates because their NEVPT2 reference is confirmed unreliable by
CCSD(T):

- **rxn0896** (caveat‡): every method predicts a barrier ~2700 meV above
  NEVPT2(OptTS) = 2484 meV. CCSD(T)@OptTS = 4548 meV (Δ = +2064 meV)
  confirms the NEVPT2 reference is badly wrong due to an intruder orbital;
  all errors relative to this reference are uninformative.

- **rxn1147** (reliable\*‡): CCSD(T) = 4023 meV vs NEVPT2(OptTS) = 2114 meV
  (Δ = +1909 meV). Classic 0@R failure: active space idle at R → near-zero
  NEVPT2 correction at R → systematically underestimated barrier.

**rxn7060** (now reliable†) also shows large barrier errors: ORCA +2245,
MACE bare +1524, MACE+delta +1225 meV. These are genuine errors against a
validated NEVPT2 reference (CCSD(T) Δ = +97 meV) — consistent with a
DFT-optimised NEB TS sitting at a different point on the CASSCF surface.

On the clean reliable subset (n=11), the barrier ranking is:
> MACE+delta (501) < MACE bare (518) < UMA-m (535) < UMA-s (565) < ORCA (578) < eSEN (610) meV

MACE+delta gives the lowest barrier MAE on the reliable subset, followed
closely by MACE bare — both outperforming ORCA DFT (578 meV). MACE bare
errors are systematically in the same direction as DFT, consistent with the
expected single-reference bias rather than a random ML failure.

---

## Recommendation

**Decouple geometry search from energy correction.** Run NEB with bare MACE
forces only (smooth, reliable, demonstrably good geometry — competitive with
UMA-S even against the CASSCF+NEVPT2 gold standard), then apply the delta
head only as a **post-hoc single-point energy correction** on the converged
geometry — never as a force field driving the search itself. This combines
bare MACE's reliable geometry-finding with the delta head's genuinely better
pointwise energy accuracy, without letting the correction's roughness
destabilize the optimization.

This has not yet been benchmarked end-to-end (i.e., bare-MACE-NEB geometry +
delta-corrected barrier vs CCSD(T)/wB97M-V) — a natural next step.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `pipeline/mace_delta_neb.py` | MACE+delta ASE calculator + NEB pipeline; `--no-delta` flag runs bare MACE only |
| `pipeline/job_mace_delta_neb_fw2.sh` | SLURM array (30 reactions, h200), MACE+delta fw=2.0 |
| `pipeline/job_mace_bare_neb.sh` | SLURM array (30 reactions, h200), bare MACE ablation |
| `pipeline/_collect_bare_mace_and_refresh.py` | Computes bare-MACE NEB barriers, refreshes ablation RMSD plot + MR-Optimized comparison |
| `pipeline/_check_force_conservativeness.py` | Finite-difference vs analytic force consistency check |
| `pipeline/_compare_neb_convergence.py` | Compares NEB iteration counts, bare vs delta, all 30 reactions |
| `pipeline/delta/eval_benchmark_sp_fw2.py` | Single-point energy/force MAE eval with current fw=2.0 head |
| `pipeline/uma_neb.py`, `pipeline/uma_m_neb.py` | UMA-S / UMA-M NEB baselines |
| `pipeline/esen_neb.py` | eSEN NEB baseline |

**Result files**: `mace_bare_neb_results/`, `mace_delta_neb_results_fw2/`,
`benchmark_plots/delta_ablation_rmsd.png`, `benchmark_plots/mr_optimized_rmsd_all10.png`,
`eval_benchmark_sp_fw2.json`, `full_benchmark_results.json`
