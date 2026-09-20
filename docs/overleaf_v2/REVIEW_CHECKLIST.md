# Review checklist (check of 2026-09-20)

Working copy: `docs/overleaf_v2/`. `docs/overleaf/` stays the untouched mirror of Overleaf.
[V] = verified by Claude at the source or in `results/`; otherwise a checker's finding.
Tick items as they are applied.

## Tier 1: affects the argument

- [x] **1. "Tested nowhere" is false** [V]. APPLIED 2026-09-20 (approved): abstract, preface 60-62,
  intro 205-208 and the Hait/Marks paragraph 220-235 (now two paragraphs), conclusion 243-244 ->
  "not tested systematically"; Hait's bicyclobutane case now mentioned in the intro. Hait et al. Sec. IV.1: eSEN guess converges to the
  unrestricted bicyclobutane TS in 5 P-RFO iterations (FSM: 53), <S2> = 0.24; butadiene + ethene
  eSEN paths break symmetry. Affected: Abstract 41-42, Preface 60-61, Intro 205-208, 225-227,
  Conclusion 232-233, 243-244. Reword to "not tested systematically" and discuss their result.
- [x] **2. Ch. 3 barrier results rest on unconverged searches** [V, results/neb_runs.csv].
  APPLIED 2026-09-20, text-only + figure: 03 convergence paragraph scoped to force results;
  Panel A/B disclosure + converged-only numbers; abstract clause; new Pictures/fig3_v2.png
  (open circles = band not converged, from pipeline/plot_paper_figs_v2.py; fig3.png kept);
  caption clause. "32 times" at 03:705 and 04:157 now followed by "14 times where all searches
  converged" (approved).
  Same open-marker convention in new Pictures/Fig1_v2.png (Fig1.png kept), caption clause added.
  fig2_force_mae.png unchanged (force error is measured at fixed geometries).
  UPLOAD fig3_v2.png AND Fig1_v2.png TO OVERLEAF.
  rxn8837, rxn0894, rxn8885: all 9 bands missed the criterion (f 0.12-0.32); spreads 4434/662/337 meV.
  Converged-only: median spread ratio x13.5 (not x32); BS above chem. acc. 2 of 14 (not 5 of 18);
  MAE ratios ~1.2/2.9/2.5 (not 9/53/23). 03:178 "does not change the result" only shown for residual force.
  DECISION NEEDED: report converged-only alongside or instead.
- [ ] **3. "Other reactive data" paragraph** [V]. 03:730-744 and 04:193-198. Grambow: unrestricted
  ansatz AND an OMol25 OOD test set -> remove. RGD1: B3LYP-D3/TZVP, closed-shell species, spin
  formalism not stated; OMol25 geodesic interpolation, 19 snapshots, UKS. ANI-1xBB: constrained
  bond-stretch scans + MD, labelled with OMol25 open-shell-singlet procedure, no TS search ->
  add a sentence. MechDB: UFF/GFN-FF + AFIR on MACE-MP-0, not "semi-empirical". Transition1x bands
  are relaxed, not interpolated. Replacement paragraph drafted in chat. Needs bib entries RGD1, ANI-1xBB.
- [ ] **4. Causal claim stronger than evidence.** tab:ladder: closed-shell training points are also
  non-stationary (0.61 vs 1.64 eV/A) -> difference of degree, not kind. Align abstract 53-57,
  Intro 297-300, subsection title 03:616, 03:695 "The training data explain it", control 03:640-645,
  04:142-143, 158-160 with what 03:634-638 and 726-728 already say ("consistent with", factor ~1.6).
  Add alternatives: kink at BS onset; no input distinguishing CS/OS singlets; 38 of 45 reactions in
  two isomer families; triplets of same reaction treated as independent; models move ~90 % of the
  way (0.13 vs 1.64). State that the 45 test-split reactions are OMol25 TRAINING data (44 of 45;
  rxn5691 not in release).
- [ ] **5. "That is how the field raises fidelity" (Intro 160-165).** ramakrishnan2015 [V]: target
  labels on re-optimised target-level geometries, Delta "accounts for ... changes in geometry".
  shiota2024: aligns energies toward PBE, supplies no higher-level labels. messerly2025, ren2026 fit.
  Better: smith2019ani1ccx, allen2026. Conclusion 72-75, 135-136: Allen goes RKS DFT -> UCCSD(T),
  contradicts "as long as the spin formalism is not changed"; 3119 configs is not "at scale".
- [ ] **6. Ch. 2 claims contradicted by own tables** [V]. 02:594-597 "same errors, no new one": biases
  shift mid +151->+45 / -134->-204, high +21->-3 / -185->-235; mid MAE gain 254->147 becomes 234->212;
  all-30 NEB barrier MAE 164.7->166.7 never mentioned. 02:564 "All 30 runs converge" vs commented
  987-996 (24 of 30). 0.007 A is a median (max 0.086 A; NEB started from T1x band) at Intro 185,
  03:25, 04:88, 02:574. RQ1 verdict differs: abstract / 02:689-694 / 04:72-73.

## Tier 2: factual errors

- [ ] 7. Transition1x [V]: bands subsampled (cumulative Fmax > 0.1 eV/A), not "every ... kept"
  (Intro 1261-1263); level chosen "For compatibility with ANI1x", cost not in paper (101-103, 601-603,
  1257-1259); NEB ran in ASE, ORCA = DFT engine (1256-1257); "restricted ... never tested the solution"
  not in paper, support is scripts/neb.py in the T1x GitLab (274-276, 1266); "5 % each" vs 287/225
  of 10 073 (1270-1271); "reproduces DFT" too strong (98-100); "neutral, closed-shell" (1251-1253).
- [ ] 8. UMA-S-1.2: ~290M total / ~6M active / 64 experts, conservative end-to-end, more training
  data; Intro 1390-1391, 1396-1400, 1377-1379. "differ in capacity, not in training data" contradicts
  next sentence. "task head" only for 1.2 (03:156-157, Intro 1498-1500).
- [ ] 9. bensberg2023 / seal2025wasp [V] automate consistent active spaces along paths; Intro 885-887
  cites them for "is not". Check the CASSCF-dropped argument in 04.
- [ ] 10. wB97M-V not in goerigk2017; cite Najibi & Goerigk, JCTC 2018, 14, 5725 (Intro 104-107, 606-609).
  "with a triple-zeta basis" is the thesis's addition (Mardirossian result is at def2-QZVPPD).
- [ ] 11. crawford2001 does not support "restricted saddle is not a stationary point" (Intro 247-249);
  guner2003: CASSCF structures + CASPT2 single points (04:57-60); "largest reactive dataset" not in
  allen2026 (02:51-52); bursch2022 weak for Intro 779-780; grafenstein2002 needs a small hedge.
- [ ] 12. Hait/Marks (PARTLY DONE 2026-09-20 in intro 220-235: "defective" removed, 121 set attributed
  to Asgeirsson, "true TS lies elsewhere" replaced; STILL OPEN: 91.8 % qualifier, tab:surfaces row,
  "a fifth" at Intro 243, Zenodo sentence and FSM failures in the appendix): "excluded as defective" not in source (removed 32 -> 88; "often" changed
  geometry) Intro 223-225; 121 set is Asgeirsson's (Intro 222, 243 vs 04:235); "true TS lies
  elsewhere" contradicted by Hait for bicyclobutane (Intro 231-235), no unrestricted gradient in
  appendix; 91.8 % = FSM + DFT refinement, 29 reactions, CI-NEB 61-71 % (Intro 211-214);
  tab:surfaces row does not fit Hait; Marks Zenodo 10.5281/zenodo.19379882 exists (Appendix 19);
  Appendix 19-21: paper text says 29 unique / 4 shared, tables show 5 shared -> footnote;
  Appendix 34-36: FSM native fails for UMA-S and MACE-OMol25 on reaction 6.
- [ ] 13. Theory: "total spin stays zero" -> M_S (Intro 773-775); "functional is the only
  approximation in a KS calculation" (549-551); stability analysis = local minimum, not global
  (table row 25, 798-800, 820-830; 03:236-238); N_FOD 5000 K only for a_x = 0, abbreviation row 68;
  basis table 15 Cartesian vs 31/37 spherical (672-677); two "configurations" picture 399-402 vs
  862-864; tab:functionals six rungs vs Perdew's five, caption; dispersion "missing from all rungs",
  D3 "empirical pairwise" (591-595); "two errors are independent" (685-687); CCSD(T) < 1 kcal/mol
  needs "near basis-set limit" (852-853); T1 "as expensive as the method it is meant to replace"
  (897-899); "characterise a structure, not a calculation" (917-918); "beta HOMO and LUMO" (792-794);
  RIJCOSX "density fitting" (1469-1470); "no spin" vs 1385 (999-1001); cutoff vs receptive field
  (991-993, 1091); "16 384 per layer" only layer 1 (1054-1057); equivariant data-efficiency needs
  NequIP cite (1019-1020); "No error estimate" overgeneralised (1226-1230).
  Abstract 35-36 "100 million structures labelled unrestricted" vs Intro 1285-1289; now 140M.
- [ ] 14. Ch. 2 counts: 4 997 x 20 = 99 940 vs 80 592 (02:203-227); validation 47/reaction, two
  groups, not "in the same way" (229-231); ~3 eV / 0.09 eV shown nowhere (02:42-46, Intro 140-142,
  02:379 "constant"); caption "1 to 5 %" -> 0-14 % (542-543); mid -91 meV dominated by rxn0896,
  tiers confounded with barrier height, head "errors" are signed means (472-477, 495-496, 702-707);
  "force error drops to a third on every reaction" -> ratios 0.22-0.55 (Intro 148, 04:79-80);
  "underestimation of similar size" fails at high MR (463-465); "correction struggles at high MR"
  vs head error +7 (696-697, 723-724; 03:323-324 footnote); 04:119-123 "gap shrinks"; baseline
  passes through because head is trained on DFT - DFT, not frozen features; 42 meV is no bound
  (430-435; 04:102-105, 113-114); "three to four times" = 2.7 / 4.1 (415-417); tab:delta-audit shows
  only winning rows; lambda_F sweep on defective head; 17 442 vs 17 424; cost argument Intro 117-122.
- [ ] 15. Preface 48-52 contradicts Ch. 3; Preface RQ2 wording differs from Intro/Abstract.
- [ ] 16. Ch. 3 details: control "factor of 32" = median of per-reaction ratios, n = 33, rows missing
  from tab:ladder (640-645); tab:ladder grouping = stability at stored geometry, differs for
  rxn10054, rxn1147, "17" unexplained (296, 660-670); "49 of 53 ... these three" -> four (586-591);
  per-model residual ratios 1.41/1.57/0.90 (439-442); spread is a disagreement measure (593-606);
  spring constant unit eV/A^2 (164); "478 steps" vs 500 per phase (180-183); tab:protocol "two rows
  differ" -> more; rank 11 mid in Ch. 2, high in Ch. 3; 04:97-100 "seven of ten" vs 02:728 / 03:323-328;
  Intro 125-126 footnote "this work omits them" -> Chapter 2.

## Tier 3: bibliography and polish

- [ ] messerly2025 authors [V]: Mitchell Messerly, Sakib Matin, Alice E. A. Allen, Benjamin Nebgen,
  Kipton Barros, Justin S. Smith, Nicholas Lubbers, Richard Messerly; number = 3.
- [ ] marks2024fsm title: "Incorporation of internal coordinates interpolation into the freezing
  string method", JCTC 21(23), 12110-12120 (2025), 10.1021/acs.jctc.5c01492; remove note.
- [ ] guner2003 "Lee, Patrick S.", number 51. hait2025 -> JCTC 21(22), 11632-11644, 10.1021/acs.jctc.5c01221
  (changed title). esen -> ICML 2025, PMLR 267:17875-17893. uma -> NeurIPS 2025. ren2026 -> Nat.
  Commun. 17, 6253, full author names. mace_code / fairchem year 2026. seal2025wasp number 38,
  e2513693122. crawford2001 number 24. batatia2022 pages 11423-11436. orca_manual URL behind login.
- [ ] New entries: Najibi & Goerigk 2018; RGD1 (Zhao et al., Sci. Data 2023); ANI-1xBB (Zhang et al.,
  JCTC 2025, 10.1021/acs.jctc.5c00347); NequIP (Batzner et al. 2022); ANI-1x (Smith et al. 2018).
- [ ] Ten uncited entries; header comment "43 Eintraege".
- [ ] Terminology: residual force (three definitions); "closed-shell" for even-electron singlets in
  appendix; "broken-symmetry surface" undefined; tier/group/stratum; level gap never defined in Ch. 2;
  many names for the two levels; "bare MACE" never introduced.
- [ ] Style: thousands separators; optimizer/optimiser, initialization, Reevaluation; Sec./Section,
  Fig./Figure; footnote before full stop (Intro 127-129); eq. error-split notation (wB97M vs wB97M-V,
  superscripts); Huber loss argument (02:251); units in tab:mace-settings / tab:delta-training;
  garbled sentence 04:235-237; unreferenced floats fig:ladder-protocol, tab:panels,
  tab:marks-stability; missing Pictures: stratified_sampling.png, neb_results.png, rmsd_parity.png.
