# Stability class vs. MLIP transition-state search outcome

Reference transition states of the Marks benchmark (arXiv:2604.00405),
classified RKS-stable / RKS-unstable at that paper's own level,
wB97X-V/def2-TZVP, joined onto its per-reaction results.

## Class sizes (paper table rows: 24 Baker + 9 Sharada = 33)

| class | rows |
|---|---|
| stable | 28 |
| unstable | 2 |
| open-shell | 3 |

The Sharada set repeats five Baker reference structures, so the 33 rows
cover 28 distinct structures. Counts below are over table rows, i.e. over
the runs the paper actually reports.

## FSM

### Aggregated over all six models

| class | runs | failures | failure rate | mean gradients (successes) |
|---|---|---|---|---|
| stable | 336 | 33 | 9.8% | 13.36 |
| unstable | 24 | 6 | 25.0% | 22.44 |
| open-shell | 36 | 4 | 11.1% | 8.09 |

### Per model

| model | class | runs | failures | failure rate | mean gradients |
|---|---|---|---|---|---|
| GFN2-xTB | stable | 56 | 8 | 14.3% | 14.98 |
| GFN2-xTB | unstable | 4 | 2 | 50.0% | 14.00 |
| GFN2-xTB | open-shell | 6 | 0 | 0.0% | 8.33 |
| AIMNet2 | stable | 56 | 6 | 10.7% | 16.54 |
| AIMNet2 | unstable | 4 | 0 | 0.0% | 35.50 |
| AIMNet2 | open-shell | 6 | 3 | 50.0% | 18.33 |
| eSEN-S | stable | 56 | 4 | 7.1% | 12.90 |
| eSEN-S | unstable | 4 | 0 | 0.0% | 31.50 |
| eSEN-S | open-shell | 6 | 0 | 0.0% | 7.17 |
| UMA-S | stable | 56 | 7 | 12.5% | 10.63 |
| UMA-S | unstable | 4 | 2 | 50.0% | 4.00 |
| UMA-S | open-shell | 6 | 0 | 0.0% | 7.33 |
| UMA-M | stable | 56 | 6 | 10.7% | 13.52 |
| UMA-M | unstable | 4 | 0 | 0.0% | 23.00 |
| UMA-M | open-shell | 6 | 0 | 0.0% | 6.00 |
| MACE-OMol25 | stable | 56 | 2 | 3.6% | 11.72 |
| MACE-OMol25 | unstable | 4 | 2 | 50.0% | 4.00 |
| MACE-OMol25 | open-shell | 6 | 1 | 16.7% | 6.20 |

## CI-NEB

### Aggregated over all six models

| class | runs | failures | failure rate | mean gradients (successes) |
|---|---|---|---|---|
| stable | 336 | 109 | 32.4% | 6.42 |
| unstable | 24 | 4 | 16.7% | 6.30 |
| open-shell | 36 | 12 | 33.3% | 5.00 |

### Per model

| model | class | runs | failures | failure rate | mean gradients |
|---|---|---|---|---|---|
| GFN2-xTB | stable | 56 | 21 | 37.5% | 8.49 |
| GFN2-xTB | unstable | 4 | 0 | 0.0% | 16.50 |
| GFN2-xTB | open-shell | 6 | 0 | 0.0% | 5.50 |
| AIMNet2 | stable | 56 | 18 | 32.1% | 10.18 |
| AIMNet2 | unstable | 4 | 4 | 100.0% | nan |
| AIMNet2 | open-shell | 6 | 4 | 66.7% | 14.50 |
| eSEN-S | stable | 56 | 14 | 25.0% | 3.67 |
| eSEN-S | unstable | 4 | 0 | 0.0% | 4.00 |
| eSEN-S | open-shell | 6 | 2 | 33.3% | 3.00 |
| UMA-S | stable | 56 | 20 | 35.7% | 5.86 |
| UMA-S | unstable | 4 | 0 | 0.0% | 4.00 |
| UMA-S | open-shell | 6 | 2 | 33.3% | 3.25 |
| UMA-M | stable | 56 | 17 | 30.4% | 6.92 |
| UMA-M | unstable | 4 | 0 | 0.0% | 3.50 |
| UMA-M | open-shell | 6 | 2 | 33.3% | 3.00 |
| MACE-OMol25 | stable | 56 | 19 | 33.9% | 3.73 |
| MACE-OMol25 | unstable | 4 | 0 | 0.0% | 3.50 |
| MACE-OMol25 | open-shell | 6 | 2 | 33.3% | 5.25 |

## Raw list

Failure modes: (a) alternate first-order saddle, (b) local minimum,
(c) spurious imaginary frequency [counted a success by the paper],
(d) no P-RFO convergence in 250 cycles, (e) SCF error,
(f) multiple strong imaginary frequencies, (g) P-RFO step failure,
(h) internal-coordinate back-transformation failure.

| set | # | reaction | class | algorithm | failures / 12 runs | modes |
|---|---|---|---|---|---|---|
| baker | 1 | HCN -> HNC | stable | FSM | 0 | - |
| baker | 1 | HCN -> HNC | stable | CI-NEB | 0 | - |
| baker | 2 | HCCH -> CCH2 | stable | FSM | 0 | - |
| baker | 2 | HCCH -> CCH2 | stable | CI-NEB | 0 | - |
| baker | 3 | H2CO -> H2 + CO | stable | FSM | 1 | b |
| baker | 3 | H2CO -> H2 + CO | stable | CI-NEB | 4 | a,b |
| baker | 4 | CH3O -> CH2OH | open-shell | FSM | 0 | - |
| baker | 4 | CH3O -> CH2OH | open-shell | CI-NEB | 0 | - |
| baker | 5 | cyclopropyl ring opening | open-shell | FSM | 3 | a |
| baker | 5 | cyclopropyl ring opening | open-shell | CI-NEB | 2 | a |
| baker | 6 | bicyclo[1.1.0]butane -> trans-butadiene | unstable | FSM | 3 | a |
| baker | 6 | bicyclo[1.1.0]butane -> trans-butadiene | unstable | CI-NEB | 2 | a,b |
| baker | 7 | formyloxyethyl 1,2-migration | open-shell | FSM | 1 | b |
| baker | 7 | formyloxyethyl 1,2-migration | open-shell | CI-NEB | 10 | b,d,e |
| baker | 8 | parent Diels-Alder cycloaddition | stable | FSM | 0 | - |
| baker | 8 | parent Diels-Alder cycloaddition | stable | CI-NEB | 12 | b,d |
| baker | 9 | s-tetrazine -> 2HCN + N2 | stable | FSM | 1 | b |
| baker | 9 | s-tetrazine -> 2HCN + N2 | stable | CI-NEB | 7 | a,f |
| baker | 10 | trans-butadiene -> cis-butadiene | stable | FSM | 0 | - |
| baker | 10 | trans-butadiene -> cis-butadiene | stable | CI-NEB | 0 | - |
| baker | 11 | CH3CH3 -> CH2CH2 + H2 | stable | FSM | 3 | a,d |
| baker | 11 | CH3CH3 -> CH2CH2 + H2 | stable | CI-NEB | 7 | a,d |
| baker | 12 | CH3CH2F -> CH2CH2 + HF | stable | FSM | 0 | - |
| baker | 12 | CH3CH2F -> CH2CH2 + HF | stable | CI-NEB | 0 | - |
| baker | 13 | acetaldehyde keto-enol tautomerism | stable | FSM | 0 | - |
| baker | 13 | acetaldehyde keto-enol tautomerism | stable | CI-NEB | 0 | - |
| baker | 14 | HCOCl -> HCl + CO | stable | FSM | 0 | - |
| baker | 14 | HCOCl -> HCl + CO | stable | CI-NEB | 0 | - |
| baker | 15 | H2O + PO3- -> H2PO4- | stable | FSM | 1 | d |
| baker | 15 | H2O + PO3- -> H2PO4- | stable | CI-NEB | 12 | b,d,g |
| baker | 16 | CH2CHCH2CH2CHO Claisen rearrangement | stable | FSM | 2 | a,d |
| baker | 16 | CH2CHCH2CH2CHO Claisen rearrangement | stable | CI-NEB | 0 | - |
| baker | 17 | SiH2 + CH3CH3 -> SiH3CH2CH3 | stable | FSM | 0 | - |
| baker | 17 | SiH2 + CH3CH3 -> SiH3CH2CH3 | stable | CI-NEB | 0 | - |
| baker | 18 | HNCCS -> HNC + CS | stable | FSM | 2 | a,b |
| baker | 18 | HNCCS -> HNC + CS | stable | CI-NEB | 2 | b,g |
| baker | 19 | HCONH3+ -> NH4+ + CO | stable | FSM | 1 | d |
| baker | 19 | HCONH3+ -> NH4+ + CO | stable | CI-NEB | 3 | b,g |
| baker | 20 | acrolein rotational TS | stable | FSM | 0 | - |
| baker | 20 | acrolein rotational TS | stable | CI-NEB | 0 | - |
| baker | 21 | HCONHOH -> HCOHNHO | stable | FSM | 0 | - |
| baker | 21 | HCONHOH -> HCOHNHO | stable | CI-NEB | 0 | - |
| baker | 22 | HNC + H2 -> H2CNH | stable | FSM | 8 | a,b,d,e |
| baker | 22 | HNC + H2 -> H2CNH | stable | CI-NEB | 12 | a,b,h |
| baker | 23 | H2CNH -> HCNH2 | stable | FSM | 0 | - |
| baker | 23 | H2CNH -> HCNH2 | stable | CI-NEB | 5 | f |
| baker | 24 | HCNH2 -> HCN + H2 | stable | FSM | 6 | a,f |
| baker | 24 | HCNH2 -> HCN + H2 | stable | CI-NEB | 6 | a,d,f |
| sharada | 1 | H2CO -> H2 + CO | stable | FSM | 1 | b |
| sharada | 1 | H2CO -> H2 + CO | stable | CI-NEB | 4 | a,b |
| sharada | 2 | SiH2 + H2 -> SiH4 | stable | FSM | 0 | - |
| sharada | 2 | SiH2 + H2 -> SiH4 | stable | CI-NEB | 2 | b |
| sharada | 3 | CH2CHOH <-> CH3CHO | stable | FSM | 0 | - |
| sharada | 3 | CH2CHOH <-> CH3CHO | stable | CI-NEB | 0 | - |
| sharada | 4 | CH3CH3 -> CH2CH2 + H2 | stable | FSM | 3 | a,d |
| sharada | 4 | CH3CH3 -> CH2CH2 + H2 | stable | CI-NEB | 7 | a,b,d |
| sharada | 5 | bicyclo[1.1.0]butane -> trans-butadiene | unstable | FSM | 3 | a |
| sharada | 5 | bicyclo[1.1.0]butane -> trans-butadiene | unstable | CI-NEB | 2 | a,b |
| sharada | 6 | parent Diels-Alder cycloaddition | stable | FSM | 0 | - |
| sharada | 6 | parent Diels-Alder cycloaddition | stable | CI-NEB | 12 | b,d |
| sharada | 7 | cis,cis-2,4-hexadiene <-> 3,4-dimethylcyclobutene | stable | FSM | 0 | - |
| sharada | 7 | cis,cis-2,4-hexadiene <-> 3,4-dimethylcyclobutene | stable | CI-NEB | 4 | b,d |
| sharada | 8 | alanine dipeptide C5 <-> C7AX | stable | FSM | 0 | - |
| sharada | 8 | alanine dipeptide C5 <-> C7AX | stable | CI-NEB | 0 | - |
| sharada | 9 | silyl ketene acetal -> silyl ester Ireland-Claisen | stable | FSM | 4 | b,d,g |
| sharada | 9 | silyl ketene acetal -> silyl ester Ireland-Claisen | stable | CI-NEB | 10 | b,d |
