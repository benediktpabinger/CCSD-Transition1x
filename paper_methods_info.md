# Methods-Infos für das Workshop-Paper

Alles aus Inputs, Skripten und Logs gezogen. Jeder Punkt nennt die Quelle.
Wo eine Angabe nicht auffindbar war, steht **NOT FOUND** — nicht geraten.

Zwei Pfadwurzeln:
`REPO` = `c:\Transition 1X\Transition 1x\Transition1x`,
`H` = `/home/energy/s242862` (Cluster `slid.fysik.dtu.dk`).

---

## 1. Reaktionsauswahl

### 1.1 Herkunft des 279er-Kandidatensatzes

Quelle: `REPO/pipeline/job_fod_screen.sh` (Zeilen 15–24, Kommentarblock mit dem
Generator-Einzeiler).

```bash
# Generate reaction list first (done once):
#   python3 -c "
#     import os; test=set(l.strip() for l in open('/home/energy/s242862/ccsd_dataset/test_reactions.txt'))
#     neb='/home/energy/s242862/orca_neb_results'
#     rxns=sorted(r for r in os.listdir(neb) if r in test
#                 and os.path.exists(f'{neb}/{r}/converged')
#                 and os.path.exists(f'{neb}/{r}/transition_state.xyz'))
#     open('/home/energy/s242862/fod_rxn_list.txt','w').write('\n'.join(rxns))
#   "
```

Ausgangsmenge ist der Transition1x-**test**-Split, Datei `H/ccsd_dataset/test_reactions.txt`
mit **287** Zeilen. Zwei Filter: der ωB97M-V-NEB muss eine `converged`-Datei und
eine `transition_state.xyz` hinterlassen haben. Ergebnis `H/fod_rxn_list.txt`,
`wc -l` = 278 bei fehlendem Zeilenumbruch am Ende (`'\n'.join`), also **279 Einträge**.

Der Split selbst wird in `REPO/pipeline/orca_neb.py` (`--split`, default `test`)
und in `REPO/pipeline/job_orca_neb.sh` (`--split test`, `--array 0-286`) gewählt.

### 1.2 Strata-Konstruktion der 45er-Auswahl

Quelle (identischer Code in beiden Dateien):
`REPO/pipeline/collect_stability45.py` Zeilen 5–10,
`REPO/pipeline/sweep_summary.py` Zeilen 164–170.

```python
res = sorted(json.load(open(f'{H}/fod_ranking.json'))['results'],
             key=lambda r: -r['nfod'])
n = len(res)
TOP = [res[i]['rxn'] for i in range(26)]
MID = [res[i - 1]['rxn'] for i in [11, 40, 68, 97, 126, 154, 183, 212, 240, 269]]
LOW = [res[i]['rxn'] for i in range(n - 10, n)]
```

Rangbasis ist `H/fod_ranking.json`, absteigend nach `nfod` sortiert. Die
MID-Indizes sind 1-basierte Ränge (`res[i - 1]`).

Warum 45 und nicht 46: Quelle `REPO/pipeline/job_stab_pipeline.sh` Zeilen 14–15.

```bash
#   45 reactions (top-26 by N_FOD + mid-10 + low-10; rxn0896 is rank 11 and in
#   both top-26 and mid, hence 45 not 46)
```

### 1.3 Strata-Grenzen (Ränge, N_FOD-Bereiche, n)

Quelle: `REPO/chapter_mr_v2.md` Zeilen 183–188. Die Tabelle ist im Repo nur
dort abgelegt; ein erzeugendes Skript für genau diese Zeilen wurde nicht
gefunden — die Rohwerte stehen in `H/fod_ranking.json`.

```
Schicht                              n   Ränge     N_FOD          MR   Kontrolle
oberste 26 nach N_FOD               26     1–26    0.684–1.146    18       8
zehn über die Rangliste verteilt     9   40–269    0.017–0.566     1       8
unterste 10                         10  270–279    0.003–0.014     0      10
------------------------------------------------------------------------------
gesamt                              45    1–279    0.003–1.146    19      26
```

Die harte Reaktionsliste inkl. Gruppenlabel steht in
`REPO/pipeline/job_stab_pipeline.sh` Zeilen 31–32 (`RXNS=(...)`, `GRPS=(...)`,
26× `high`, 9× `mid`, 10× `low`).

---

## 2. Referenzrechnungen (RKS-TS)

### 2.1 ORCA-Keyword-Zeile

**Der Referenz-NEB ist kein ORCA-NEB.** Er läuft als ASE-NEB, in dem jedes Bild
einen ORCA-`EnGrad`-Einzelpunkt rechnet.
Quelle: `REPO/pipeline/orca_neb.py` Zeilen 47–54.

```python
return ORCA(
    profile=profile,
    charge=0,
    mult=1,
    orcasimpleinput='wB97M-V def2-TZVP def2/J RIJCOSX TightSCF EnGrad',
    orcablocks='%pal nprocs 1 end\n%maxcore 4000\n%scf maxiter 200 end',
    directory=scratchdir,
)
```

Kein Grid-Keyword in der Eingabe. Die tatsächlich benutzten Gitterparameter,
ausgelesen aus einer Ausgabedatei desselben Niveaus
(`H/orca_freq/rxn0101_UMA-M/bs_sp.out`):

```
Program Version 5.0.4 -  RELEASE  -
General Integration Accuracy     IntAcc      ... 4.388
Radial Grid Type                 RadialGrid  ... OptM3 with GC (2021)
Angular Grid (max. ang.)         AngularGrid ... 4 (Lebedev-302)
```

### 2.2 NEB-Einstellungen

Quelle: `REPO/pipeline/orca_neb.py` Zeilen 68–69 (Bilder), 148–163 (NEB/CI),
`REPO/pipeline/job_orca_neb.sh` Zeilen 31–40 (verwendete Argumente).

```python
# R + last 8 interior + P
final_positions = [positions[0]] + list(positions[-8:]) + [positions[9]]
...
neb = NEB(images, climb=False, parallel=False)
relax_neb = NEBOptimizer(neb, logfile=...)
relax_neb.run(fmax=args.neb_fmax, steps=args.steps)
neb.climb = True
converged = relax_neb.run(fmax=args.cineb_fmax, steps=args.steps)
```

```bash
--neb-fmax   0.15      # eV/Å, Bandphase
--cineb-fmax 0.05      # eV/Å, ab hier climb=True
--steps      500
```

- **Bildzahl:** 10, keine Interpolation — Bild 0 = `positions[0]`, Bilder 1–8 =
  die letzten 8 Konfigurationen des wB97x-Bandes aus `Transition1x.h5`,
  Bild 9 = `positions[9]`.
- **Climbing Image ab:** nach Erreichen von fmax 0.15 eV/Å in der Bandphase.
- **Federkonstante:** in keinem Skript gesetzt. ASE-Default `k=0.1`, verifiziert
  in der installierten Quelle `H/.local/lib/python3.13/site-packages/ase/mep/neb.py`
  Zeile 802 (`def __init__(self, images, k=0.1, climb=False, parallel=False, ... method=None ...)`).
- **Tangentenmethode:** `orca_neb.py` übergibt **kein** `method`. ASE 3.28.0
  löst `method=None` auf `'improvedtangent'` auf (Quelle: dieselbe Datei,
  Zeilen 328–341, Warnung + `method = 'improvedtangent'`; die Warnung ist an
  `method is None` gebunden). **Welcher Default zur
  Laufzeit der April-Läufe galt, lässt sich aus den Logs nicht festnageln:**
  nur 71 von 522 `H/logs/orca_neb_*.log` tragen diese Warnung, und die
  `transition_state.xyz` in `H/orca_neb_results/` datieren 2026-04-03 bis 04-19.
  Die Modell-Skripte setzen `method='improvedtangent'` explizit (Abschnitt 3.2).

### 2.3 Endpunktrelaxation

Quelle: `REPO/pipeline/orca_neb.py` Zeilen 136–144.

```python
BFGS(images[0],  logfile=.../relax_r.log).run(fmax=0.05)
BFGS(images[-1], logfile=.../relax_p.log).run(fmax=0.05)
```

Gleiches Niveau wie das Band (derselbe ORCA-Calculator, Abschnitt 2.1),
ASE-BFGS, `fmax = 0.05 eV/Å`, kein Schrittlimit gesetzt.

---

## 3. Modell-NEB (UMA-S, UMA-M, eSEN)

### 3.1 Checkpoints und Versionen

Quelle: `REPO/pipeline/job_uma_neb.sh`, `job_uma_m_neb.sh`,
`job_esen_neb.sh` (jeweils die Zeile `CHECKPOINT=...`); Dateilisting `H/checkpoints/`.

```
UMA-S   H/checkpoints/uma-s-1p2.pt              2 333 393 167 B   2026-06-02
UMA-M   H/checkpoints/uma-m-1p1.pt             11 174 706 771 B   2026-06-14
eSEN    H/checkpoints/esen_sm_conserving_all.pt    50 958 067 B   2026-06-02
```

Herkunft, soweit im Repo dokumentiert:

```python
# REPO/pipeline/_deploy_uma.py, Z. 36–41
hf_hub_download(repo_id='facebook/UMA',
                filename='checkpoints/uma-s-1p2.pt', ...)

# REPO/pipeline/_download_esen_ckpt.py, Z. 22–28
hf_hub_download(repo_id='facebook/OMol25',
                filename='checkpoints/esen_sm_conserving_all.pt', ...)
```

- **Release-Tag für `uma-m-1p1.pt`: NOT FOUND** — kein Download-Skript im Repo
  oder auf dem Cluster nennt eine `repo_id` für UMA-M; nur der Dateiname belegt
  die Version.
- **fairchem-Version:** `fairchem_core-2.20.0`
  (`H/.local/lib/python3.13/site-packages/fairchem_core-2.20.0.dist-info`).

Task-Head — **die drei Skripte sind hier nicht gleich**:

```python
# uma_neb.py Z. 106, uma_m_neb.py Z. 77
atoms.calc = FAIRChemCalculator(predict_unit, task_name='omol')

# esen_neb.py Z. 104   -- kein task_name
atoms.calc = FAIRChemCalculator(predict_unit)
```

Ladung und Spin werden nirgends in `atoms.info` gesetzt; fairchem meldet dazu
pro Bild (Quelle `H/logs/uma_neb_10438881_0.log`):

```
WARNING:root:task_name='omol' detected, but charge is not set in atoms.info. Defaulting to charge=0.
WARNING:root:task_name='omol' detected, but spin multiplicity is not set in atoms.info. Defaulting to spin=1.
```

### 3.2 ASE-Setup

Quelle: `REPO/pipeline/uma_neb.py` Z. 60–61 und 144–160,
`uma_m_neb.py` Z. 112, `esen_neb.py` Z. 142,
Argumente aus den drei `job_*_neb.sh`.

```python
# Bilder: identisch zu orca_neb.py, keine Interpolation
final_positions = [positions[0]] + list(positions[-8:]) + [positions[9]]

neb = NEB(images, climb=False, parallel=False, method='improvedtangent')
relax_neb = NEBOptimizer(neb, logfile=.../neb.log)
relax_neb.run(fmax=args.neb_fmax, steps=args.steps)
neb.climb = True
converged = relax_neb.run(fmax=args.cineb_fmax, steps=args.steps)
```

```bash
--neb-fmax 0.15   --cineb-fmax 0.05   --steps 500     # alle drei Modelle
```

- **Images:** 10. **Interpolation:** keine, dieselben H5-Startbilder wie die Referenz.
- **Optimizer:** `ase.mep.neb.NEBOptimizer`, Default-Methode `'ODE'`
  (Quelle: installierte `ase/mep/neb.py` Z. 896–901).
- **Endpunkte:** BFGS, `fmax=0.05`, mit dem jeweiligen Modell.
- **Federkonstante:** nicht gesetzt → ASE-Default `k=0.1` (Quelle wie 2.2).
- **climb ab:** fmax 0.15 eV/Å erreicht.

### 3.3 Exakte Definition von `F_model` in der CSV

Quelle: `REPO/pipeline/paper_rows.py` Z. 66–79 (`model_maxforce`).

```python
def model_maxforce(p):
    L = open(p, errors='replace').read().split('\n')
    n = int(L[0].split()[0])
    if 'forces' not in L[1]:
        return None
    F = []
    for line in L[2:2 + n]:
        f = line.split()
        if len(f) < 7:
            return None
        F.append([float(x) for x in f[4:7]])
    return float(np.abs(np.array(F)).max()) if len(F) == n else None
```

- **Größe:** `max |Komponente|` über alle Atome und alle drei kartesischen
  Richtungen. **Keine Norm.**
- **Gelesen aus:** `H/<modeldir>/<rxn>/transition_state.xyz`, Spalten 5–7 des
  extxyz-Körpers. Header-Zeile z. B.
  `Properties=species:S:1:pos:R:3:forces:R:3 energy=-8755.525390625 pbc="F F F"`.
- **Welche Struktur:** das energiehöchste Bild nach der CI-Phase, geschrieben in
  `uma_neb.py` Z. 169 — `ts_out = max(images, key=lambda x: x.get_potential_energy())`.
  Die Auswahl erfolgt über die Energie, nicht über den CI-Index; nach `climb=True`
  ist das per Konstruktion das Climbing Image.
- **Welche Kraft:** die vom Calculator gemeldete Kraft (`atoms.calc.results`),
  die ASE beim Schreiben mit ausgibt — **nicht** die projizierte NEB-Kraft.

---

## 4. DFT-Kräfte an Modellgeometrien (`F_dft`)

### 4.1 Die Zwei-Lauf-Sequenz

Zwei Skripte erzeugen dieselben Verzeichnisse `H/orca_freq/<rxn>_<Modell>/`.
Ursprungslauf: `REPO/pipeline/job_orca_freq_sweep.sh` Z. 64–101
(92 Aufgaben aus `H/freq_tasks.txt`).

```
! UKS wB97M-V def2-TZVP def2/J RIJCOSX TightSCF
%pal nprocs 12 end
%maxcore 3000
%scf
  STABPerform true
  STABRestartUHFifUnstable true
  MaxIter 300
end
* xyzfile 0 1 start.xyz
```

```
! UKS wB97M-V def2-TZVP def2/J RIJCOSX TightSCF EnGrad MORead
%moinp "bs_start.gbw"
%pal nprocs 12 end
%maxcore 3000
%scf
  MaxIter 300
end
* xyzfile 0 1 start.xyz
```

Nachzug der 52 fehlenden Paare: `H/job_orca_grad_gap.sh` (52 Aufgaben aus
`H/grad_gap_tasks.txt`, Format `rxn0101:UMA-S`). Identische Keyword-Zeilen,
abweichend nur `%pal nprocs 8` und `%maxcore 4500`, plus harte Abbrüche:

```bash
grep -q "ORCA TERMINATED NORMALLY" bs_sp.out || { echo "ABBRUCH: bs_sp nicht normal beendet"; tail -4 bs_sp.out; exit 7; }
[ -f bs_sp.gbw ] || { echo "ABBRUCH: keine Orbitale aus bs_sp"; exit 8; }
grep -q "CARTESIAN GRADIENT" engrad.out || { echo "ABBRUCH: kein Gradient in der Ausgabe"; exit 10; }
```

Geometrie unverändert: `cp "$SRC" start.xyz` mit
`SRC=$H/$DIR/$RXN/transition_state.xyz`.

### 4.2 Auf welcher Lösung der Gradient läuft

Entschieden wird das von ORCA im ersten Lauf, nicht von uns:
`STABPerform true` + `STABRestartUHFifUnstable true`. Ist die restringierte
Lösung stabil, bleibt der Einzelpunkt darauf; ist sie instabil, startet ORCA als
UHF neu und landet auf der gebrochenen Lösung. Der zweite Lauf liest mit
`MORead` genau diese Orbitale, rechnet also auf derselben Fläche.

Begründung wörtlich aus `H/job_orca_grad_gap.sh` Z. 13–17:

```
# The recipe is copied verbatim from the pairs that already exist
# (orca_freq/rxn0101_UMA-M): a single point with stability analysis to obtain
# the ground-state orbitals, then EnGrad reading exactly those orbitals, so
# the gradient sits on the same surface the model is being judged against.
```

Warum zwei Läufe: `REPO/pipeline/job_orca_freq_sweep.sh` Z. 26–27 —
`STABPerform` verträgt nur `RunTyp SinglePoint`.

Auslesen in die CSV: `REPO/pipeline/paper_rows.py` Z. 82–103 — `max|dE/dx|` über
alle Komponenten des Blocks `CARTESIAN GRADIENT` aus `engrad.out`, mal
`51.42208` Eh/Bohr → eV/Å.

---

## 5. Stabilitätsanalyse (Prädiktor)

### 5.1 Programm und Keywords

**ORCA-Keywords für diesen Schritt: NOT FOUND — die Prädiktor-Stabilitätsanalyse
ist PySCF, nicht ORCA.** `STABPerform` taucht im Projekt nur in den
ORCA-Ketten der Abschnitte 4 und 8 auf, nicht hier.

Quelle: `REPO/pipeline/stability_pipeline.py` Z. 52–57 und 82–88.

```python
BASIS = 'def2-tzvp'
XC = 'wb97m_v'
HA_BOHR_TO_EV_ANG = 51.42207
HA_TO_MEV = 27211.386
S2_MIN = 0.05          # below this the BS solution counts as collapsed
LAMBDA_MAX = 1.0       # |lambda| above this = Davidson breakdown

def make_rks(mol, mem, level_shift=0.0):
    mf = dft.RKS(mol)
    mf.xc = XC; mf.grids.level = 3
    mf.max_cycle = 300; mf.conv_tol = 1e-10; mf.max_memory = mem
```

Aufruf der Analyse, Z. 203–204:

```python
_, mo_ext, int_st, ext_st = mf.stability(internal=True, external=True,
                                         return_status=True)
```

Niveau also ωB97M-V/def2-TZVP in PySCF, `grids.level 3`, `conv_tol 1e-10`,
`max_cycle 300`. Job: `REPO/pipeline/job_stab_pipeline.sh` (45 Reaktionen ×
4 Geometriequellen).

### 5.2 Was `lambda_min` in der CSV ist

Quelle: `REPO/pipeline/stability_pipeline.py` Z. 126–135 und 205–210,
Übernahme in die CSV in `REPO/pipeline/paper_rows.py` Z. 139 (`g['lmin_ext']`).

```python
def lam(eigs, key):
    """lowest eigenvalue, with a plausibility guard against Davidson breakdown"""
    v = eigs.get(key)
    if not v:
        return None, None
    m = min(v)
    if abs(m) > LAMBDA_MAX:
        return None, f'davidson_breakdown ({m:.3g})'
    return round(m, 8), None
```

- **Welcher Eigenwert:** der kleinste Eigenwert der **externen** (RKS→UKS)
  Orbital-Hesse der restringierten Lösung, Feld `lmin_ext`.
- **An welcher Geometrie:** am RKS-TS, Eintrag `source == 'RKS-ref'`, also
  `H/orca_neb_results/<rxn>/transition_state.xyz` (Z. 59–64, `SOURCES`).
- **Einheit:** Hartree. Die Werte werden aus dem PySCF-Log per Regex
  (`EIG_RE`, Z. 66) gelesen und unverändert übernommen — keine Umrechnung.
- **Vorzeichen:** negativ = instabil. Der Prädiktor-Score ist `-lambda_min`.
- Beträge über 1 Ha werden als Davidson-Zusammenbruch verworfen (`None`), nicht
  als Zahl geführt.

### 5.3 `breaking_depth`

Quelle: `REPO/pipeline/stability_pipeline.py` Z. 226 und 243–244,
Vorzeichenwechsel in `REPO/pipeline/paper_rows.py` Z. 121–128.

```python
# stability_pipeline.py
de = (e_bs - float(mf.e_tot)) * HA_TO_MEV
bs = {'route': route, 'converged': bool(n1.converged),
      'e_uks': round(e_bs, 10), 'de_meV': round(de, 3), 's2': round(s2, 6)}
```

```python
# paper_rows.py
bs = g.get('bs')
if g['ext_stable']:
    depth = 0.0
elif bs and bs.get('de_meV') is not None:
    depth = -float(bs['de_meV'])
```

- **Definition:** ja, `E(RKS) − E(BS)` am RKS-TS. `de_meV` ist `E(BS) − E(RKS)`
  und damit negativ; die CSV dreht das Vorzeichen, sodass `breaking_depth`
  positiv ist.
- **Niveau:** ωB97M-V/def2-TZVP, PySCF, dieselbe Rechnung wie 5.1. `E(BS)` kommt
  aus einer Second-Order-Newton-UKS-Rechnung, geseedet aus dem externen
  Instabilitätsvektor (Route 1) oder aus einem Triplett-β-HOMO-Flip (Route 2,
  Fallback, Z. 151–170).
- **Einheit im File:** meV, drei Nachkommastellen.
- **Stabile Reaktionen:** exakt `0.000` (es existiert keine BS-Lösung; `bs` ist
  `None`).
- Gemessene Spanne über die 44 instabilen Zeilen: 1.3 bis 648.5 meV,
  Median 44.5 (Ausgabe von `REPO/pipeline/paper_rows.py`).

---

## 6. N_FOD

Quelle: `REPO/pipeline/screen_fod.py` Z. 24–25, 35–51 und 65,
Job `REPO/pipeline/job_fod_screen.sh`.

```python
K_TO_HA = 3.16681e-6   # Boltzmann constant in Ha/K
T_EL    = 5000.0        # K  (Grimme standard)

def compute_fod(xyz_path, basis='def2-SVP', xc='PBE'):
    ...
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.max_cycle = 300
    sigma = T_EL * K_TO_HA
    mf = smearing_(mf, sigma=sigma, method='fermi')
    ...
    nfod = float(np.sum(np.abs(mo_occ - n0)))
```

- **Programm:** PySCF (`pyscf.dft.RKS` + `pyscf.scf.addons.smearing_`), kein
  ORCA-Keyword beteiligt.
- **Funktional / Basis:** PBE / def2-SVP.
- **T_el:** 5000 K, umgesetzt als `sigma = T_el * k_B` in Hartree,
  `method='fermi'`.
- **Definition:** `N_FOD = Σ_i |n_i − n0_i|` mit `n0 = 2` für die untersten
  `n_elec/2` MOs, sonst 0; gegengeprüft gegen `2 · Σ_virt n_i`
  (Feld `nfod_check`).
- **Geometrie:** die RKS-TS-Struktur, `<neb-dir>/<rxn>/transition_state.xyz`.
- **Ladung/Spin:** `charge=0, spin=0`.
- Ergebnisdatei: `H/fod_ranking.json`, im Repo gespiegelt als
  `REPO/fod_ranking.json`.

---

## 7. Verifikation (drei Stufen)

Referenzimplementierung: `REPO/pipeline/sweep_summary.py`.
Schwellen dort Z. 30–31; die Nachnutzer `verdict_final.py`, `imag_mode.py`,
`saddle_matrix.py`, `stage3_new.py` tragen dieselben Konstanten.

```python
GRAD_OK = 0.15          # above this the point is not stationary
FRAC_MIN, RATE_MIN = 0.10, 0.05
```

### 7.1 Stufe 1 — Stationarität

Quelle: `REPO/pipeline/sweep_summary.py` Z. 104–133 (`orca`) und 135–152
(`stab_grad`).

`max |Komponente|` des Gradienten, Schwelle 0.15 eV/Å. **Fläche: der
Grundzustand.** Zwei Bezugsquellen, in dieser Reihenfolge:

1. der ORCA-`engrad.out` aus der Kette in Abschnitt 4 — dort hat
   `STABRestartUHFifUnstable` die Fläche bereits gewählt;
2. Fallback aus `H/stab_pipeline/<rxn>/result.json`, explizit nach Stabilität
   verzweigt:

```python
if g['ext_stable']:
    return (g.get('rks_grad') or {}).get('max_evang')
return ((g.get('bs') or {}).get('bs_grad') or {}).get('max_evang')
```

Grund für den Fallback, wörtlich Z. 136–141: die fünfzehn Modellstrukturen mit
PySCF-Hesse haben keinen ORCA-`engrad`, und ein fehlender Gradient als
gescheiterte Stufe 1 zu werten hätte dreizehn Kandidaten still verworfen.

### 7.2 Stufe 2 — imaginäre Moden

Quelle: `REPO/pipeline/sweep_summary.py` Z. 66–78 (`trans_rot`) und 81–102
(`analyse`).

```python
m = np.array([atomic_masses[atomic_numbers[s]] for s in sym])
msqrt = np.sqrt(m)
w = np.repeat(1.0 / msqrt, 3)
Hm = hess * w[:, None] * w[None, :]
P = trans_rot(msqrt, xyz / BOHR)
Q = np.eye(len(Hm)) - P @ P.T
ev, vec = np.linalg.eigh(Q @ Hm @ Q)
fr = np.sign(ev) * np.sqrt(np.abs(ev)) * CM
k = int(np.argmin(ev))
q = vec[:, k].reshape(-1, 3)
q = q / np.linalg.norm(q)
out = {'n_imag': int((fr < -20).sum()), 'imag': float(fr[k])}
```

- **Zählschwelle:** `fr < -20` cm⁻¹. (Anmerkung: das Auswerteskript des
  §8-Jobs `H/job_orca_nebci_split.sh` benutzt für seine Konsolenausgabe
  `v < -1.0` cm⁻¹ — das ist eine Anzeige, keine Stufe-2-Entscheidung.)
- **Projektion:** Translation und Rotation werden massengewichtet aufgestellt
  (3 Translations- + 3 Rotationsvektoren), per SVD orthonormiert
  (`s > 1e-8`) und mit `Q = I − P Pᵀ` beidseitig herausprojiziert.
- **Hesse-Typ:** `numfreq.hess` aus ORCA, `NumFreq` mit
  `%freq CentralDiff true / Increment 0.005` — Quelle
  `REPO/pipeline/job_orca_freq_sweep.sh` Z. 103–116.
  Numerisch ist Pflicht: ωB97M-V trägt VV10, und ORCA 5.0.4 hat dafür keine
  analytischen zweiten Ableitungen (`Calc_Hess true` bricht mit
  `ORCA_CPSCF: The CPSCF equations can not yet handle non-local correlation` ab).
- **Level:** ωB97M-V/def2-TZVP def2/J RIJCOSX TightSCF, `MORead` von
  `bs_start.gbw` — dieselben Orbitale wie Gradient und Einzelpunkt.

### 7.3 Stufe 3 — Modenzuordnung

Quelle: `REPO/pipeline/sweep_summary.py` Z. 94–101; Schwellen Z. 31,
geprüft in Z. 291.

```python
if pairs:
    idx = sorted({i for a, b, _ in pairs for i in (a, b)})
    bonds = [(nm, abs(float(np.dot(q[a] - q[b],
              (xyz[a] - xyz[b]) / np.linalg.norm(xyz[a] - xyz[b])))),
              float(np.linalg.norm(xyz[a] - xyz[b]))) for a, b, nm in pairs]
    out.update({'frac': float((q[idx] ** 2).sum()), 'bonds': bonds,
                'maxrate': max(b[1] for b in bonds)})
...
elif a['frac'] >= FRAC_MIN and a['maxrate'] >= RATE_MIN:
```

- **`reactive pair`:** die zwei Atompaare mit dem größten `|d_P − d_R|`.
  Definition in `REPO/pipeline/bs_tsopt_v2.py` Z. 246–260:

```python
def reactive_bonds(rxn, syms):
    """Top-2 pairs by |d_P - d_R| -- same rule as ~/_rxn_coord_full.py."""
    r = ase_read(f'{HOME}/orca_neb_results/{rxn}/reactant.xyz')
    p = ase_read(f'{HOME}/orca_neb_results/{rxn}/product.xyz')
    br, bp = get_all_bonds(r), get_all_bonds(p)
```

  `get_all_bonds` (Z. 235–243) zählt ein Paar als gebunden, wenn
  `d < 1.3 · (r_cov,i + r_cov,j)` (Z. 241, `scale=1.3` aus Z. 235). Gelesen werden die Paare zur Laufzeit aus
  `H/{bs_tsopt_fromneb|bs_tsopt_v2|bs_tsopt_batch}/<rxn>/result.json`,
  Feld `reactive_bonds`, erste zwei Einträge (`sweep_summary.py` Z. 154–160).

- **`mode fraction` (`frac`):** Summe der quadrierten Komponenten des
  normierten Modenvektors über alle Atome, die in den beiden reaktiven Paaren
  vorkommen. **Massengewichtet** — `q` ist der Eigenvektor der
  massengewichteten, projizierten Hesse, nicht kartesisch (siehe
  `REPO/chapter_mr_v2.md` Anhang A.3, wo die kartesische Vorfassung
  zurückgenommen wird).
- **`bond rate` (`maxrate`):** je Paar `|(q_a − q_b) · ê_ab|` mit
  `ê_ab` = Einheitsvektor entlang der aktuellen Bindung; genommen wird das
  Maximum über die zwei Paare.
- **Schwellen:** `frac >= 0.10` **und** `maxrate >= 0.05`.
- Ohne hinterlegte `reactive_bonds` lautet das Urteil
  `'no reactive bonds recorded'`, nicht „bestanden".

### 7.4 Pfad zum Filter-Code

```
REPO/pipeline/sweep_summary.py      Referenz: alle drei Stufen, Definition der Schwellen
REPO/pipeline/verdict_final.py      dieselbe Regel, symmetrisch auf beide Gruppen
REPO/pipeline/imag_mode.py          Modenanalyse einzeln
REPO/pipeline/stage3_new.py         Stufe 3 über mehrere Ergebnisverzeichnisse
REPO/pipeline/saddle_matrix.py      eine Zeile je Reaktion, GRAD_OK = 0.15
```

---

## 8. Refinement (Sektion Consequences)

### 8.1 Schritt-2/3-Keywords

**Achtung Provenienz:** die Repo-Kopie
`REPO/pipeline/job_orca_nebci_split.sh` ist die Pilotfassung mit drei fest
verdrahteten Reaktionen und nur dem billigen Niveau. Die Fassung, die die
Produktionsläufe erzeugt hat, liegt auf dem Cluster unter
`H/job_orca_nebci_split.sh` (6070 B, mit `LEVEL`, `RXN_LIST`, `OUT_ROOT`).
Unten die Cluster-Fassung.

Niveauwahl, `H/job_orca_nebci_split.sh`, Block `case "${LEVEL:-cheap}"`:

```bash
case "${LEVEL:-cheap}" in
  prod)  METHOD="wB97M-V def2-TZVP def2/J RIJCOSX"
         PROCS=${PROCS:-12} ; MEMC=${MEMC:-3000} ;;
  *)     METHOD="wB97X 6-31G(d)"
         PROCS=${PROCS:-8}  ; MEMC=${MEMC:-3500} ;;
esac
if [ "${LEVEL:-cheap}" = "prod" ]; then
  FREQKW=NumFreq; HESSKW="  NumHess true
"
else
  FREQKW=Freq;    HESSKW=""
fi
```

Schritt 1, Band:

```
! UKS $METHOD NEB-CI TightSCF SlowConv
%pal
  nprocs $PROCS
end
%maxcore $MEMC
%scf
  BrokenSym 1,1
  MaxIter 500
end
%neb
  Product "$W/product.xyz"
  NImages 8
  MaxIter 500
  Preopt true
  PrintLevel 3
end
* xyzfile 0 1 $W/reactant.xyz
```

Schritt 2, Einzelpunkt mit Stabilitätsanalyse am Climbing Image:

```
! UKS $METHOD SP TightSCF SlowConv
%pal
  nprocs $PROCS
end
%maxcore $MEMC
%scf
  STABPerform true
  STABRestartUHFifUnstable true
  MaxIter 500
end
* xyzfile 0 1 $W/$CI
```

Schritt 3, TS-Optimierung von genau diesen Orbitalen:

```
! UKS $METHOD OptTS $FREQKW TightSCF SlowConv MORead
%moinp "$W/bs.gbw"
%pal
  nprocs $PROCS
end
%maxcore $MEMC
%geom
  Calc_Hess true
$HESSKW  MaxIter 200
end
%scf
  MaxIter 500
end
* xyzfile 0 1 $W/$CI
```

Auf `LEVEL=prod` expandiert das zu `OptTS NumFreq` und `%geom Calc_Hess true /
NumHess true / MaxIter 200`. Das Climbing Image wird als Datei
`*NEB-CI_converged.xyz` übernommen; findet sich keine, bricht das Skript mit
`exit 3` ab.

Die beiden anderen Startpunkte:
- **B, vorhandenes Band:** `H/job_orca_sep_step23.sh` → `H/sep_step23/`.
- **C, Modellgeometrie:** `H/job_orca_umam_eval.sh`, dieselbe Dreierkette
  (`bs_sp` mit `STABPerform` → `engrad MORead` → `numfreq MORead` mit
  `CentralDiff true / Increment 0.005`), Quelle
  `SRC=$H/bs_tsopt_umam/$RXN/ts_opt.xyz`, Ziel
  `H/orca_freq/tsopt_<rxn>_UMA-M/`.

### 8.2 Wall-clock-Zahlen

Gemessen aus `TOTAL RUN TIME:` der jeweiligen ORCA-Ausgaben, Summe über die
Stufen je Reaktion. **Die Zahlen im Kapitel sind gerundet und stimmen nicht
überall mit dem Gemessenen überein** — hier das Gemessene:

```
Startpunkt A   H/bs_uks_nebci_prod/<rxn>/{neb,bs,tsopt,tsopt2}.out
   Summe je Reaktion   6.39 bis 46.88 h   (n = 11 mit vollständiger Kette)
   davon Band allein   5.48 bis 45.05 h
   Kapitel sagt        "7 bis 45 h"

Startpunkt B   H/sep_step23/<rxn>/{bs,tsopt,tsopt2}.out
   Summe je Reaktion   0.72 bis 2.44 h   (n = 9 konvergiert)
   Kapitel sagt        "1 bis 2 h"

Startpunkt C   H/orca_freq/tsopt_<rxn>_UMA-M/{bs_sp,engrad,numfreq}.out
   Summe je Reaktion   0.40 bis 0.93 h, Median 0.59 h   (n = 18)
   Kapitel sagt        "~1 h"
```

**Wichtige Einschränkung zu C:** diese 0.4–0.9 h sind nur die ORCA-Bewertungskette.
Die vorgelagerte TS-Optimierung an der Modellgeometrie läuft in PySCF und ist
nicht enthalten; aus `H/bs_tsopt_umam/<rxn>/result.json`, Feld `elapsed_s`:

```
2.73 bis 23.13 h, Median 4.19 h   (n = 9; rxn0894 ohne Wert, Status RKS_NICHT_KONVERGIERT)
```

Die acht Wandzeit-Abbrüche in A tragen kein `TOTAL RUN TIME` und sind in der
Spanne oben nicht enthalten: rxn0894, rxn1283, rxn3107, rxn4518, rxn4522,
rxn5691, rxn7060, rxn7949 (SLURM-Limit `--time=24:00:00` in der Pilotfassung;
die Produktionsläufe liefen mit 48 h laut `REPO/chapter_mr_v2.md` Anhang B — ein
Job-Skript mit `--time=48:00:00` wurde **NOT FOUND**).

---

## 9. Hardware / Software

### 9.1 ORCA

```
Program Version 5.0.4 -  RELEASE  -          (H/orca_freq/rxn0101_UMA-M/bs_sp.out)
Modul: ORCA/5.0.4-gompi-2023a, dazu gompi/2023a
Binärdatei über $(which orca); $EBROOTORCA/orca existiert NICHT (liegt in bin/)
Parallel: MPI-Ränge -> SBATCH --ntasks=N --cpus-per-task=1
```

Quellen: `REPO/pipeline/job_orca_freq_sweep.sh` Z. 5–6 und 40–41,
`H/job_orca_grad_gap.sh`, Blöcke `module load` und `ORCA=$(which orca)`.

### 9.2 Python-Umgebung

Aus `python3 -m pip list` auf dem Login-Knoten nach
`module load Python/3.13.5-GCCcore-14.3.0` und aus
`H/.local/lib/python3.13/site-packages/`:

```
Python        3.13.5      (Modul Python/3.13.5-GCCcore-14.3.0)
ase           3.28.0
fairchem-core 2.20.0      (aus fairchem_core-2.20.0.dist-info)
torch         2.8.0
numpy         2.4.3
scipy         1.17.1
pyscf         2.12.1
h5py          3.14.0
e3nn          0.6.0
```

**mace-torch-Version: NOT FOUND** — nicht in der pip-Liste der Standardumgebung.

### 9.3 Rechenknoten

```
Modell-NEBs (UMA-S, UMA-M, eSEN)   Partition sm3090el8   gpu:RTX3090, 1 GPU
                                   --cpus-per-task=4, --mem=16GB (UMA-M 32GB)
                                   --time=1:00:00
MACE+Delta (fw2)                   Partition h200        gpu:H200, 1 GPU
ORCA (alle DFT-Läufe)              Partition xeon24el8   CPU, --ntasks 8 oder 12
PySCF (Stabilität, FOD, BS-OptTS)  Partition xeon24el8   --cpus-per-task 12 bzw. 24
```

Quellen: `REPO/pipeline/job_uma_neb.sh`, `job_uma_m_neb.sh`, `job_esen_neb.sh`
(je der `#SBATCH`-Block), `H/job_mace_delta_fw2_missing.sh` (`#SBATCH`-Block),
`REPO/pipeline/job_orca_freq_sweep.sh` (`#SBATCH`-Block),
`REPO/pipeline/job_stab_pipeline.sh` (`#SBATCH`-Block),
`REPO/pipeline/job_fod_screen.sh` (`#SBATCH`-Block); GPU-Typen aus `sinfo -o "%20P %10G ..."`:

```
sm3090el8            gpu:RTX309 s[002-007]
h200                 gpu:H200:4 sd[655-656]
```

**Exaktes CPU-Modell von `xeon24el8`: NOT FOUND** — `sinfo` gibt Partition,
GRES und Speicher aus, kein Prozessormodell.

---

## Offene Punkte

```
1.3   Skript, das die Strata-Tabelle (Ränge/N_FOD-Spannen) erzeugt   NOT FOUND
2.2   ASE-Tangentenmethode der April-Referenzläufe                   nicht festnagelbar
3.1   Release-Tag / repo_id für uma-m-1p1.pt                         NOT FOUND
8.1   Repo-Kopie von job_orca_nebci_split.sh ist die Pilotfassung    Cluster-Kopie gilt
8.2   Job-Skript mit --time=48:00:00 für die Produktionsbänder       NOT FOUND
9.2   mace-torch-Version                                             NOT FOUND
9.3   CPU-Modell der Partition xeon24el8                             NOT FOUND
```
