# omol25_model_geoms.csv

Alles, was auf **OMol25-Niveau an den Modellgeometrien** gemessen wurde, in einer
Datei. Nichts anderes steht darin. Erzeugt von
[`pipeline/omol25_model_geoms.py`](../pipeline/omol25_model_geoms.py), das auf
dem Cluster läuft und die Rohoutputs liest.

135 Zeilen, eine je (Reaktion, Modell): 45 Reaktionen × 3 MLIPs. 82 Zeilen
RKS-stabil, 53 instabil.

## Niveau — für jede Zahl in dieser Datei identisch

```
ORCA 5.0.4
! wB97M-V def2-TZVPD def2/J RIJCOSX TightSCF DEFGRID3
%scf Thresh 1e-12 / TCut 1e-13 / MaxIter 300 end
```

Die UKS-Läufe zusätzlich mit `STABPerform true` und
`STABRestartUHFifUnstable true`. Der RKS-Lauf bewusst **ohne** Stabilitätsanalyse,
weil dort die restringierte Lösung gewollt ist, auch wo sie nicht der Grundzustand
ist — ihr Abstand zur gebrochenen Lösung *ist* die Brechungstiefe.

## Geometrien

Immer die drei Strukturen, die das jeweilige MLIP selbst erzeugt hat,
`<modeldir>/<rxn>/{reactant,transition_state,product}.xyz`, **unrelaxiert**.
Keine DFT-Optimierung, keine Referenzgeometrie. Weil MLIP- und DFT-Seite jeder
Differenz am selben Punkt gelesen werden, fällt die Geometrie aus jedem Fehler
heraus.

Die OMol25-NEB unter `orca_neb_omol25/` hat mit dieser Datei **nichts** zu tun.
Das ist ein anderer Satz Geometrien für eine andere Frage (Figur 4).

## Quellen

| Datei | liefert |
|---|---|
| `orca_om25/<rxn>_<Modell>/ts_sp.out` | UKS + STABPerform am Modell-TS |
| `orca_om25/<rxn>_<Modell>/r_sp.out` | dasselbe am Modell-Edukt |
| `orca_om25/<rxn>_<Modell>/p_sp.out` | dasselbe am Modell-Produkt |
| `orca_om25/<rxn>_<Modell>/ts_engrad.out` | Gradient auf den Orbitalen von `ts_sp` (MORead), also auf der von STABPerform gewählten Fläche |
| `orca_rks_sheet/<rxn>_<Modell>/ts_rks.out` | RKS-Einzelpunkt am selben Modell-TS, Slurm-Jobs **10767516** (4 Zeilen) und **10767531** (50 Zeilen) |
| `<modeldir>/<rxn>/*.xyz` | Energie und Kräfte des MLIP aus der Kommentarzeile bzw. den letzten drei Spalten |

## Spalten

| Spalte | Einheit | Bedeutung |
|---|---|---|
| `rxn`, `model` | | Schlüssel, `uma-s` / `uma-m` / `esen` |
| `e_r_uks_ha`, `e_ts_uks_ha`, `e_p_uks_ha` | Hartree | UKS-Einzelpunkte an Edukt, TS, Produkt |
| `e_ts_rks_ha` | Hartree | RKS-Einzelpunkt am TS; leer, wo keiner gerechnet wurde |
| `s2_r`, `s2_ts`, `s2_p` | | ⟨S²⟩ desselben Einzelpunkts |
| `unstable_ts` | 0/1 | `abs(s2_ts) > 0.05` |
| `depth_ts_mev` | meV | Brechungstiefe **an der Modellgeometrie** = E_RKS(TS) − E_BS(TS) |
| `depth_src` | | `rks_sp` = gemessen, `stabperform_stable` = 0 per Konstruktion |
| `f_model_max` | eV/Å | max‑Komponente der MLIP-Kraft am TS |
| `f_dft_max` | eV/Å | max‑Komponente der DFT-Kraft am TS |
| `f_err_max` | eV/Å | maxᵢ \|F_MLIP,i − F_DFT,i\| über alle 3N Komponenten |
| `f_err_mae` | eV/Å | ⟨\|F_MLIP,i − F_DFT,i\|⟩ über alle 3N Komponenten |
| `barr_model`, `barr_dft` | eV | E(TS) − E(R), MLIP bzw. DFT-BS |
| `barr_rks` | eV | E_RKS(TS) − E(R); leer, wo kein RKS-Lauf |
| `err_barr` | eV | `barr_model − barr_dft` |
| `rxne_model`, `rxne_dft`, `err_rxne` | eV | dasselbe für E(P) − E(R) |
| `scf_cyc_ts_uks`, `scf_cyc_ts_rks` | | SCF-Zyklen, rein diagnostisch |

Der Nullpunkt aller Barrieren ist das **Modell-Edukt desselben Laufs**. Es ist in
allen 135 Zeilen geschlossenschalig (`s2_r` = 0), RKS und BS fallen dort zusammen —
der Nullpunkt bevorzugt also keine der beiden Flächen.

## Warum die stabilen Zeilen keine Rechnung brauchen

Bei `unstable_ts = 0` hat STABPerform am selben Punkt bestätigt, dass es keine von
der restringierten verschiedene Lösung gibt. Die Tiefe ist dort 0 per Konstruktion,
nicht geschätzt. Der Vermerk steht in `depth_src`.

**Nullprobe:** `rxn7060 / esen` ist stabil, hat aber aus der Sheet-Prüfung trotzdem
einen RKS-Lauf. Gemessene Tiefe **0.0008 meV**, wo 0 stehen muss — die beiden Wege,
die Tiefe zu bestimmen, stimmen damit auf ein Mikroelektronenvolt überein.

## Prüfungen, die das Skript erzwingt

Das Skript bricht ab, wenn eine davon fehlschlägt.

- 135 Zeilen, 53 davon instabil
- die Schwelle 0.05 liegt in einer leeren Zone: kleinster ⟨S²⟩ über null ist 0.057936
- Tiefe für alle 135 Zeilen belegt
- Tiefe der stabilen Zeilen unter 1 meV (größte 0.0008)
- Tiefe der instabilen Zeilen durchweg > 0
- 54 Tiefen gemessen, 81 per STABPerform auf 0 gesetzt
- die gemeinsamen Spalten stimmen mit `omol25_compare.csv` überein, größte
  Abweichung 5·10⁻⁵

## Was drinsteht, in Zahlen

| | stabil (n=82) | instabil (n=53) |
|---|---|---|
| `f_err_mae` Median | 0.0095 | 0.0250 eV/Å |
| `f_err_max` Median | 0.0353 | 0.1034 eV/Å |
| \|`err_barr`\| Median | 0.0023 | 0.0079 eV |

Brechungstiefe der 53 instabilen Zeilen: 0.6 bis 3986 meV, Median 1027, Quartile
68 und 1724. Zwölf Zeilen liegen unter 50 meV — RKS-instabil heißt also nicht
automatisch tief gebrochen. Instabile Zeilen je Modell: UMA-S 18, UMA-M 18, eSEN 17.

## Abgrenzung zu den anderen Tabellen

| Datei | Niveau | Geometrien | wofür |
|---|---|---|---|
| **`omol25_model_geoms.csv`** | ORCA, def2-TZVPD, OMol25 | Modell-TS/R/P | Figuren 9.1 bis 9.3c, Brechungstiefe |
| `omol25_compare.csv` | mischt TZVP und TZVPD | Modell-TS/R/P | der Vergleich der beiden Niveaus; nicht für neue Auswertungen |
| `paper_rows_ext.csv` | PySCF, def2-TZVP | Modell-TS und Referenz-TS | Figuren 1 bis 8, Stufe-1-Kette |
| `rks_sheet_tzvpd.json` | ORCA, def2-TZVPD | vier Modell-TS | Rohwerte der Sheet-Prüfung, geht hier vollständig auf |
| `hinge_rows.csv`, `control_rks.csv` | PySCF, def2-TZVP | Referenz-TS | Figur 4, alte Fassung |
