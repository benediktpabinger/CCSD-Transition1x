# results/ — die geprüften Tabellen

Acht Tabellen werden von einem Skript erzeugt, das bei fehlgeschlagener
Prüfung **abbricht**, statt eine falsche Datei zu schreiben. Was hier steht,
ist damit entweder gemessen oder als fehlend gekennzeichnet.

| Tabelle | Zeilen | Skript | Doku |
|---|---|---|---|
| [`omol25_model_geoms.csv`](omol25_model_geoms.csv) | 135 | [`pipeline/omol25_model_geoms.py`](../pipeline/omol25_model_geoms.py) | [`omol25_model_geoms.md`](omol25_model_geoms.md) |
| [`paper_reactions.csv`](paper_reactions.csv) | 45 | [`pipeline/paper_reactions.py`](../pipeline/paper_reactions.py) | hier |
| [`neb_runs.csv`](neb_runs.csv) | 135 | [`pipeline/neb_runs.py`](../pipeline/neb_runs.py) | hier |
| [`model_ts_rmsd.csv`](model_ts_rmsd.csv) | 45 | [`pipeline/model_ts_rmsd.py`](../pipeline/model_ts_rmsd.py) | im Docstring |
| [`hinge_t1x.csv`](hinge_t1x.csv) | 45 | [`pipeline/hinge_tables.py`](../pipeline/hinge_tables.py) | hier + [`docs/methods_for_paper.md`](../docs/methods_for_paper.md) §7 |
| [`hinge_omol25.csv`](hinge_omol25.csv) | 33 | [`pipeline/hinge_tables.py`](../pipeline/hinge_tables.py) | hier + [`docs/methods_for_paper.md`](../docs/methods_for_paper.md) §7 |
| [`rotation_check.csv`](rotation_check.csv) | 135 | [`pipeline/rot_check.py`](../pipeline/rot_check.py) | hier + [`docs/methods_for_paper.md`](../docs/methods_for_paper.md) §4 |
| [`barrier_spread.csv`](barrier_spread.csv) | 45 | [`pipeline/barrier_spread.py`](../pipeline/barrier_spread.py) | hier |

Alle übrigen Dateien in diesem Verzeichnis stammen aus früheren
Arbeitsschritten und tragen diese Disziplin nicht. Sie sind unten einzeln
benannt — **keine davon gehört in die Übergabe**.

### Die nicht übergebenen Dateien

| Datei | Zeilen | Ära | erzeugt von | wird noch gelesen von |
|---|---|---|---|---|
| `barrier_frozen.csv` | 133 | def2-TZVP (`orca_freq/`, `orca_ep/`, Job 10765094) | `pipeline/barrier_frozen.py` | `pipeline/omol25_compare.py` |
| `saddle_residuals.csv` | 40 | Schwellenkalibrierung der Kapitel-Pipeline, nicht des Papers | `pipeline/saddle_residuals.py` | `pipeline/threshold_sensitivity.py`, `pipeline/model_saddle_stats.py`, `chapter_mr_v2.md`, `paper_methods_thresholds.md` |
| `hinge_rows.csv` | 19 | def2-TZVP, PySCF-Gradienten | `pipeline/paper_rows.py` | `plot_paper_figs.py`, `plot_fig4a_en.py`, `threshold_sensitivity.py` |
| `paper_rows.csv` | 122 | def2-TZVP | `pipeline/paper_rows.py` | `pipeline/paper_figdata.py` |
| `paper_rows_ext.csv` | 135 | def2-TZVP | `pipeline/paper_figdata.py` | `plot_paper_figs.py`, `threshold_sensitivity.py` |
| `control_rks.csv` | 26 | def2-TZVP, Kontrollpanel zur alten Figur 4 | `pipeline/paper_figdata.py` | `plot_paper_figs.py` |
| `cost_hours.csv` | 92 | Wandzeiten der Kapitel-Routen A/B/C | `pipeline/cost_hours.py` | `plot_paper_figs.py` |
| `omol25_compare.csv` | 135 | **mischt Niveaus** — TZVP- neben TZVPD-Spalten | `pipeline/omol25_compare.py` | `pipeline/omol25_model_geoms.py` (Gegenprobe), `energy_outlier_check.py`, `sheet_check_outliers.py` |

**Zwei davon sind abgelöst, nicht bloß alt.** `barrier_frozen.csv` trägt
`barr_model`, `barr_dft`, `err_barr`, `rxne_model`, `rxne_dft`, `err_rxne` auf
def2-TZVP; genau dieselben Spalten stehen in `omol25_model_geoms.csv` auf
def2-TZVPD und sind dort die gültigen. `hinge_rows.csv` ist der Vorgänger der
beiden Hinge-Tabellen, mit dem obsoleten Median 1.697 eV Å⁻¹ (Nachfolger:
1.636 und 1.870, siehe `docs/methods_for_paper.md` §6).

**`saddle_residuals.csv` gehört zur Dissertation, nicht zum Paper.** Es
kalibriert die Stufe-1-Stationaritätsschwelle der Kapitel-Pipeline. Diese
Schwelle wird in keinem übergebenen Dokument zitiert.

**Nichts davon wurde nach `attic/` verschoben, und zwar mit Absicht:** jede
dieser Dateien hat mindestens einen lebenden Leser (rechte Spalte). Bei
`omol25_compare.csv` ist es sogar `omol25_model_geoms.py` selbst, das seine
gemeinsamen Spalten dagegen prüft und auf 5·10⁻⁵ Übereinstimmung besteht — ein
Verschieben würde den Bau der Audit-Tabelle brechen. Ein Verschieben hätte
also nichts geordnet, sondern still etwas kaputt gemacht. Die Abgrenzung
leistet diese Tabelle.

---

## paper_reactions.csv — 45 Zeilen, eine je Reaktion

`rxn, nfod, stratum, formula, group_rxn`

**Herkunft.** `nfod` aus `~/fod_ranking.json` (279 Reaktionen; Felder `rxn`,
`nfod`, `nfod_check`, `n_atoms`, `n_elec`, `energy_Ha`). Wie diese 279 zustande
kamen, steht nicht in der Datei — für die Auswahl der 45 wird nur ihre
Rangfolge nach `nfod` gebraucht.

`stratum` wird nicht von Hand gesetzt, sondern die ursprüngliche
Auswahlvorschrift nachgebaut, wörtlich aus
[`pipeline/which_sheet_did_models_learn.py`](../pipeline/which_sheet_did_models_learn.py)
und bestätigt vom Kommentar in
[`pipeline/job_neb_omol25_45.sh`](../pipeline/job_neb_omol25_45.sh):

```
high    res[i]      for i in range(26)                       Ränge 1–26
spread  res[i - 1]  for i in [11,40,68,97,126,154,183,
                              212,240,269]                   Ränge 11,40,…,269
low     res[i]      for i in range(n - 10, n)                Ränge 270–279
```

`res` ist nach `-nfod` sortiert. **rxn0896 hat Rang 11 und fällt in high und
spread**; es wird high zugeordnet, daher 45 statt 46 und `spread` = 9.

`formula` aus dem Referenz-Edukt `~/orca_neb_results/<rxn>/reactant.xyz`, gegen
alle drei Modell-Edukte geprüft. `group_rxn` abgeleitet aus
`omol25_model_geoms.csv`: **unstable**, wenn mindestens eine der drei
Modellzeilen `unstable_ts = 1` hat.

**Prüfungen:** n = 45 · Strata 26/9/10 · `group_rxn` 27 stable / 18 unstable ·
dieselben 45 Reaktionen wie in `omol25_model_geoms.csv` · je Reaktion genau drei
Modellzeilen · Summenformel überall gefunden und zwischen Referenz und allen
drei Modellen identisch · genau eine Überschneidung high/spread · high sind die
Ränge 1–26, low die Ränge 270–279.

---

## neb_runs.csv — 135 Zeilen, eine je (Reaktion, MLIP)

`rxn, model, converged_marker, criterion_met, f_band_final, n_steps`

### Marker und Kriterium sind zwei verschiedene Fakten

**`converged_marker`** ist die Existenz der Datei `<modeldir>/<rxn>/converged`.
Sie entsteht, wenn `relax_neb.run(fmax=cineb_fmax)` `True` zurückgibt —
[`pipeline/uma_neb.py`](../pipeline/uma_neb.py), Zeilen 159–163. Dieser
Rückgabewert bedeutet **nicht** „Toleranz erreicht":

- `NEBOptimizer.run_ode` gibt `True` zurück, sofern `ode12r` keine
  `OptimizerConvergenceError` wirft.
- `ode12r` wirft bei drei Bedingungen (Anfangsresiduum zu groß, Residuum
  wächst zu stark, Schrittweite zu klein). Läuft dagegen die Schleife
  `for nit in range(1, steps + 1)` durch, **fällt die Funktion ohne `raise`
  ans Ende** und liefert `None`.
- `NEBOptimizer.run()` setzt `self.max_steps = steps`, die CI-Phase bekommt
  also erneut die vollen 500 Versuche. Gezählt werden **Versuche**,
  protokolliert nur **angenommene** Schritte — bei zähen Bändern werden viele
  verworfen, das Budget läuft aus, ohne dass es im Log auffällt.

Der Marker heißt damit: *der ODE-Löser ist nicht mit einer Ausnahme
ausgestiegen.*

**`criterion_met`** ist `f_band_final <= 0.05`, also das tatsächlich erreichte
Bandkriterium. Verglichen wird mit einer Toleranz von 5·10⁻⁵, weil die
Logspalte auf vier Nachkommastellen rundet: ein Lauf, der bei 0.049996 anhielt,
steht dort als `0.0500`.

**Die 21 Zeilen, in denen sich beide unterscheiden, sind der eigentliche
Inhalt dieser Tabelle** und namentlich im Skript eingefroren. Die Prüfung
schlägt an, sobald sich die Menge ändert.

| Status | n | stabil | instabil |
|---|---|---|---|
| Marker **und** Kriterium | 112 | 70 | 42 |
| Marker, Kriterium nein | 21 | 11 | 10 |
| kein Marker | 2 | 1 | 1 |
| alle | 135 | 82 | 53 |

Ohne Marker: `rxn0894/uma-s` (0.1191) und `rxn8837/esen` (0.1653).
Die 21 reichen von `rxn1061/uma-m` (0.0699) bis `rxn8837/uma-s` (0.3193), alle
mit 441 bis 638 Logzeilen — passend zum ausgeschöpften Versuchsbudget.

### Die weiteren Spalten

`f_band_final` ist die **projizierte Bandkraft** der letzten Logzeile in
ASE-Konvention, also die größte Kraftnorm je Atom über das Band — nicht die
rohe Kraft am TS-Bild und nicht die größte kartesische Komponente. Alle drei
Größen unterscheiden sich; `omol25_model_geoms.csv` führt die beiden anderen
als `f_model_norm_max` und `f_model_max`.

`n_steps` ist die Zahl der Logzeilen über beide Phasen, nicht der letzte
Schrittindex.

### Robustheit

Ob die Figuren an den 23 nicht sauber konvergierten Zeilen hängen:

| Teilmenge | | stabil | instabil | Faktor |
|---|---|---|---|---|
| alle 135 | `f_model_max` | 0.0299 | 0.0344 | 1.15 |
| alle 135 | `f_dft_max` | 0.0507 | 0.1293 | **2.55** |
| nur `criterion_met` | `f_model_max` | 0.0327 | 0.0338 | 1.03 |
| nur `criterion_met` | `f_dft_max` | 0.0507 | 0.1245 | **2.45** |

Die Trennung der DFT-Restkraft zwischen stabil und instabil geht von Faktor
2.55 auf 2.45. **Die Aussage der Figuren hängt nicht an den ausgeschlossenen
Zeilen.**

### Prüfungen

n = 135 · `neb.log` für alle Zeilen gefunden · `converged_marker` = 1 in genau
133 Zeilen · `criterion_met` = 1 in genau 112 Zeilen · die 21 Diskrepanz-Zeilen
unverändert · die 2 ohne Marker unverändert · kein Lauf erfüllt das Kriterium
ohne Marker.

---

## hinge_t1x.csv und hinge_omol25.csv — der Hinge-Test an zwei Geometrien

`rxn, s2_ts, f_rks, f_bs, ratio, depth_mev, group, group_local` — in
`hinge_t1x.csv` zusätzlich `f_ref`.

Beide Tabellen entstehen aus **einem** Skript,
[`pipeline/hinge_tables.py`](../pipeline/hinge_tables.py), damit die
Spaltendefinitionen garantiert identisch sind. Es löst die früheren
`pipeline/hinge_t1x.py` und `pipeline/hinge_omol25.py` ab.

Nichts in diesem Abschnitt ist Paper-Text. Die formulierte Fassung ist §7 von
[`docs/methods_for_paper.md`](../docs/methods_for_paper.md); `docs/methods_hinge.md`
ist nur noch ein Stub, der dorthin verweist.

### Welche Frage welche Tabelle beantwortet

Der Hinge-Test misst an **unveränderten Kernkoordinaten** zwei Restkräfte: eine
auf der restringierten Fläche (`f_rks`) und eine auf der Fläche, die an diesem
Punkt tatsächlich der Grundzustand ist (`f_bs`). Zwischen den beiden Zahlen
bewegt sich nichts; nur die elektronische Lösung wechselt.

**`hinge_t1x.csv` — 45 Zeilen, die Punkte, an denen die Trainingslabels
stehen.** Geometrie ist der Übergangszustand aus Transition1x selbst (H5-Gruppe
`transition_state`, Niveau ωB97x/6-31G(d)), extrahiert von
[`pipeline/extract_t1x_ts.py`](../pipeline/extract_t1x_ts.py) nach
`~/t1x_ts/<rxn>.xyz`. Kein eigener NEB, keine Nachoptimierung. Diese Tabelle
sagt, was an den Strukturen los ist, auf denen die Modelle trainiert wurden.

Sie trägt deshalb **zwei** Effekte übereinander: den Niveauwechsel
ωB97x/6-31G(d) → wB97M-V/def2-TZVPD und, auf den instabilen Zeilen, den
Flächenwechsel. Der Niveauanteil ist groß — die stabilen Zeilen, wo beide
Flächen zusammenfallen, messen ihn allein und liegen bei Median 0.6088 eV Å⁻¹.
`f_ref` (aus dem H5, Feld `transition_state/wB97x_6-31G(d).forces`) gibt
daneben die Restkraft desselben Punkts auf seinem **eigenen** Niveau, sodass
der Weg `f_ref → f_rks → f_bs` die beiden Anteile trennt.

**`hinge_omol25.csv` — 33 Zeilen, dieselben Sättel auf Trainingsniveau
nachoptimiert.** Geometrie ist `~/orca_neb_omol25/<rxn>/transition_state.xyz`,
also unser eigener RKS-CI-NEB auf wB97M-V/def2-TZVPD
([`pipeline/orca_neb_omol25.py`](../pipeline/orca_neb_omol25.py)). Weil die
Geometrie hier auf demselben Niveau stationär ist, fällt der Niveauversatz
heraus und übrig bleibt **allein der Flächeneffekt**. Das ist das
Kontrollexperiment zu Tabelle 1, nicht ein zweiter Datensatz.

Der Unterschied ist der ganze Befund: das Verhältnis `f_bs / f_rks` liegt an
der Label-Geometrie bei 2.80 (Median, instabile Zeilen) und an der
nachoptimierten bei 32.4 — nicht weil `f_bs` sich stark ändert (1.64 gegen
1.87 eV Å⁻¹), sondern weil der Nenner um mehr als eine Größenordnung fällt
(0.5885 gegen 0.0420 eV Å⁻¹).

### Niveau und Läufe

In beiden Tabellen identisch und wörtlich wie in `omol25_model_geoms.csv`:
wB97M-V/def2-TZVPD, def2/J, RIJCOSX, TightSCF, DEFGRID3, `Thresh 1e-12`,
`TCut 1e-13`, `MaxIter 300`, ORCA 5.0.4. Drei Läufe je Reaktion:

| Lauf | Schlüsselwörter | liefert |
|---|---|---|
| `rks_sp` | `RKS … EnGrad`, keine Stabilitätsanalyse | E_RKS, F_RKS |
| `uks_sp` | `UKS …` + `STABPerform` + `STABRestartUHFifUnstable` | ⟨S²⟩ |
| `uks_engrad` | `UKS … EnGrad MORead` auf den Orbitalen von `uks_sp` | E_BS, F_BS |

| Tabelle | Slurm-Job | Verzeichnis |
|---|---|---|
| `hinge_t1x.csv` | **10773547** | `~/orca_hinge_t1x/<rxn>/` |
| `hinge_omol25.csv` | **10773167** | `~/orca_hinge25/<rxn>/` |

E_BS kommt aus `uks_engrad`, nicht aus `uks_sp`: ORCA liefert für dieselbe
Lösung in einem EnGrad-Lauf eine um rund 2.4·10⁻⁵ Ha andere Energie als in
einem reinen Einzelpunkt. EnGrad gegen EnGrad ist an den stabilen Zeilen
sub-nanohartree genau, EnGrad gegen Einzelpunkt nicht.

`f_rks` und `f_bs` sind `max_i |F_i|` über alle 3N kartesischen Komponenten —
dieselbe Konvention wie `f_dft_max` in `omol25_model_geoms.csv` und **nicht**
die ASE-Norm-je-Atom-Konvention, die das NEB-Kriterium benutzt (siehe
`neb_runs.csv` oben).

### group gegen group_local — zwei verschiedene Klassenspalten

| Spalte | Herkunft | Was sie beschreibt |
|---|---|---|
| `group` | übernommen aus `paper_reactions.csv`, Spalte `group_rxn` | ein **Reaktionslabel**, abgeleitet von den **Modellgeometrien**: unstable, wenn mindestens einer der drei Modell-TS `unstable_ts = 1` hat |
| `group_local` | ⟨S²⟩ des `uks_sp` **an genau dem Punkt, der hier tabelliert ist**, gleiche 0.05-Regel | die Klasse **dieser** Geometrie |

Beide sind gültig und beantworten verschiedene Fragen. Sie **müssen** nicht
übereinstimmen, denn die Klasse hängt an der Geometrie. Alle Prüfungen laufen
gegen `group_local`, weil nur das die hier gerechneten Zahlen beschreibt;
`group` steht importiert daneben.

**Eingefroren: die zwei Wechsler.** In beiden Tabellen weichen genau dieselben
zwei Reaktionen ab, mit demselben Vorzeichen:

| rxn | `group` | `group_local` | ⟨S²⟩ hier (t1x / omol25) |
|---|---|---|---|
| `rxn10054` | unstable | stable | −0.000000 / 0.000000 |
| `rxn1147` | stable | unstable | 0.5542 / 0.5562 |

`rxn1147` ist am Referenzsattel gebrochen, an allen drei Modell-TS nicht;
`rxn10054` umgekehrt — es hat zugleich die flachsten Brechungen überhaupt
(0.6, 1.0, 21.6 meV an den Modellgeometrien). Die Prüfung schlägt an, sobald
sich diese Menge ändert.

Weil die beiden in entgegengesetzte Richtungen wechseln, sind die
Gruppengrößen in beiden Spalten gleich: 27/18 in Tabelle 1, 18/15 in Tabelle 2.

**Eingefroren: die drei Kipp-Zeilen.** Unter den **instabilen** Zeilen der
T1X-Geometrie ist `f_bs < f_rks` bei genau `rxn4113`, `rxn6196`, `rxn7957`;
ihr Verhältnis liegt unter 1 (0.974, 0.395, 0.671). (An den stabilen Zeilen
liegt `f_bs` naturgemäß mal knapp über, mal knapp unter `f_rks` — das deckt die
Nullprobe ab, nicht diese Prüfung.) Das ist kein Widerspruch, sondern der
Niveauanteil: bei `f_rks` um 0.5–1.1 eV Å⁻¹ beherrscht der Niveauversatz beide
Kräfte, und die Reihenfolge kann kippen. An den nachoptimierten Geometrien
tritt es bei **keiner instabilen Zeile** auf. Auch diese Menge ist namentlich
eingefroren.

### Kernzahlen, nach group_local

| Geometrie | Gruppe | n | `f_rks` | `f_bs` | `ratio` |
|---|---|---|---|---|---|
| T1x-Label | stable | 27 | 0.6088 | 0.6087 | 1.00 |
| T1x-Label | unstable | 18 | 0.5885 | 1.6359 | 2.80 |
| TZVPD-optimiert | stable | 18 | 0.0391 | 0.0392 | 1.00 |
| TZVPD-optimiert | unstable | 15 | 0.0420 | 1.8695 | 32.36 |

Mediane in eV Å⁻¹. Diese acht Zeilen sind im Skript als `CORE` **eingefroren**;
weicht eine Zahl um mehr als 5·10⁻⁴ (Kräfte) bzw. 5·10⁻³ (Verhältnisse) ab,
bricht der Lauf ab. Damit können die Tabelle im Text und die Tabelle auf der
Platte nicht stillschweigend auseinanderlaufen.

Dieselben Mediane nach `group` zum Vergleich: T1x stable 0.5916 / 0.6087 / 1.00,
T1x unstable 0.6066 / 1.6359 / 2.49; TZVPD stable 0.0399 / 0.0400 / 1.00,
TZVPD unstable 0.0404 / 1.7201 / 32.36.

### Die 12 in Tabelle 2 fehlenden Reaktionen

Alle zwölf haben denselben Grund, aus den Dateien belegt und nicht
angenommen: kein `converged`-Marker, der Bandoptimierer erreichte fmax 0.05
nicht. Letztes Bandresiduum aus `~/orca_neb_omol25/<rxn>/neb.log`:

| rxn | letzte fmax | | rxn | letzte fmax |
|---|---|---|---|---|
| rxn4519 | 0.0592 | | rxn4060 | 0.0876 |
| rxn7060 | 0.0600 | | rxn0894 | 0.0952 |
| rxn3107 | 0.0619 | | rxn4004 | 0.1056 |
| rxn0101 | 0.0637 | | rxn4003 | 0.1076 |
| rxn7937 | 0.0750 | | rxn1154 | 0.1118 |
| rxn1061 | 0.0847 | | rxn7949 | 0.1170 |

Kreuzung mit `group_local` aus Tabelle 1 (an der NEB-Geometrie existiert für
sie keine Messung): 3 der 12 sind unstable, also 25 % gegen 40 % im vollen
Satz. Der Ausschluss ist damit in instabilen Reaktionen **verdünnt**, nicht
angereichert — die günstige Richtung.

Satz, der als Tabellenfußnote taugt:

> Twelve of the 45 reactions have no CI-NEB transition state at this level —
> the band optimiser did not reach fmax 0.05, last band residual 0.0592 to
> 0.1170 eV Å⁻¹ — and the exclusion is depleted in unstable reactions
> (3 of 12, 25 %, against 40 % overall).

### 13 von 33 mit f_rks > 0.05 — bekannte Eigenschaft, kein Fehler

In Tabelle 2 liegen 13 der 33 Zeilen über 0.05 eV Å⁻¹ auf der restringierten
Fläche, Median über alle 0.0404, Maximum 0.1668 (`rxn1150`):

```
rxn1150 0.1668 s   rxn4522 0.0985 u   rxn4113 0.0810 u   rxn0346 0.0551 u
rxn8832 0.1496 u   rxn4500 0.0901 s   rxn2553 0.0788 s   rxn4498 0.0515 s
rxn4513 0.1292 s   rxn4518 0.0892 u   rxn9246 0.0697 s
rxn1147 0.0985 u   rxn6196 0.0812 u
```

(s/u = `group_local`.) Diese Geometrien sind CI-NEB-Übergangszustände, keine
Sattelpunktoptimierungen — sie sind zum Kriterium 0.05 gelaufen, aber das NEB-
Kriterium ist die **projizierte Bandkraft in ASE-Norm je Atom**, während
`f_rks` die größte kartesische Komponente der rohen Kraft ist. Die beiden
Größen sind nicht dieselbe Zahl, und zusätzlich greift der oben unter
`neb_runs.csv` beschriebene Mechanismus: der `converged`-Marker bedeutet
*der ODE-Löser ist nicht mit einer Ausnahme ausgestiegen*, nicht *die Toleranz
ist erreicht* — in `neb_runs.csv` erfüllen 21 von 133 markierten Läufen das
Kriterium nicht.

Für den Hinge-Test ist das unschädlich: verglichen werden zwei Kräfte am
**selben** Punkt, und `f_rks` ist die gemessene Bezugsgröße, keine
vorausgesetzte Null. Die Trennung trägt trotzdem — der größte `f_rks` (0.1668)
liegt noch immer eine Größenordnung unter dem Median-`f_bs` der instabilen
Zeilen (1.8695).

### Prüfungen

Je Tabelle: alle drei Läufe normal beendet · n = 45 bzw. 33 · `group` für jede
Zeile in `paper_reactions.csv` gefunden · `group_local` konsistent (stabile
Zeilen unter ⟨S²⟩ 0.05) · Nullprobe Kraft an den stabilen Zeilen
`|f_bs − f_rks| < max(10⁻³, 0.005·f_rks)` eV Å⁻¹ · Nullprobe Energie
`|depth| < 1` meV · instabile Zeilen mit `depth > 0` · die zwei
`group`/`group_local`-Wechsler unverändert · die Menge mit `f_bs < f_rks`
unverändert · `f_ref` aus dem H5 für jede Zeile in Tabelle 1.
Übergreifend: für jede der 12 fehlenden Reaktionen ein belegter Grund · die
acht Kernzahlen unverändert gegen den eingefrorenen Stand.

Die gemischte Nullproben-Schranke ist bewusst so gewählt: das
Konvergenzrauschen der SCF skaliert relativ zur Kraftgröße, und an der
Label-Geometrie sind die Kräfte rund fünfzehnmal so groß wie an der
nachoptimierten. Unterhalb von `f_rks` = 0.2 eV Å⁻¹ bleibt die absolute
Schranke 10⁻³ maßgeblich, die Prüfung an der nachoptimierten Geometrie wird
also nicht weicher.

---

## rotation_check.csv — 135 Zeilen, eine je (Reaktion, MLIP)

`rxn, model, s2_rotation, s2_stabperform, dE, scf_cycles_rot, scf_cycles_stab,
verdict_match`

**Die Frage.** OMol25 bricht die Spinsymmetrie mit einer 20°-HOMO–LUMO-Drehung
im β-Raum, `Rotate {HOMO, LUMO, 20, 1, 1}`. Unsere Audit-Tabelle benutzt
stattdessen `STABPerform true` mit `STABRestartUHFifUnstable true`. Finden die
beiden Wege dieselbe Lösung? Diese Tabelle beantwortet das an **allen 135**
Audit-Geometrien, nicht an einer Stichprobe.

**Herkunft.** Slurm-Job **10771382**, 135 Array-Tasks, Ausgaben in
`~/orca_rot_check/<rxn>_<Modell>/ts_rot.out`. Geometrie und Niveau sind
identisch zu den `ts_sp`-Läufen der Audit-Tabelle; der einzige Unterschied ist
der `%scf`-Block. Kein Stabilitätslauf in diesen Rechnungen. Skripte:
[`pipeline/mk_rot_tasks.py`](../pipeline/mk_rot_tasks.py) (Taskdatei, Orbitale
aus `HOMO = n_elec/2 − 1`, `LUMO = n_elec/2`, gegen die 26 handgepflegten
Indexpaare geprüft), [`pipeline/job_rot_check.sh`](../pipeline/job_rot_check.sh)
(Läufe), [`pipeline/rot_check.py`](../pipeline/rot_check.py) (Sammler und
Prüfungen). Unter `orca_om25/` und `orca_rks_sheet/` wurde nichts geschrieben.

**`dE` ist `E_Rotation − E_STABPerform`** in Hartree. Positiv heißt: die
Rotation landet **höher**, hat also die tiefere Lösung verfehlt.
`verdict_match` ist 1, wenn beide ⟨S²⟩ auf derselben Seite von 0.05 liegen.

**Ergebnis.** 134 von 135 stimmen im Urteil überein. Für diese 134 ist
`|dE| ≤ 4.10·10⁻⁷ Ha` — eine Schranke, kein Median, weil die eine Ausnahme fast
drei Größenordnungen darüber liegt.

**Die eingefrorene Abweichung.** `rxn4113/uma-s` ist die einzige Zeile, in der
die beiden Wege verschiedene Lösungen finden:

| | ⟨S²⟩ | E [Ha] | SCF-Zyklen |
|---|---|---|---|
| 20°-Rotation | 0.000000 | −322.370328867312 | 80 |
| STABPerform | 0.128374 | −322.370591167142 | 28 |

`dE = +2.623·10⁻⁴ Ha = 7.14 meV`. Die Richtung ist günstig: **STABPerform
findet die tiefere Lösung**, die Audit-Tabelle steht also auf der besseren von
beiden. Die Gegenrichtung kommt in keiner Zeile vor — das negativste `dE` über
alle 135 ist −8.85·10⁻⁸ Ha (rxn7957/uma-m, 0.0024 meV), reines
Konvergenzrauschen.

Diese eine Zeile ist im Skript **namentlich eingefroren** (`ROT_MISMATCH`).
Vorher brach `rot_check.py` an ihr ab; das war die falsche Behandlung, denn die
Abweichung ist das Ergebnis des Vergleichs und nicht ein Fehler im Lauf. Die
Prüfung schlägt jetzt an, sobald sich die Menge ändert — wie die 21
Diskrepanzzeilen in `neb_runs.csv` und die Wechsler in den Hinge-Tabellen.

**SCF-Zyklen.** Rotation Median 25, max 303; STABPerform Median 18, max 184.
Mehr Zyklen mit verdrehtem Startbild sind erwartbar und kein Befund, solange
⟨S²⟩ am Ende stimmt.

**Reichweite.** Beide Seiten liefen unter ORCA 5.0.4. Es existiert in dieser
Arbeit keine Rechnung mit ORCA 6.0.0, die Versionsdifferenz zu OMol25 bleibt
also unberührt. Der Test deckt allein den Symmetriebruch-Weg ab.

**Prüfungen.** `ts_rot.out` für alle Zeilen vorhanden · alle 135 normal beendet
· Klassenaufteilung 82 stabil / 53 instabil · alle 82 stabilen Zeilen fallen
unter Rotation auf ⟨S²⟩ ≈ 0 zurück (größter Wert 0.000126) · von den 53
instabilen weicht genau `rxn4113/uma-s` ab · keine Zeile mit Rotation unter
STABPerform · Elektronenzahl in allen Outputs gerade.

---

## barrier_spread.csv — 45 Zeilen, eine je Reaktion

`rxn, group_rxn, spread_mev`

**Was die Datei beantwortet.** Jede Reaktion hat drei Übergangszustände, einen
je MLIP. An jedem steht eine DFT-Barriere, alle drei auf demselben Niveau und
aus derselben Rechnung — sie unterscheiden sich also **nur darin, wo der Punkt
liegt**, nicht wie er bewertet wurde. `spread_mev` = max − min über diese drei
Zahlen ist damit der reine Geometrieeffekt. Die Datei speist **Panel b** der
Barrieren-Spannweiten-Figur ([`fig9_5`](../figures/fig9_5_barrier_spread_omol25.png),
`pipeline/plot_omol25_figs.py`) und ist das energetische Gegenstück zu
[`model_ts_rmsd.csv`](model_ts_rmsd.csv), das dieselben drei Punkte rein
geometrisch vergleicht.

**Quelle und Join.** `spread_mev` aus [`omol25_model_geoms.csv`](omol25_model_geoms.csv),
Spalte `barr_dft` (eV, hier in meV umgerechnet), gruppiert über die drei
Modellzeilen. `group_rxn` aus [`paper_reactions.csv`](paper_reactions.csv),
**übernommen und nicht neu abgeleitet** — es ist das Reaktionslabel (unstable,
wenn mindestens einer der drei Modell-TS `unstable_ts = 1` hat).

**Join-Schlüssel ist `rxn`**, in beiden Dateien. Die Prüfung verlangt, dass die
Schlüsselmengen deckungsgleich sind *und* jede Reaktion in der Master genau
drei Modellzeilen hat — ein Join über eine unvollständige Gruppe würde eine zu
kleine Spannweite melden, ohne dass es auffiele.

### Kennzahlen

| Gruppe | n | Median | > 43 meV | min | max |
|---|---|---|---|---|---|
| stable | 27 | 0.335 | 2 | 0.029 | 512.749 |
| unstable | 18 | 10.705 | 5 | 0.292 | 4434.359 |
| alle | 45 | 1.052 | 7 | 0.029 | 4434.359 |

Alles in meV. Der Median der instabilen Gruppe liegt **32-fach** über dem der
stabilen, beide aber weit unter der chemischen Genauigkeit von 43 meV
(1 kcal/mol). Die Schwelle wird hier nur ausgezählt, nicht als Kriterium
verwendet.

**Maximum: `rxn8837`, 4434.359 meV** — 4.4 eV Barrierenunterschied, je nachdem
welches Modell den Sattel geliefert hat. Die sieben Reaktionen über 43 meV:

| rxn | Gruppe | spread [meV] | | rxn | Gruppe | spread [meV] |
|---|---|---|---|---|---|---|
| rxn8837 | unstable | 4434.359 | | rxn8885 | unstable | 337.494 |
| rxn4113 | unstable | 1045.221 | | rxn5691 | unstable | 61.408 |
| rxn0894 | unstable | 662.165 | | rxn7060 | stable | 44.832 |
| rxn7945 | stable | 512.749 | | | | |

Fünf der sieben sind instabil, aber zwei sind es nicht — der Geometrieeffekt
ist nicht auf die gebrochene Gruppe beschränkt.

### Der Gegensatz, der die Figur trägt

Aus `omol25_model_geoms.csv`: **49 der 53 instabilen Zeilen (92 %) haben
`|err_barr| < 0.043 eV`.** Die Barriere an einer instabilen Geometrie ist also
meist genau — der Fehler sitzt nicht darin, wie ein gegebener Punkt bewertet
wird, sondern darin, **welchen Punkt** das Modell liefert. Genau das trennt
Panel b von der Fehlerdarstellung.

### Prüfungen

Schlüsselmengen deckungsgleich · jede Reaktion mit genau drei Modellzeilen ·
`barr_dft` in jeder Modellzeile belegt · `group_rxn` für jede Zeile gefunden ·
`spread_mev` überall vorhanden und nicht negativ · **elf eingefrorene
Kennzahlen** (n, beide Gruppengrößen, beide Mediane, beide Überschreitungs-
zahlen, Name und Wert des Maximums, 49 von 53) — weicht eine ab, bricht der
Lauf ab.
