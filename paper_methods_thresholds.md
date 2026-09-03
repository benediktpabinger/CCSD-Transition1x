# Paper Methods — Schwellen

Jede Schwelle dieser Arbeit: woher sie kommt, ob sie gewählt oder festgelegt
ist, wie sie kalibriert wurde, und ob die Aussagen an ihr hängen.

Anlass ist ein Einwand, der zu Recht kommt: *die Schwellen sind arbiträr
gesetzt*. Das stimmt für die meisten. Dieses Dokument sagt, für welche, und was
das für die Ergebnisse bedeutet.

Zahlen aus [results/threshold_sensitivity.txt](results/threshold_sensitivity.txt)
und [results/saddle_residuals.csv](results/saddle_residuals.csv), erzeugt von
`pipeline/threshold_sensitivity.py` und `pipeline/saddle_residuals.py`.

---

## 1 · Die Liste

| Schwelle | Wert | Rolle | Herkunft |
|---|---|---|---|
| Stufe 1, Stationarität | 0.15 eV/Å | Prüfschwelle am Ende | von uns gesetzt, kalibriert (§2) |
| NEB-Abbruch, Climbing | 0.05 eV/Å | Abbruch der Modellläufe | ASE-Konvention, argparse-Default |
| NEB-Bandphase | 0.15 eV/Å | Umschaltpunkt im Lauf | argparse-Default |
| Endpunktrelaxation | 0.05 eV/Å | BFGS-Abbruch | ASE-Konvention |
| Stufe 2, imaginäre Moden | −20 cm⁻¹ | Zählschwelle | von uns gesetzt, unbegründet |
| Stufe 3, Modenanteil | ≥ 0.10 | Modenzuordnung | von uns gesetzt, unbegründet |
| Stufe 3, Bindungsrate | ≥ 0.05 | Modenzuordnung | von uns gesetzt, unbegründet |
| N_FOD-Teilmenge | > 0.5 | Anti-Zirkel-Panel | von uns gesetzt, unbegründet |
| BS-Kollaps | ⟨S²⟩ < 0.05 | Wache in der Pipeline | von uns gesetzt |
| Davidson-Zusammenbruch | \|λ\| > 1 Ha | Wache in der Pipeline | physikalisch plausibel |
| **Instabilität** | **λ_min^ext < 0** | **Prädiktor** | **mathematisch, nicht gewählt** |

Die letzte Zeile ist die einzige, die kein Parameter ist.

---

## 2 · Die Stufe-1-Schwelle, 0.15 eV/Å

### Was sie ist

`GRAD_OK = 0.15` in [pipeline/sweep_summary.py:30](pipeline/sweep_summary.py#L30),
gleichlautend in `verdict_final.py`, `saddle_matrix.py`, `lowest_saddle.py`,
`model_saddle_stats.py`, `plot_saddle_landscape.py`. Eine Struktur gilt als
**nicht stationär**, wenn die grösste Betragskomponente des DFT-Gradienten dort
0.15 eV/Å überschreitet.

### Sie ist kein Konvergenzkriterium

Zum Vergleich, ORCAs eigene Voreinstellung für Geometrieoptimierung, ausgelesen
aus `bs_uks_nebci_prod/rxn8827/tsopt2.out`:

```
Max. Gradient   TolMAXG   3.0e-04 Eh/Bohr  =  0.0154 eV/Å
RMS Gradient    TolRMSG   1.0e-04          =  0.0051 eV/Å
```

ASE-Konvention für Relaxationen: 0.05 eV/Å. **0.15 ist zehnmal lockerer als
ORCAs Default und dreimal lockerer als ASE.**

Das ist beabsichtigt. Ein Konvergenzkriterium beantwortet „ist diese
Optimierung fertig". Diese Schwelle beantwortet „steht diese Struktur überhaupt
in der Nähe eines Stationärpunkts". Für die zweite Frage muss man locker sein,
sonst verwirft man Strukturen als „nicht stationär", die bloss schlecht
auskonvergiert sind — und der Befund wäre trivial.

### Kalibrierung

Gemessen am Restgradienten, den die konvergierten TS-Optimierungen dieser
Arbeit hinterlassen, alle drei Startpunkte am Zielniveau:

```
n = 37    Median 0.0116    Spanne 0.0018 bis 0.0314 eV/Å

Route A  neues Band          n=11   Median 0.0064   0.0018 – 0.0175
Route B  vorhandenes Band    n= 9   Median 0.0132   0.0031 – 0.0314
Route C  Modellgeometrie     n=17   Median 0.0131   0.0068 – 0.0180
```

**0.15 liegt damit 13-fach über dem Median und 4.8-fach über dem
ungünstigsten Fall.**

Ausgeschlossen sind drei Läufe, die nicht stationär bzw. nicht konvergiert
sind und die Kalibrierung sonst verfälschen würden: rxn1283 über Route B
(0.943, Iterationslimit), rxn6196_maxiter50 (0.087, abgebrochene Variante),
rxn7060 über Route C (1.707, fällt durch Stufe 1). Läufe, die Stufe 2 oder 3
verfehlen, bleiben drin — sie sind trotzdem Stationärpunkte und sagen genauso
viel darüber, was eine konvergierte TS-Optimierung hinterlässt.

> **Korrektur gegenüber früheren Fassungen.** Der Docstring von
> `pipeline/model_saddle_stats.py` nannte 0.006 bis 0.011 eV/Å, und diese Zahl
> ist von dort in die Argumentation gewandert. Sie stammt aus der Phase vor den
> Produktionsläufen. Am Zielniveau gemessen ist die Spanne 0.002 bis 0.031, der
> Faktor zum ungünstigsten Fall also 4.8 und nicht 15 bis 25.

### Keine Literaturkonvention

Für genau diese Rolle — eine bewusst lockere Falsifikationsschwelle neben dem
Konvergenzkriterium — ist **keine Referenz bekannt**. Publikationen geben
typischerweise ihr Konvergenzkriterium an und nicht eine zweite, weitere
Grenze. Die Schwelle gehört deshalb als eigene Setzung deklariert, nicht als
Standard zitiert.

---

## 3 · Das NEB-Abbruchkriterium, 0.05 eV/Å

argparse-Default in [pipeline/orca_neb.py:198](pipeline/orca_neb.py#L198), von
dort in jedes Jobskript durchgereicht — ORCA-, CCSD-, UMA-, eSEN- und
MACE-NEBs benutzen alle `--cineb-fmax 0.05`. Dieselbe Zahl steht bei der
Endpunktrelaxation (`BFGS(...).run(fmax=0.05)`).

Herkunft ist die ASE-Konvention. **Nirgends im Repo begründet.**

Das ist hier kein Nachteil, sondern methodisch wichtig: die Zahl wurde im April
2026 gesetzt, bevor irgendetwas über Multireferenzcharakter bekannt war. Für
die Aussage über stille Ausfälle zählt gerade, dass 0.05 *nicht* für diesen
Test gewählt wurde — es ist die Bedingung, unter der ein Anwender das Modell
tatsächlich einsetzen würde.

---

## 4 · Die Doppelung 0.15 / 0.15

Dieselbe Zahl erscheint an zwei unabhängigen Stellen:

```
orca_neb.py:197        --neb-fmax  default 0.15   Umschaltpunkt: ab hier climb=True
sweep_summary.py:30    GRAD_OK   =        0.15    Prüfschwelle an der fertigen Struktur
```

**Folgenlos, und das ist geprüft, nicht geschlossen.** Von 135 Modell-NEB-Läufen
sind 133 als `converged` markiert, haben also die Bandphase bei 0.15 durchlaufen
*und* die anschliessende Climbing-Phase bei 0.05 erreicht. Die beiden Ausnahmen
stehen bei 3.56 (rxn8837 · eSEN) und 0.37 eV/Å (rxn0894 · UMA-S), beide weit von
0.15 entfernt. **Keine ausgewertete Struktur ist an der Bandschwelle
stehengeblieben.**

Zwei Präzisierungen gehören dazu:

Das NEB konvergiert auf der **projizierten** Kraft, `F_model` in den Tabellen
ist die **blanke** Maximalkomponente. Deshalb gibt es sauber konvergierte
Läufe mit `F_model` = 0.13 — verschiedene Grössen, kein abgebrochener Lauf.

Und die Doppelung ist auch inhaltlich harmlos: die eine Zahl misst die
Modellkraft während der Optimierung, die andere die DFT-Kraft an der fertigen
Geometrie.

---

## 5 · Empfindlichkeit — hängen die Aussagen daran?

### Stufe-1-Schwelle variiert

```
    cut  Positive  AUC λ   AUC bin  AUC FOD   stille Ausfälle
   0.05        99   0.764    0.695    0.764         65/87
   0.08        66   0.771    0.734    0.772         39/87
   0.10        51   0.807    0.763    0.759         29/87
   0.15        29   0.836    0.829    0.776         16/87
   0.20        16   0.840    0.796    0.751          9/87
   0.30        12   0.836    0.808    0.704          5/87
   0.50         8   0.838    0.775    0.620          4/87
```

**Was trägt:** AUC(−λ_min) bleibt zwischen 0.764 und 0.840 über eine ganze
Dekade Schwellenvariation, nie in der Nähe von 0.5.

**Was kippt:** der Vorsprung gegenüber N_FOD. Bei 0.05 und 0.08 ist N_FOD
gleich gut oder minimal besser (0.764/0.764, 0.771/0.772). Erst ab 0.10 zieht
die Instabilitätsanalyse davon. Zusammen mit dem ΔAUC-Bootstrap-CI, das die
Null enthält (−0.034 bis +0.166), heisst das: **die Aussage „Instabilität
schlägt N_FOD" ist nicht abgesichert** und muss im Text abgeschwächt werden.

### NEB-Abbruchkriterium variiert

```
Modell meldet < 0.02:   39 Zeilen, davon  5 keine Stationärpunkte = 13 %
Modell meldet < 0.03:   58 Zeilen, davon 12                       = 21 %
Modell meldet < 0.05:   87 Zeilen, davon 16                       = 18 %
Modell meldet < 0.08:  101 Zeilen, davon 17                       = 17 %
Modell meldet < 0.10:  107 Zeilen, davon 19                       = 18 %
Modell meldet < 0.15:  116 Zeilen, davon 23                       = 20 %
```

Die Rate bleibt bei 13 bis 21 %. **Der stille Ausfall hängt nicht an der Wahl
des Abbruchkriteriums.**

### Grenzfälle

**19 von 122 Zeilen liegen innerhalb von ±0.03 um 0.15.** Für die ist das
Urteil fragil — dieselbe Reaktion, drei Modelle, entgegengesetzte Verdikte:

```
rxn4513 · eSEN    F_DFT 0.146   ok
rxn4513 · UMA-M   F_DFT 0.150   ok
rxn4513 · UMA-S   F_DFT 0.154   AUSFALL
```

Das ist die generische Fragilität einer harten Schwelle bei einer stetigen
Grösse, nicht ein Artefakt dieser Wahl. Es ist der Grund, warum die Kernaussagen
im Text nicht auf der Zählung 29 ruhen sollten.

### Hinge-Aussage variiert

```
    cut   stationär RKS   stationär BS
   0.05            7/19           0/19
   0.10           16/19           0/19
   0.15           18/19           0/19
   0.20           19/19           1/19
   0.30           19/19           1/19
   0.50           19/19           2/19
```

„0 von 19" gilt bis 0.162 eV/Å, dem kleinsten F_bs (rxn5690). Darüber wird es
1 von 19. Der **Faktor** zwischen den Spalten, 4× bis 63×, ist schwellenfrei.

---

## 6 · Was ohne jede Schwelle steht

```
Median F_Modell     stabil 0.0315    instabil 0.0316
Median F_DFT        stabil 0.0675    instabil 0.1626
Spearman(−λ_min, F_DFT)        +0.582
Spearman(−λ_min, maxcomp_err)  +0.585
Median F_bs über die 19 MR-Reaktionen   1.697 eV/Å
```

Diese fünf Grössen brauchen keinen Cutoff. Sie tragen den Kern: **die
Modellkraft trennt nicht, die DFT-Kraft trennt, und die Referenzgeometrien
stehen auf der falschen Fläche.** Die Argumentation im Paper sollte auf ihnen
aufbauen und die schwellenbasierten Zählungen als Illustration führen.

---

## 7 · Die eine Schwelle, die keine ist

**λ_min^ext < 0.** Das ist die exakte Grenze zwischen „die restringierte Lösung
ist ein Minimum im Orbitalraum" und „es existiert eine tiefere Lösung". Kein
Parameter, kein Spielraum, kein Tuning möglich.

Dass ausgerechnet der binäre Prädiktor mit AUC 0.829 fast so gut ist wie der
kontinuierliche mit 0.836, ist deshalb bemerkenswert: die Trennung sitzt genau
dort, wo die Physik sie hinlegt, und nicht an einem angepassten Wert.

---

## 8 · Was im Paper stehen sollte

Drei Sätze in der Methodik, plus die Tabelle aus §5 in den Anhang:

> Stationarität wird mit max|F| < 0.15 eV/Å geprüft. Die Schwelle ist bewusst
> lockerer als jedes Konvergenzkriterium (ORCA-Default 0.015, ASE 0.05) und
> liegt eine Grössenordnung über dem, was die konvergierten TS-Optimierungen
> dieser Arbeit hinterlassen (Median 0.012, Spanne 0.002–0.031, n = 37).
> Ergebnisse für Schwellen von 0.05 bis 0.50 eV/Å stehen in Tabelle X; die
> Kernaussagen sind zusätzlich schwellenfrei formuliert.

Und die N_FOD-Aussage abschwächen: nicht „die Instabilitätsanalyse schlägt
N_FOD", sondern „sie ist mindestens ebenbürtig und trennt zusätzlich innerhalb
der N_FOD-Flags (89 % gegen 44 %)".

---

## Offene Punkte

```
Stufe 2, −20 cm⁻¹              nicht kalibriert, nicht variiert
Stufe 3, 0.10 / 0.05           nicht kalibriert, nicht variiert
N_FOD > 0.5 als Teilmenge      willkürlicher Schnitt, Ergebnis nicht variiert
```

Für Stufe 2 und 3 fehlt dieselbe Empfindlichkeitsanalyse. Sie wäre billig — die
Hesse-Matrizen und Modenanalysen liegen vor — und gehört nachgezogen, bevor das
Paper eingereicht wird.
