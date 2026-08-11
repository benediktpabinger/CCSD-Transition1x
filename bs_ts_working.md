# Broken-Symmetry-Übergangszustände — Arbeitsstand

Arbeitsdokument, laufend fortgeschrieben. Alle Zahlen werden aus den
gespeicherten Ergebnissen erzeugt, nicht abgetippt. Die erzeugenden Skripte
liegen unter `pipeline/`; die eingebetteten Blöcke sind unveränderte Ausgaben.

## Die Frage

Der Benchmark vergleicht von MLIPs vorhergesagte Übergangszustände gegen eine
DFT-Referenz aus einem ORCA-NEB bei wB97M-V/def2-TZVP, gerechnet mit
restringierter Wellenfunktion. Bei **19 von 45 Reaktionen** ist diese RKS-Lösung
an der Referenzgeometrie extern instabil — es existiert eine spingebrochene
Lösung, die tiefer liegt. Dort ist die Referenz nicht der Grundzustand.

Daraus folgen zwei Fragen, und sie sind unterschiedlich schwer:

- **Ist unsere Ersatzstruktur richtig?** Eine globale Aussage. Man müsste
  ausschließen, dass irgendwo ein tieferer Sattelpunkt liegt, den niemand
  gesucht hat. Keine lokale Analyse kann das.
- **Ist die Modellvorhersage falsch?** Eine lokale Falsifikation. Ein großer
  Gradient an der vorhergesagten Geometrie entscheidet das, unabhängig davon,
  was sonst auf der Fläche liegt.

> **Vor jeder Interpretation dieser Datei: die dreistufige Regel im nächsten
> Abschnitt lesen.** Sie bindet jede Aussage darüber, welcher Punkt der richtige
> Übergangszustand ist, und hat in diesem Projekt bereits zwei Urteile
> umgedreht — je eines gegen jede Seite.

---

# REGEL — dreistufige Prüfung, bevor ein Punkt als Übergangszustand gilt

**Wer diese Datei liest und daraus Schlüsse zieht: Diese Regel bindet jede
Aussage darüber, welcher von zwei Punkten der richtige Übergangszustand ist.
Eine Einstufung, die eine der drei Stufen überspringt, ist ungültig.**

Ein Punkt zählt nur dann als Übergangszustand einer bestimmten Reaktion, wenn
**alle drei** Bedingungen erfüllt sind:

| Stufe | Frage | Prüfung |
|---|---|---|
| 1 | Ist der Punkt stationär und liegt er tiefer als der Vergleichspunkt? | Gradient, Energie |
| 2 | Ist es ein Sattelpunkt erster Ordnung? | genau eine imaginäre Frequenz |
| 3 | **Gehört der Sattelpunkt zu dieser Reaktion?** | **imaginäre Mode gegen die reaktiven Bindungen** |

Stufe 3 wird regelmäßig vergessen und ist die einzige, die hier ein Urteil
umgedreht hat — zweimal, in beide Richtungen.

## Warum die Energie überhaupt schiedsrichtern kann

Sind zwei Strukturen beide Sattelpunkte erster Ordnung und verbinden sie
dasselbe Edukt mit demselben Produkt, dann läuft die Reaktion über den
**tieferen**. Energie ist damit ein physikalisches Kriterium, und sie behandelt
beide Seiten gleich: sie fragt nicht, wessen Struktur es ist.

Das hat den Vergleich erst möglich gemacht. Jedes frühere Maß fragte, wie weit
das Modell von *unserer* Struktur entfernt liegt — damit war unsere Struktur der
Maßstab, und jeder Zweifel an ihr ging in die Antwort ein.

Zwei Bedingungen müssen erfüllt sein, damit eine Energiedifferenz etwas
bedeutet:

**Beide müssen echte Sattelpunkte sein.** An einem nicht-stationären Punkt sagt
die Energie nichts über Übergangszustände. Das schließt die meisten
Modellgeometrien aus — in der MR-Gruppe liegt der Median des Gradienten bei
0.14 eV/Å, zehn von 56 Geometrien liegen über 0.3 und die schlechteste bei
1.12 eV/Å.

**Beide müssen dieselben Minima verbinden.** Ein tieferer Sattelpunkt, der zu
einer anderen Umlagerung gehört, ist kein Konkurrent. Das ist die Schwachstelle:
Wir können es nicht direkt prüfen, weil die dafür gebaute Endpunktprüfung sich
als untauglich erwiesen hat (siehe *Verworfen*). Die Modenanalyse ist der
verfügbare Ersatz — dehnt die imaginäre Mode dieselben reaktiven Bindungen, liegt
der Sattelpunkt sehr wahrscheinlich auf demselben Pfad.

## Die Falle: ein tieferer Punkt ist nicht automatisch ein tieferer Sattelpunkt

Ein fast-stationärer Punkt unterhalb unseres Sattels hat zwei mögliche Lesarten:

- ein **tieferer Sattelpunkt** — dann ist er der relevante und unserer nicht;
- ein **Minimum bergab vom Übergangszustand** — dann ist er gar kein Kandidat.

Beide sind stationär, beide liegen tiefer. Die Frequenz trennt Minimum von
Sattelpunkt. Sie trennt **nicht** einen Sattelpunkt dieser Reaktion von dem
einer anderen Bewegung. Dafür braucht es Stufe 3.

## Die beiden Spiegelfälle

Sie sind der Beleg, dass die Prüfung symmetrisch ist und nicht zu unseren
Gunsten kalibriert. Beide bestanden Stufe 1 und 2; Stufe 3 hat sie entschieden,
in entgegengesetzte Richtungen.

### rxn1147 — die Modelle liegen hinter dem Übergangszustand

```
Struktur        Anteil  C1-C2 d/dQ  C1-C2 [Å]  C1-O5 d/dQ  C1-O5 [Å]
unser BS-TS      0.601       0.134      3.196       0.943      1.864
UMA-S            0.239       0.053      3.570       0.059      1.497   dE -234 meV
UMA-M            0.217       0.057      3.565       0.072      1.499   dE -231 meV
eSEN             0.230       0.055      3.565       0.068      1.499   dE -232 meV
```

Die Modelle bestanden Stufe 1 (231–234 meV unter unserem Punkt, Gradienten
0.050–0.077 eV/Å) und Stufe 2 (je genau eine imaginäre Frequenz). Danach hatten
sie den relevanten Übergangszustand gefunden, und genau so war es notiert.

Stufe 3 kehrt es um. Die Vorzeichen stimmen überein — das allein hätte in die
Irre geführt. Zwei andere Größen entscheiden: die zu knüpfende C1-O5-Bindung
liegt bei **1.497 Å**, einer normalen Einfachbindung, gegen 1.864 Å bei uns —
sie ist bereits fertig. Und die Mode bewegt sie mit 0.06 gegen unsere 0.94,
dreizehnmal schwächer. Die Modellgeometrie sitzt im Produkttal; der dort
gefundene Sattelpunkt gehört zu einer anderen Bewegung, und die 231 meV sind der
Abfall von einem Übergangszustand zu seinem Produkt, kein tieferer Pass.

### rxn7957 — wir liegen hinter dem Übergangszustand

```
Struktur        Anteil  C1-H7 d/dQ  C1-H7 [Å]  C5-H7 d/dQ  C5-H7 [Å]
unser BS-TS      0.275       0.544      2.462       0.061      1.120
UMA-S            0.551       0.846      1.887       0.310      1.170   dE -890 meV
UMA-M            0.744       1.012      1.866       0.570      1.190   dE -890 meV
eSEN             0.570       0.871      1.884       0.339      1.173   dE -890 meV
```

Dieselbe Prüfung, umgekehrtes Ergebnis. **Unsere** Struktur hat das wandernde
Wasserstoffatom bei 1.120 Å von C5 — eine fertige C-H-Bindung — und bei 2.462 Å
von C1, also abgelöst. Der Transfer ist vorbei. Die Modelle liegen bei 1.87 und
1.19, mitten im Transfer, mit Modenanteilen bis 0.74 gegen unsere 0.275 und
Raten bis 0.570 gegen unsere 0.061. Sie liegen 890 meV tiefer. Die Modelle haben
recht, wir sind hinausgeschossen.

## Praktische Marker für Stufe 3

- **Modenanteil** auf den vier reaktiven Atomen — unter etwa 0.10 sitzt die
  Bewegung woanders im Molekül
- **Bindungsrate** — unter etwa 0.05 berührt die Mode die Reaktionskoordinate
  nicht
- **Bindungslänge selbst** — liegt eine reaktive Bindung bereits bei ihrem
  normalen Wert, ist die Reaktion dort abgeschlossen, gleich welche Frequenz der
  Punkt trägt

Der letzte ist der billigste und war in beiden Spiegelfällen der entscheidende:
er braucht nur die Geometrie, keine Hesse-Matrix. Er ist in den Skripten
bewusst **nicht** automatisiert, weil eine Schwelle dafür an genau den zwei
Fällen kalibriert wäre, die sie entscheiden soll. Die Abstände werden stattdessen
ausgegeben.

---

# Urteil je Reaktion

```
rxn      Gr     N_FOD  Kand  Urteil           Begruendung
------------------------------------------------------------------------------------------------------------------------
rxn7949  high   1.146   0/3  unsere Referenz  Mode reaktiv (max 1.157, Anteil 0.98); kein Modell besteht Stufe 3 und liegt tiefer
rxn8832  high   1.000   0/0  unsere Referenz  Mode reaktiv (max 1.216, Anteil 0.96); Modelle ungeprueft oder hoeher
rxn4113  high   0.960   0/0  unsere Referenz  Mode reaktiv (max 0.979, Anteil 0.74); Modelle ungeprueft oder hoeher
rxn8885  high   0.923   0/0  unsere Referenz  Mode reaktiv (max 0.595, Anteil 0.40); Modelle ungeprueft oder hoeher
rxn6196  high   0.869   0/0  unsere Referenz  Mode reaktiv (max 1.263, Anteil 0.96); Modelle ungeprueft oder hoeher
rxn0346  high   0.847   0/0  unsere Referenz  Mode reaktiv (max 1.012, Anteil 0.67); Modelle ungeprueft oder hoeher
rxn3107  high   0.801   0/0  unsere Referenz  Mode reaktiv (max 0.600, Anteil 0.41); Modelle ungeprueft oder hoeher
rxn8837  high   0.798   0/0  unsere Referenz  Mode reaktiv (max 1.310, Anteil 0.91); Modelle ungeprueft oder hoeher
rxn7060  high   0.788   0/0  unsere Referenz  Mode reaktiv (max 0.583, Anteil 0.57); Modelle ungeprueft oder hoeher
rxn8827  high   0.760   0/0  unsere Referenz  Mode reaktiv (max 1.389, Anteil 0.97); Modelle ungeprueft oder hoeher
rxn0894  high   0.716   0/0  unsere Referenz  Mode reaktiv (max 1.026, Anteil 0.57); Modelle ungeprueft oder hoeher
rxn1147  high   0.725   3/3  strittig         beide bestehen Stufe 3; UMA-S liegt -234 meV tiefer
rxn7957  high   0.684   3/3  strittig         beide bestehen Stufe 3; UMA-S liegt -890 meV tiefer
rxn5691  high   0.778   2/2  MODELLE          unser TS: Mode bewegt die reaktiven Bindungen nicht (max 0.014); UMA-S -164 meV, Mode reaktiv (max 0.527, Anteil 0.45)
rxn4522  high   0.731   3/3  MODELLE          unser TS: Mode nicht bestimmbar; UMA-S -1846 meV, Mode reaktiv (max 0.351, Anteil 0.38)
rxn1320  high   0.968   0/0  offen            unser TS: Mode bewegt die reaktiven Bindungen nicht (max 0.001); kein Modellkandidat besteht Stufe 3
rxn4518  high   0.833   0/0  offen            unser TS: Mode sitzt ausserhalb der reaktiven Atome (Anteil 0.03); kein Modellkandidat besteht Stufe 3
rxn1283  high   0.769   0/0  offen            unser TS: Mode nicht bestimmbar; kein Modellkandidat besteht Stufe 3
rxn5690  mid    0.433   0/0  offen            unser TS: kein konvergierter Sattelpunkt; kein Modellkandidat besteht Stufe 3

  unsere Referenz  11   rxn7949 rxn8832 rxn4113 rxn8885 rxn6196 rxn0346 rxn3107 rxn8837 rxn7060 rxn8827 rxn0894
  strittig          2   rxn1147 rxn7957
  MODELLE           2   rxn5691 rxn4522
  offen             4   rxn1320 rxn4518 rxn1283 rxn5690
```

Nach Auflösung der beiden strittigen Fälle über die Bindungslängen — rxn1147 an
uns, rxn7957 an die Modelle:

| | n | Reaktionen |
|---|---|---|
| unsere Referenz ist besser | **12** | die 11 unstrittigen plus rxn1147 |
| die Modelle sind besser | **3** | rxn5691, rxn4522, rxn7957 |
| keine Seite besteht | 4 | rxn1320, rxn4518, rxn1283, rxn5690 |

## Wie belastbar die 12 wirklich sind

Der Zählstand allein überzeichnet. Die 12 zerfallen in drei sehr
unterschiedliche Gruppen:

| Grundlage | n | Reaktionen |
|---|---|---|
| **direkter Vergleich** — Modellfrequenzen gerechnet, beide Seiten voll geprüft | 2 | rxn7949, rxn1147 |
| Modell liegt tiefer, ist aber **nicht stationär** (Stufe 1 gescheitert) | 2 | rxn4113 (UMA-M −40 meV, Gradient 0.185), rxn8885 (UMA-S −342 meV, Gradient 0.484) |
| **kein Modell liegt tiefer** — es gab nie einen Konkurrenten | 8 | rxn8832 (+12), rxn6196 (+8), rxn0346 (+2), rxn3107 (+1), rxn7060 (+0), rxn8827 (+20), rxn0894 (+68), rxn8837 (+1034) |

**Wo beide Seiten wirklich gegeneinander geprüft wurden, steht es 2 zu 3 gegen
uns.** Modellfrequenzen liegen für fünf Reaktionen vor: rxn7949 und rxn1147
gehen an uns, rxn5691, rxn4522 und rxn7957 an die Modelle. Bei acht der zwölf
lag schlicht kein Modell tiefer — das ist kein gewonnener Vergleich, sondern ein
nicht stattgefundener. Nur bei zweien haben wir einen tieferen Modellpunkt aktiv
entkräftet, und zwar über den Gradienten, nicht über die Mode.

## Vollständige Zuverlässigkeitsliste

```
RELIABILITY LIST — 19 reactions with an externally unstable reference
====================================================================================================

### rxn7949   N_FOD 1.146   [OUR REFERENCE]
    ours clears all three stages; UMA-M lies lower but its mode does not belong to this reaction
    reactive bonds: C3-C5, C4-C5
    ours       n_imag 1   mode fraction 0.98   C3-C5 rate 1.157 at 1.840 A  C4-C5 rate 0.315 at 1.503 A   [from batch]
    UMA-S      dE -519 meV   grad 0.248   n_imag 1   fraction 0.23   C3-C5 rate 0.008 at 2.357 A  C4-C5 rate 0.007 at 1.478 A
    UMA-M      dE -535 meV   grad 0.051   n_imag 1   fraction 0.13   C3-C5 rate 0.003 at 2.404 A  C4-C5 rate 0.011 at 1.477 A
    eSEN       dE -534 meV   grad 0.074   n_imag 1   fraction 0.17   C3-C5 rate 0.008 at 2.410 A  C4-C5 rate 0.009 at 1.477 A

### rxn8832   N_FOD 1.000   [OUR REFERENCE]
    ours clears all three stages; no model lies lower (closest UMA-M +12 meV)
    reactive bonds: C1-C6, C1-C2
    ours       n_imag 1   mode fraction 0.96   C1-C6 rate 1.216 at 1.735 A  C1-C2 rate 0.109 at 1.477 A   [from v2]
    UMA-S      dE +28 meV   grad 0.096   frequency not computed
    UMA-M      dE +12 meV   grad 0.075   frequency not computed
    eSEN       dE +18 meV   grad 0.232   frequency not computed

### rxn4113   N_FOD 0.960   [OUR REFERENCE]
    ours clears all three stages; UMA-M lies -40 meV lower but is not stationary (gradient 0.185 eV/A)
    reactive bonds: O0-C3, N2-C3
    ours       n_imag 1   mode fraction 0.74   O0-C3 rate 0.084 at 3.699 A  N2-C3 rate 0.979 at 3.137 A   [from fromneb]
    UMA-S      dE +1025 meV   grad 0.173   frequency not computed
    UMA-M      dE -40 meV   grad 0.185   frequency not computed
    eSEN       dE +1025 meV   grad 0.186   frequency not computed

### rxn8885   N_FOD 0.923   [OUR REFERENCE]
    ours clears all three stages; UMA-S lies -342 meV lower but is not stationary (gradient 0.484 eV/A)
    reactive bonds: C1-O2, C1-N6
    ours       n_imag 1   mode fraction 0.40   C1-O2 rate 0.595 at 2.086 A  C1-N6 rate 0.024 at 2.483 A   [from v2]
    UMA-S      dE -342 meV   grad 0.484   frequency not computed
    UMA-M      dE +4 meV   grad 0.190   frequency not computed
    eSEN       dE -304 meV   grad 0.375   frequency not computed

### rxn6196   N_FOD 0.869   [OUR REFERENCE]
    ours clears all three stages; no model lies lower (closest UMA-S +8 meV)
    reactive bonds: C2-C5, C2-H10
    ours       n_imag 1   mode fraction 0.96   C2-C5 rate 1.263 at 2.010 A  C2-H10 rate 0.341 at 2.192 A   [from fromneb]
    UMA-S      dE +8 meV   grad 0.090   frequency not computed
    UMA-M      dE +9 meV   grad 0.138   frequency not computed
    eSEN       dE +10 meV   grad 0.140   frequency not computed

### rxn0346   N_FOD 0.847   [OUR REFERENCE]
    ours clears all three stages; no model lies lower (closest UMA-M +2 meV)
    reactive bonds: C5-H10, C2-C5
    ours       n_imag 1   mode fraction 0.67   C5-H10 rate 0.105 at 2.244 A  C2-C5 rate 1.012 at 1.881 A   [from batch]
    UMA-S      dE +5 meV   grad 0.244   frequency not computed
    UMA-M      dE +2 meV   grad 0.173   frequency not computed
    eSEN       dE +9 meV   grad 0.485   frequency not computed

### rxn3107   N_FOD 0.801   [OUR REFERENCE]
    ours clears all three stages; no model lies lower (closest UMA-M +1 meV)
    reactive bonds: C2-O3, C2-N5
    ours       n_imag 1   mode fraction 0.41   C2-O3 rate 0.600 at 2.086 A  C2-N5 rate 0.061 at 2.459 A   [from v2]
    UMA-S      dE +4 meV   grad 0.163   frequency not computed
    UMA-M      dE +1 meV   grad 0.092   frequency not computed
    eSEN       dE +2 meV   grad 0.143   frequency not computed

### rxn8837   N_FOD 0.798   [OUR REFERENCE]
    ours clears all three stages; no model lies lower (closest eSEN +1034 meV)
    reactive bonds: N0-C6, C4-C6
    ours       n_imag 1   mode fraction 0.91   N0-C6 rate 1.310 at 2.046 A  C4-C6 rate 0.056 at 2.331 A   [from batch]
    UMA-S      dE +5469 meV   grad 0.757   frequency not computed
    UMA-M      dE +3352 meV   grad 0.764   frequency not computed
    eSEN       dE +1034 meV   grad 0.101   frequency not computed

### rxn7060   N_FOD 0.788   [OUR REFERENCE]
    ours clears all three stages; no model lies lower (closest UMA-M +0 meV)
    reactive bonds: O0-C1, O0-C5
    ours       n_imag 1   mode fraction 0.57   O0-C1 rate 0.457 at 1.615 A  O0-C5 rate 0.583 at 1.287 A   [from v2]
    UMA-S      dE +22 meV   grad 0.882   frequency not computed
    UMA-M      dE +0 meV   grad 0.334   frequency not computed
    eSEN       dE +46 meV   grad 1.123   frequency not computed

### rxn8827   N_FOD 0.760   [OUR REFERENCE]
    ours clears all three stages; no model lies lower (closest eSEN +20 meV)
    reactive bonds: N0-C5, C4-C5
    ours       n_imag 1   mode fraction 0.97   N0-C5 rate 1.389 at 2.027 A  C4-C5 rate 0.094 at 2.581 A   [from batch]
    UMA-S      dE +34 meV   grad 0.173   frequency not computed
    UMA-M      dE +21 meV   grad 0.134   frequency not computed
    eSEN       dE +20 meV   grad 0.228   frequency not computed

### rxn1147   N_FOD 0.725   [OUR REFERENCE]
    models sit past the transition state: the forming C1-O5 bond is at 1.497 A, a finished single bond, against 1.864 A at ours, and their mode moves it at 0.06 against our 0.94
    reactive bonds: C1-C2, C1-O5
    ours       n_imag 1   mode fraction 0.60   C1-C2 rate 0.134 at 3.196 A  C1-O5 rate 0.943 at 1.864 A   [from batch]
    UMA-S      dE -234 meV   grad 0.077   n_imag 1   fraction 0.24   C1-C2 rate 0.053 at 3.570 A  C1-O5 rate 0.059 at 1.497 A
    UMA-M      dE -231 meV   grad 0.050   n_imag 1   fraction 0.22   C1-C2 rate 0.057 at 3.565 A  C1-O5 rate 0.072 at 1.499 A
    eSEN       dE -232 meV   grad 0.068   n_imag 1   fraction 0.23   C1-C2 rate 0.055 at 3.565 A  C1-O5 rate 0.068 at 1.499 A

### rxn0894   N_FOD 0.716   [OUR REFERENCE]
    ours clears all three stages; no model lies lower (closest UMA-S +68 meV)
    reactive bonds: C4-H8, C0-H8
    ours       n_imag 1   mode fraction 0.57   C4-H8 rate 0.314 at 1.171 A  C0-H8 rate 1.026 at 1.913 A   [from batch]
    UMA-S      dE +68 meV   grad 0.781   frequency not computed
    UMA-M      frequency not computed
    eSEN       dE +319 meV   grad 0.799   frequency not computed

### rxn5691   N_FOD 0.778   [MODELS]
    our mode misses the reactive bonds (max rate 0.014, fraction 0.58); UMA-S clears all three stages and lies -164 meV lower
    reactive bonds: O0-N6, C4-N6
    ours       n_imag 1   mode fraction 0.58   O0-N6 rate 0.014 at 2.912 A  C4-N6 rate 0.009 at 2.507 A   [from batch]
    UMA-S      dE -164 meV   grad 0.154   n_imag 1   fraction 0.45   O0-N6 rate 0.031 at 2.967 A  C4-N6 rate 0.527 at 2.465 A
    UMA-M      dE -101 meV   grad 0.085   n_imag 2   fraction 0.35   O0-N6 rate 0.483 at 2.809 A  C4-N6 rate 0.379 at 2.487 A
    eSEN       dE -163 meV   grad 0.068   n_imag 1   fraction 0.44   O0-N6 rate 0.057 at 2.973 A  C4-N6 rate 0.530 at 2.465 A

### rxn4522   N_FOD 0.731   [MODELS]
    no converged saddle of ours; UMA-S clears all three stages and lies -1845 meV lower
    reactive bonds: O3-C4, N0-O3
    ours       no frequency   [from batch]
    UMA-S      dE -1845 meV   grad 0.075   n_imag 1   fraction 0.38   O3-C4 rate 0.017 at 1.388 A  N0-O3 rate 0.351 at 2.879 A
    UMA-M      dE -1842 meV   grad 0.083   n_imag 1   fraction 0.41   O3-C4 rate 0.018 at 1.389 A  N0-O3 rate 0.361 at 2.882 A
    eSEN       dE -1843 meV   grad 0.073   n_imag 1   fraction 0.42   O3-C4 rate 0.019 at 1.392 A  N0-O3 rate 0.352 at 2.862 A

### rxn7957   N_FOD 0.684   [MODELS]
    we sit past the transition state: C5-H7 is at 1.120 A, a finished C-H bond, and C1-H7 at 2.462 A is already detached; the models have 1.87 and 1.19 with mode rates up to 0.57 against our 0.06
    reactive bonds: C1-H7, C5-H7
    ours       n_imag 1   mode fraction 0.27   C1-H7 rate 0.544 at 2.462 A  C5-H7 rate 0.061 at 1.120 A   [from v2]
    UMA-S      dE -890 meV   grad 0.137   n_imag 1   fraction 0.55   C1-H7 rate 0.846 at 1.887 A  C5-H7 rate 0.310 at 1.170 A
    UMA-M      dE -890 meV   grad 0.113   n_imag 1   fraction 0.74   C1-H7 rate 1.012 at 1.866 A  C5-H7 rate 0.570 at 1.190 A
    eSEN       dE -890 meV   grad 0.109   n_imag 1   fraction 0.57   C1-H7 rate 0.871 at 1.884 A  C5-H7 rate 0.339 at 1.173 A

### rxn1320   N_FOD 0.968   [UNRESOLVED]
    our mode misses the reactive bonds (max rate 0.001, fraction 0.00); no model candidate clears all three stages
    reactive bonds: C2-H6, O0-H6
    ours       n_imag 1   mode fraction 0.00   C2-H6 rate 0.001 at 3.359 A  O0-H6 rate 0.000 at 0.969 A   [from batch]
    UMA-S      dE +252 meV   grad 0.069   frequency not computed
    UMA-M      dE +251 meV   grad 0.044   frequency not computed
    eSEN       dE +251 meV   grad 0.117   frequency not computed

### rxn4518   N_FOD 0.833   [UNRESOLVED]
    our mode misses the reactive bonds (max rate 0.206, fraction 0.03); no model candidate clears all three stages
    reactive bonds: N0-O5, N0-C1
    ours       n_imag 1   mode fraction 0.03   N0-O5 rate 0.039 at 3.334 A  N0-C1 rate 0.206 at 3.269 A   [from batch]
    UMA-S      dE +96 meV   grad 0.058   frequency not computed
    UMA-M      dE +98 meV   grad 0.055   frequency not computed
    eSEN       dE +96 meV   grad 0.051   frequency not computed

### rxn1283   N_FOD 0.769   [UNRESOLVED]
    no converged saddle of ours; no model candidate clears all three stages
    reactive bonds: C4-O5, O2-O5
    ours       no frequency   [from v2]
    UMA-S      dE +48 meV   grad 0.160   frequency not computed
    UMA-M      dE +82 meV   grad 0.125   frequency not computed
    eSEN       dE +50 meV   grad 0.105   frequency not computed

### rxn5690   N_FOD 0.433   [UNRESOLVED]
    no converged saddle of ours; no model candidate clears all three stages
    reactive bonds: C3-H8, C1-C4
    ours       no converged saddle
    UMA-S      grad 0.191   frequency not computed
    UMA-M      grad 0.112   frequency not computed
    eSEN       grad 0.159   frequency not computed

====================================================================================================
OUR REFERENCE    12   rxn7949 rxn8832 rxn4113 rxn8885 rxn6196 rxn0346 rxn3107 rxn8837 rxn7060 rxn8827 rxn1147 rxn0894
MODELS            3   rxn5691 rxn4522 rxn7957
UNRESOLVED        4   rxn1320 rxn4518 rxn1283 rxn5690
```

Als Abbildung: `ts_reliability_table.png`, nach Ausgang gruppiert, jede Zeile mit
dem Beleg, der sie entschieden hat.

---

# Referenzfreie Befunde

Diese Befunde brauchen keine optimierte Struktur und bleiben gültig, auch wenn
sich jeder Broken-Symmetry-Sattelpunkt hier als falsch herausstellen sollte. Sie
sind das Belastbarste, was das Projekt hat.

## 1. Abstand von der Stationarität

Der Gradient an einer Geometrie sagt, ob dort noch eine Kraft wirkt. Ein
Übergangszustand hat keine. Gemessen jeweils auf der Fläche, die dort der
Grundzustand ist: RKS wo die restringierte Lösung extern stabil ist, BS wo nicht.

```
134 Modellgeometrien

Klasse          n   median     mean       Q1       Q3      max   >0.3
---------------------------------------------------------------------
einfach        78   0.0661   0.0822   0.0490   0.0958   0.5909      1
MR             56   0.1375   0.2224   0.0817   0.2002   1.1231     10

Modell        einfach: median   MR: median   Faktor
UMA-S                  0.0653       0.1633      2.5
UMA-M                  0.0563       0.1125      2.0
eSEN                   0.0755       0.1404      1.9

=== zum Vergleich: die Referenzgeometrie selbst ===
  einfach   n= 26  median 0.0434  min 0.0135  max 0.1746
  MR        n= 19  median 1.6974  min 0.1620  max 2.9493

=== auf welcher Flaeche gemessen ===
  einfach   BS: 3  RKS: 75
  MR        BS: 49  RKS: 7

=== groesste Abweichungen in der MR-Gruppe ===
  rxn7060   eSEN    RKS    1.123
  rxn7060   UMA-S   RKS    0.882
  rxn0894   eSEN    BS     0.799
  rxn0894   UMA-S   BS     0.781
  rxn8837   UMA-M   BS     0.764
  rxn8837   UMA-S   BS     0.757
  rxn0346   eSEN    BS     0.485
  rxn8885   UMA-S   BS     0.484
```

**Die RKS-Referenz liegt bei den Multireferenz-Reaktionen 1.70 eV/Å von der
Stationarität entfernt, die Modellvorhersagen bei 0.14** — die Modelle sind dort
zwölfmal näher an einem gültigen Punkt als die Referenz.

**Die Modelle selbst verschlechtern sich nach diesem Maß um Faktor 2, nicht um
25.** Der große Faktor entstand dadurch, dass gegen *unseren* Sattelpunkt
gemessen wurde. Referenzfrei gemessen finden die Modelle weiterhin
fast-stationäre Punkte; was zunimmt, ist die Uneinigkeit darüber, welcher es ist.

## 2. Energie über dem richtigen Sattelpunkt

Dieselbe Größe in beiden Gruppen: der Referenz-Sattelpunkt, wo RKS gültig ist,
unser bestätigter BS-Sattelpunkt, wo nicht.

```
110 Modellgeometrien

dE = Energie an der Modellgeometrie minus Energie am richtigen Sattelpunkt, in meV.

Klasse        n    median  |median|        Q1        Q3        min        max
-----------------------------------------------------------------------------
einfach      78       0.0       0.4      -0.6       0.3     -513.5        3.4
MR           32       9.2      25.2       1.5      36.9     -341.7     5469.4

Modell        einfach |dE|   MR |dE|   Faktor
UMA-S                  0.5      33.7     65.9
UMA-M                  0.3      10.4     34.1
eSEN                   0.4      46.4    110.7

=== Verteilung |dE| ===
  einfach    0-10:  75  10-50:   2  50-200:   0  200-1000:   1  1000-∞:   0
  MR         0-10:  11  10-50:   9  50-200:   1  200-1000:   6  1000-∞:   5

=== MR-Gruppe je Reaktion, |dE| in meV ===
rxn            UMA-S     UMA-M      eSEN
rxn8832         28.1      12.1      17.9
rxn4113       1025.1     -40.0    1024.9
rxn8885       -341.7       3.9    -304.1
rxn6196          8.1       8.8       9.6
rxn0346          5.4       1.5       8.6
rxn3107          4.0       1.4       2.0
rxn8837       5469.4    3352.2    1033.8
rxn7060         22.3       0.2      46.4
rxn8827         33.7      21.3      19.8
rxn1147       -233.8    -230.9    -231.9
rxn0894         68.4         —     319.2
```

75 von 78 Modellgeometrien landen bei den einfachen Reaktionen innerhalb von
10 meV am Sattelpunkt. Bei den Multireferenz-Reaktionen nur 11 von 32; elf
verfehlen um mehr als 200 meV, fünf um mehr als ein Elektronvolt.

## 3. Uneinigkeit der Modelle untereinander

Modell gegen Modell, ohne jede Referenz. Uneinigkeit beweist, dass mindestens
zwei der drei falsch liegen; Einigkeit beweist nichts, weil alle drei dieselben
Trainingsdaten teilen.

```
Barrier disagreement between UMA-S, UMA-M and eSEN
==============================================================================

Every number is the largest pairwise difference between the three
models. No DFT reference enters -- this is model against model.

  TS energy spread   max - min of the DFT energy evaluated at each
                     model's own predicted transition state [meV].
                     With a shared reactant this equals the spread in
                     forward barrier, because the reactant cancels.
  TS geometry        largest pairwise Kabsch RMSD of those transition
                     states [A]
  reactant geometry  same for the relaxed reactants, to check whether
                     the reactant really does cancel

reactant geometries available for 45 of 45 reactions

group                  n    TS energy spread [meV]    TS geometry [A]
                             median            max     median     max
---------------------------------------------------------------------
single-reference      26        0.3          513.5     0.0045  0.4585
multireference        19       13.9         4435.6     0.0509  2.5325

group                  n    reactant geometry [A]
                               median         max
-------------------------------------------------
single-reference      26       0.0002      0.0003
multireference        19       0.0002      0.0119

How many reactions exceed a given disagreement
--------------------------------------------------------------
  TS energy spread >  10 meV   single-reference  2/26   multireference 10/19
  TS energy spread >  50 meV   single-reference  1/26   multireference  5/19
  TS energy spread > 250 meV   single-reference  1/26   multireference  4/19
  TS energy spread >   1 eV    single-reference  0/26   multireference  2/19

Multireference reactions, sorted by disagreement
--------------------------------------------------------------
reaction     N_FOD  TS energy [meV]  TS geom [A]
rxn8837      0.798           4435.6       1.9683
rxn4113      0.960           1065.0       0.7300
rxn8885      0.923            345.6       1.4111
rxn0894      0.716            250.8       2.5325
rxn5691      0.778             62.7       0.2111
rxn7060      0.788             46.2       0.0469
rxn1283      0.769             34.6       0.0639
rxn8832      1.000             16.0       0.0509
rxn7949      1.146             15.3       0.0639
rxn8827      0.760             13.9       0.0620
rxn0346      0.847              7.0       0.0446
rxn5690      0.433              5.5       0.0258
rxn4522      0.731              3.5       0.0163
rxn1147      0.725              2.8       0.0319
rxn4518      0.833              2.7       0.0783
rxn3107      0.801              2.6       0.0410
rxn6196      0.869              1.5       0.0120
rxn1320      0.968              1.4       0.0161
rxn7957      0.684              0.3       0.0192
```

**Das Edukt kürzt sich heraus.** Die drei Modelle stimmen beim relaxierten Edukt
in beiden Gruppen auf 0.0002 Å überein — die Streuung der TS-Energie *ist* damit
die Streuung der Vorwärtsbarriere. Die Uneinigkeit steckt vollständig im
Übergangszustand; Minima sind auch dann leicht, wenn die Reaktion
multireferentiell ist.

**Welches Modell man nimmt, ändert die Barriere bei einfachen Reaktionen um
weniger als ein meV und bei Multireferenz-Reaktionen um bis zu 4.4 eV.** Sechs
Reaktionen tragen fast alles davon — rxn8837, rxn4113, rxn8885, rxn0894,
rxn5691, rxn7060 — und es sind dieselben, die auch in jeder anderen Prüfung
auffallen.

Das ist das praktisch verwertbare Ergebnis: drei Modelle laufen zu lassen und die
Streuung anzuschauen kostet Sekunden und markiert dieselben Reaktionen, für deren
Identifikation die Broken-Symmetry-Optimierung Tage gebraucht hat.

Abbildungen: `model_spread.png` (log-Achsen, alle Punkte sichtbar),
`model_spread_linear.png` (linear, abgeschnitten, Ausreißer beschriftet und
benannt).

---

## Warum die beiden Fragen unterschiedlich schwer sind

**„Unsere Referenz ist richtig" ist eine globale Aussage.** Man müsste zeigen:
es ist ein Sattelpunkt (Frequenz), es ist der richtige (Modenanalyse), er
verbindet die richtigen Minima — und es gibt keinen niedrigeren. Der letzte
Punkt ist prinzipiell nicht beweisbar; eine Energiefläche lässt sich nie
vollständig absuchen. rxn4113 hat das vorgeführt: Der zweite Sattelpunkt war da,
wir hatten ihn nur nicht gesucht.

**„Das Modell liegt falsch" ist eine lokale Falsifikation.** Ist der Gradient
an der Modellgeometrie groß, ist sie kein Stationärpunkt und damit kein
Übergangszustand — unabhängig davon, was sonst auf der Fläche liegt.

## Prüfkette für einen Sattelpunkt

| Stufe | Frage | Verfahren | Kosten |
|---|---|---|---|
| 1 | Existiert eine BS-Lösung? | λ_min_ext aus der Stabilitätsanalyse | mittel |
| 2 | Ist der Punkt stationär? | analytischer Gradient | gering |
| 3 | Ist es ein Sattelpunkt? | numerische Hesse, 6N Gradienten | hoch |
| 4 | Gehört er zu dieser Reaktion? | Projektion der imaginären Mode auf die reaktiven Bindungen, plus die Bindungslängen selbst | keine |

Stufe 4 kostet nichts, weil die Hesse-Matrix ohnehin gespeichert wird, und ist
der schärfste Filter — sie hat drei Strukturen aussortiert, die alle vorherigen
Stufen bestanden hatten, und zusätzlich beide Spiegelfälle entschieden.

Eine fünfte Stufe „verbindet er Edukt und Produkt?" war vorgesehen und ist
gestrichen; warum, steht unter *Verworfen*.

## Modenanalyse über unsere Sattelpunkte

Aus der Hesse-Matrix erhält man den Eigenvektor der imaginären Mode — die
Bergab-Richtung des Sattelpunkts. Zwei Kennzahlen: welcher Anteil der Bewegung
auf den vier Atomen der beiden reaktiven Bindungen liegt, und wie stark sich
diese Bindungen entlang der Mode dehnen. Eine gehinderte Methylrotation ist
ebenfalls ein Sattelpunkt erster Ordnung; ohne diese Prüfung lässt sich der Fall
nicht ausschließen.

```
rxn         v_imag  Anteil reaktiv   d(Bindung)/dQ  [A pro Einheit]           Befund
------------------------------------------------------------------------------------------------------------
rxn0346       1313            0.67   C5-H10 -0.105  C2-C5 -1.012              reaktive Mode
rxn0894        638            0.57   C4-H8 +0.314  C0-H8 -1.026               reaktive Mode
rxn1147        591            0.60   C1-C2 +0.134  C1-O5 -0.943               reaktive Mode
rxn1320        325            0.00   C2-H6 +0.001  O0-H6 -0.000               *** ANDERE BEWEGUNG ***
rxn4518         89            0.03   N0-O5 +0.039  N0-C1 +0.206               teilweise reaktiv
rxn5691        102            0.58   O0-N6 -0.014  C4-N6 -0.009               teilweise reaktiv
rxn7949        735            0.98   C3-C5 -1.157  C4-C5 +0.315               reaktive Mode
rxn8827        596            0.97   N0-C5 +1.389  C4-C5 -0.094               reaktive Mode
rxn8837        823            0.91   N0-C6 +1.310  C4-C6 -0.056               reaktive Mode
rxn3107       1472            0.41   C2-O3 +0.600  C2-N5 -0.061               teilweise reaktiv
rxn7060       2498            0.57   O0-C1 +0.457  O0-C5 -0.583               reaktive Mode
rxn7957        677            0.27   C1-H7 -0.544  C5-H7 +0.061               teilweise reaktiv
rxn8832        653            0.96   C1-C6 -1.216  C1-C2 +0.109               reaktive Mode
rxn8885       1685            0.40   C1-O2 -0.595  C1-N6 +0.024               teilweise reaktiv
rxn4113        156            0.74   O0-C3 +0.084  N2-C3 -0.979               reaktive Mode
rxn6196        784            0.96   C2-C5 +1.263  C2-H10 +0.341              reaktive Mode
```

**Drei Strukturen fallen durch** — rxn1320 (Anteil 0.00), rxn4518 (0.03) und
rxn5691 (Bindungsrate 0.014). Alle drei galten zuvor als frequenzbestätigt.
rxn1320 ist der klarste Fall: die brechende C2-H6-Bindung ist dort von 1.981
auf 3.359 Å gelaufen, das Wasserstoffatom also vollständig abgelöst — die
Optimierung ist über den Übergangszustand hinausgeschossen.

Beachte: `Anteil` und `max rate` können auseinanderlaufen. rxn5691 hat einen
hohen Anteil (0.58) auf den reaktiven Atomen, aber die Bewegung dehnt die
Bindungen nicht (0.014 / 0.009) — die Atome bewegen sich gemeinsam, also
verschiebt oder rotiert das Fragment, statt zu reagieren. Umgekehrt bei rxn4518:
eine spürbare Rate (0.206) bei verschwindendem Anteil (0.03), das heißt fast die
gesamte Bewegung sitzt woanders im Molekül. Beide Kennzahlen müssen erfüllt sein.

## Reaktion für Reaktion — Stabilitäts- und Strukturdaten

| rxn | Gr. | N_FOD | λ_min_ext | ΔE_BS | ⟨S²⟩ | ν_imag | Mode | NEB | Faktor |
|---|---|---|---|---|---|---|---|---|---|
| rxn4518 | high | 0.833 | -0.07780 | -648.5 | 0.842 | 89 | 0.03 / 0.21 | — | 5× |
| rxn7949 | high | 1.146 | -0.06315 | -559.6 | 0.893 | 735 | 0.98 / 1.16 | — | 2× |
| rxn8832 | high | 1.000 | -0.04925 | -428.0 | 0.870 | 653 | 0.96 / 1.22 | 0.107 | 2× |
| rxn1320 | high | 0.968 | -0.04801 | -339.2 | 0.785 | 325 | 0.00 / 0.00 | 1.072 | 1× |
| rxn8837 | high | 0.798 | -0.04401 | -293.9 | 0.741 | 823 | 0.91 / 1.31 | 0.003 | 12× |
| rxn0894 | high | 0.716 | -0.04014 | -190.2 | 0.580 | 638 | 0.57 / 1.03 | 1.142 | 21× |
| rxn4522 | high | 0.731 | -0.03247 | -184.0 | 0.662 | — | — | 0.495 | 12× |
| rxn5691 | high | 0.778 | -0.02902 | -155.9 | 0.629 | 102 | 0.58 / 0.01 | — | 11× |
| rxn0346 | high | 0.847 | -0.02719 | -147.6 | 0.628 | 1313 | 0.67 / 1.01 | 0.176 | 1× |
| rxn1147 | high | 0.725 | -0.02450 | -105.2 | 0.534 | 591 | 0.60 / 0.94 | 0.022 | — |
| rxn7957 | high | 0.684 | -0.02396 | -99.8 | 0.513 | 677 | 0.27 / 0.54 | 0.019 | 4× |
| rxn1283 | high | 0.769 | -0.01389 | -44.5 | 0.419 | — | — | — | 46× |
| rxn8885 | high | 0.923 | -0.01109 | -42.8 | 0.507 | 1685 | 0.40 / 0.60 | — | 71× |
| rxn3107 | high | 0.801 | -0.01255 | -38.8 | 0.409 | 1472 | 0.41 / 0.60 | — | 0× |
| rxn8827 | high | 0.760 | -0.01096 | -27.5 | 0.338 | 596 | 0.97 / 1.39 | 0.366 | 39× |
| rxn7060 | high | 0.788 | -0.00790 | -22.1 | 0.374 | 2498 | 0.57 / 0.58 | — | — |
| rxn6196 | high | 0.869 | -0.00687 | -10.7 | 0.216 | 784 | 0.96 / 1.26 | 0.134 | 7× |
| rxn4113 | high | 0.960 | -0.00846 | -8.4 | 0.140 | 156 | 0.74 / 0.98 | 0.008 | 232× |
| rxn5690 | mid | 0.433 | -0.00268 | -1.3 | 0.068 | — | — | — | 25× |

`Faktor` ist die stärkste Symmetriebrechung über alle vier Geometrien geteilt
durch die an der Referenz. Ein hoher Wert heißt, dass eine Modellgeometrie in
einer viel stärker gebrochenen Region liegt — dort kann ein zweiter
Sattelpunkt sitzen, den eine von der Referenz gestartete Optimierung nie sieht.

Die vollständige Tabelle über alle 45 Reaktionen und alle vier Geometriequellen
(180 Rechnungen) steht in `stability_pipeline_45rxn_table.md`.

## Der Einzelfall rxn4113

Die Reaktion hat zwei getrennte Becken. An der Referenzgeometrie ist die
Symmetriebrechung minimal (ΔE_BS −8.4 meV, ⟨S²⟩ 0.14), an der UMA-M-Geometrie
voll ausgebildet (−1940 meV, 1.01). Der Unterschied steckt in der reaktiven
Bindung:

```
C1-C3:  RKS-Referenz 1.963   UMA-S 1.972   eSEN 1.983   MACE 1.915
        UMA-M        2.938   BS-NEB 2.880
```

Drei unabhängige Belege: der ORCA-BS-NEB findet einen Pfad mit ⟨S²⟩ ≈ 1.0 über
sechs Bilder und einen TS 0.93 Å von der Referenz; zwischen BS-NEB und UMA-M
weicht keine Bindung um mehr als 0.05 Å ab; die PySCF-Optimierung von der
NEB-Geometrie aus bleibt auf 0.008 Å stehen. Frequenz 156 cm⁻¹ mit Modenanteil
0.74 und Bindungsdehnung −0.98.

**UMA-M bekam für rxn4113 den schlechtesten RMSD im Feld** — 0.740 gegen 0.014
für UMA-S — weil gegen eine Referenz gemessen wurde, die im anderen Becken
sitzt. Das einzige Modell, das den richtigen Übergangszustand fand, wurde als
einziges als Versager gezählt.

Die ursprüngliche Optimierung hätte ihn nie gefunden: Sie brach an der
Referenz ab, weil ⟨S²⟩ = 0.14 unter der damaligen Schwelle von 0.3 lag.

**Das ist der Fall, der die Grenze des ganzen Vorgehens zeigt.** Das zweite
Becken wurde nur gefunden, weil eine Modellvorhersage dorthin zeigte. Ohne diesen
Zufall wäre es unentdeckt geblieben, und nichts an der Methodik hätte das
angezeigt.

## Modellbewertung nach Geometrie

Zwei Maße nebeneinander, jede Reaktion gegen die für sie gültige Referenz.
Die reaktive Koordinate (Schwelle 0.10 Å) beantwortet „findet das Modell
denselben Übergangszustand"; der All-Atom-RMSD (0.30 Å) beantwortet „trifft es
auch die Konformation". Beide werden berichtet, weil ein einzelnes Maß die
beiden Fehlerarten vermischt.

```
=== stabil  (26 Reaktionen, 130 Zeilen) ===
Modell        korrekt  RK ok, Konf.  RK daneben   falsch   RC med   RMSD med
UMA-S              24             0           2        0    0.0054     0.0051
UMA-M              24             0           2        0    0.0050     0.0050
eSEN               24             0           1        1    0.0049     0.0054
MACE               21             0           4        1    0.0195     0.0172
MACE+delta         20             0           4        2    0.0376     0.0613

=== BS  (13 Reaktionen, 65 Zeilen) ===
Modell        korrekt  RK ok, Konf.  RK daneben   falsch   RC med   RMSD med
UMA-S               4             1           3        5    0.3739     0.2960
UMA-M               5             0           2        6    0.3692     0.2487
eSEN                4             1           2        6    0.5700     0.6300
MACE                3             1           3        6    0.2907     0.3818
MACE+delta          3             1           3        6    0.4285     0.3305

=== Zeilen, bei denen die beiden Masse widersprechen ===
30 von 195
Gruppe  rxn       Modell            RC     RMSD  Befund
BS      rxn5691   UMA-S          0.055    0.639  RK ok, Konformation daneben
BS      rxn5691   eSEN           0.061    0.630  RK ok, Konformation daneben
BS      rxn8837   MACE           0.052    0.395  RK ok, Konformation daneben
BS      rxn8837   MACE+delta     0.034    0.333  RK ok, Konformation daneben
BS      rxn0894   UMA-S          0.205    0.296  RK daneben, Struktur nah
stabil  rxn1154   MACE           0.446    0.276  RK daneben, Struktur nah
BS      rxn1147   MACE+delta     0.513    0.265  RK daneben, Struktur nah
BS      rxn1147   UMA-S          0.374    0.256  RK daneben, Struktur nah
BS      rxn1147   eSEN           0.369    0.250  RK daneben, Struktur nah
BS      rxn1147   UMA-M          0.369    0.249  RK daneben, Struktur nah
stabil  rxn1154   MACE+delta     0.173    0.214  RK daneben, Struktur nah
BS      rxn7957   UMA-M          0.596    0.212  RK daneben, Struktur nah
BS      rxn7957   eSEN           0.578    0.204  RK daneben, Struktur nah
BS      rxn7957   UMA-S          0.575    0.203  RK daneben, Struktur nah
BS      rxn0346   MACE+delta     0.428    0.188  RK daneben, Struktur nah
```

**Diese Tabelle ist mit Vorsicht zu lesen.** Sie misst gegen unseren
Sattelpunkt, ist also genau das zirkuläre Maß, das die referenzfreien Befunde
oben ersetzen. Bei rxn1147 und rxn7957 zählt sie alle Modelle als Versager,
obwohl bei rxn7957 die Modelle recht haben; bei rxn4113 zählt sie das einzige
richtige Modell als das schlechteste. Sie bleibt hier stehen, weil sie den
Konformations- vom Chemiefehler trennt, nicht als Leistungsurteil.

---

# Verworfen

## Die Endpunktprüfung ist untauglich

Gebaut als Ersatz für einen IRC: entlang der imaginären Mode in beide Richtungen
auslenken, relaxieren, sehen wo man landet. Sie liefert **falsch-negative**
Ergebnisse, und zwar bewiesen:

```
rxn8832   beide Seiten laufen zum selben Minimum   — aber der ORCA-NEB findet
rxn8837   beide Seiten laufen zum selben Minimum     dieselbe Struktur auf
rxn7949   beide Seiten laufen zum selben Minimum     0.003 Å, sie verbindet also
```

Der NEB lokalisiert einen Übergangszustand konstruktionsbedingt *zwischen den
relaxierten Endpunkten*; diese drei verbinden Edukt und Produkt nachweislich.
Umgekehrt hat die Prüfung rxn5691 bestanden, dessen Mode tot ist — also auch
falsch-positiv.

Ursache: die Relaxation ist frei. Sie minimiert in allen 3N Richtungen, also ist
die Information darüber, welche Mode ausgelenkt wurde, nach wenigen Schritten
verloren, und das Ergebnis hängt nur noch davon ab, auf welcher Seite des
tatsächlichen Grats der ausgelenkte Punkt lag.

**Folge:** fünf auf dieser Grundlage vorgenommene Aufwertungen — rxn0346,
rxn0894, rxn3107, rxn7957, rxn8827 — sind zurückgenommen. Ein echter IRC
(massengewichteter steilster Abstieg, kleine Schritte) hätte diesen Fehlermodus
nicht; PySCF hat keinen, ORCA hat einen, bräuchte aber seine eigene Hesse-Matrix.

## Die `tight`-Reparatur ist gescheitert

Diagnose war: rxn1320, rxn4518 und rxn5691 konvergieren in den falschen
Sattelpunkt, weil geomeTRIC dem niedrigsten Hesse-Eigenwert folgt und eine weiche
Torsion unter der Reaktionsmode liegt. Abhilfe: Vertrauensradius 0.005 statt
Standard.

Alle drei sind in **dieselbe Struktur** zurückkonvergiert — rxn1320 auf 0.0009 Å,
rxn4518 auf 0.011 Å, rxn5691 auf 0.027 Å. Der falsche Sattelpunkt ist also
nicht die Folge zu großer Schritte, sondern das, wohin die Optimierung von der
Referenz aus robust läuft. Es braucht einen anderen Startpunkt, keine andere
Schrittweite. Meine Diagnose war falsch.

## Frühere Korrekturen

Zurückgenommene oder revidierte Aussagen aus dem Verlauf, damit sie nicht
weiterwirken:

**Die ⟨S²⟩ > 0.3-Schwelle war falsch.** Sie hat gültige, schwach diradikalische
Übergangszustände als Fehlschläge verworfen — rxn3107 (0.18) und rxn8885 (0.15)
sind echte Sattelpunkte mit den zweit- und dritthöchsten imaginären Frequenzen
im Satz. Drei weitere Reaktionen wurden vor der Optimierung abgewiesen. Das
richtige Kriterium ist das Vorzeichen von λ_min_ext.

**Das `BS_LOST`-Flag ist irreführend.** Es feuert, wenn ⟨S²⟩ *irgendwann* unter
0.3 fällt, nicht am Ende. Fünf so markierte Reaktionen sind frequenzbestätigt.

**Die Diagnose „SCF-Zweigsprung" war falsch.** Der Einbruch während der
Optimierung ist kein Löserfehler, sondern echtes Schwächerwerden der
Symmetriebrechung. λ_min_ext bleibt dabei negativ, sinkt aber — die Optimierung
wandert aus der stark diradikalischen Region heraus.

**Die Newton-Bistabilität war ein Einzelereignis.** Sechs Wiederholungen auf
fünf Nodes ergaben identische Werte; der abweichende Lauf bleibt ungeklärt,
aber unsystematisch.

**Der All-Atom-RMSD als alleiniges Maß ist untauglich.** Neun von 34 gemeldeten
Modellversagern haben Bindungslängen, die auf 0.06 Å stimmen — Konformation,
nicht Chemie. Umgekehrt übersieht er 26 Fälle, in denen die reaktive Koordinate
daneben liegt.

**„Die Modelle brechen bei Multireferenz ein" war zu stark.** Nach dem
referenzfreien Gradientenmaß verschlechtern sie sich um Faktor 2, nicht 25.

**„Die Modelle sitzen auf der BS-Fläche" war zu stark.** Das Gradientenverhältnis
sprach dafür (51 von 52), der direkte geometrische Vergleich stützt es nicht.
Zurückgenommen.

**Die BS-NEB-Route ist der TS-Optimierung unterlegen.** Die Nullmessung zeigt
0.021 Å Streuung für die Optimierung gegen 0.669 Å für den NEB an derselben
Reaktion. `BrokenSym` ist zustandslos und hält die Symmetriebrechung nicht über
das Band — nur 5 von 11 ⟨S²⟩-Profilen sind zusammenhängend.

---

# Methodik, kurz

**Stabilitätsanalyse.** `mf.stability(internal=True, external=True,
return_status=True)` bei wB97M-V/def2-TZVP. λ_min_ext < 0 heißt: eine
spingebrochene Lösung existiert und RKS ist nicht der Grundzustand. Das ist das
Kriterium für die Einteilung in die 19 und die 26.

**BS-Lösung finden.** Zwei Wege: dem Eigenvektor der externen Instabilität
folgen (Route 1), oder ein Triplett rechnen und das β-HOMO kippen (Route 2). Es
muss zweiter Ordnung (Newton) gelöst werden — DIIS lässt die BS-Lösung
zusammenfallen. Über Geometrieschritte wird die **Dichtematrix** weitergereicht,
niemals die MO-Koeffizienten (die sind nur bezüglich der Überlappungsmatrix des
Elternpunkts orthonormal).

**Frequenzen.** Numerische Hesse aus 6N zentralen Differenzen analytischer
Gradienten, δ = 0.01 Bohr. Analytisch geht nicht: VV10 hat weder in PySCF noch
in ORCA 5.0.4 zweite Ableitungen. Jeder ausgelenkte Punkt wird mit der
Dichtematrix des Referenzpunkts gestartet, ⟨S²⟩ wird mitgeschrieben.

**⟨S²⟩ an den Bandenden.** ⟨S²⟩ = 0 an Edukt und Produkt ist richtig, kein
Fehler — dort ist die Struktur geschlossenschalig, die Symmetriebrechung setzt
erst jenseits des Coulson-Fischer-Punkts ein.

---

# Offene Punkte

**Zwei Frequenzrechnungen.** Die Reparaturläufe haben zwei neue Strukturen
erzeugt, für die noch keine vorliegt:

| rxn | gestartet von | Ergebnis | Abstand zur bisherigen Struktur |
|---|---|---|---|
| rxn8885 | eSEN-Geometrie | konvergiert, ⟨S²⟩ 1.028 | **1.475 Å** — ein völlig anderer Punkt |
| rxn1283 | UMA-S-Geometrie | konvergiert, ⟨S²⟩ 1.004 | 0.376 Å — überhaupt zum ersten Mal konvergiert |

rxn8885 steht in der Urteilstabelle derzeit als „unsere Referenz", gestützt auf
eine Struktur bei ⟨S²⟩ = 0.153. Die neue bei 1.028 ist das vermutete zweite
Becken; das Urteil kann sich ändern. Je Rechnung etwa 2–3 h.

**Strukturelle Grenzen, keine Lücken, die man schließen könnte:**

- **Keine Spinprojektion.** Die BS-Lösung ist keine Spineigenfunktion (⟨S²⟩ ≈ 1
  statt 0), trägt also Triplett-Kontamination. Für Geometrien unkritisch, für
  absolute Barrieren nötig (Yamaguchi-Korrektur).
- **Nur ein Funktional** (ωB97M-V). Der Exakt-Austausch-Anteil steuert die
  Neigung zur Symmetriebrechung stark; ob dieselben 19 Reaktionen mit einem
  anderen Funktional instabil wären, ist ungeprüft.
- **Ein tieferer Sattelpunkt lässt sich nie ausschließen.** rxn4113 hat gezeigt,
  was das bedeutet: ein zweites Becken 0.93 Å entfernt, gefunden nur, weil eine
  Modellvorhersage dorthin zeigte.
- **rxn4522** ist in die Wandzeit gelaufen, 332 Schritte ohne Konvergenz. Wir
  haben dort keinen eigenen Sattelpunkt; das Urteil zugunsten der Modelle beruht
  allein darauf, dass ihre drei Strukturen alle drei Stufen bestehen.
- **rxn5690** ist mit ΔE_BS = −1.3 meV der Grenzfall; an der NEB-Geometrie
  existiert keine BS-Lösung mehr. Vermutlich gar kein Multireferenzfall.

---

# Erzeugende Skripte

| Datei | Zweck |
|---|---|
| `pipeline/stability_pipeline.py` | Stabilitätsanalyse über 45 Reaktionen × 4 Geometrien |
| `pipeline/bs_freq.py`, `bs_freq2.py` | numerische BS-UKS-Hesse (`BSFREQ_SRC` / `BSFREQ_OUT`) |
| `pipeline/imag_mode.py` | Modenanalyse — Stufe 3, kostet nichts |
| `pipeline/mode_compare.py` | Vergleich der imaginären Moden zweier konkurrierender Sattelpunkte |
| `pipeline/verdict_final.py` | dreistufige Regel, symmetrisch auf beide Seiten |
| `pipeline/reliability_list.py` | die vollständige Zuverlässigkeitsliste oben |
| `pipeline/plot_spread.py`, `plot_spread_linear.py`, `plot_reliability_table.py` | die drei Abbildungen |
| `pipeline/gradient_comparison.py` | referenzfreier Abstand von der Stationarität |
| `pipeline/energy_above_saddle.py` | \|ΔE\| über dem richtigen Sattelpunkt |
| `pipeline/barrier_spread.py`, `model_spread.py` | Modell gegen Modell, ohne Referenz |
| `pipeline/tsopt_null.py` | Nullmessung: Streuung der TS-Optimierung gegen die des NEB |
| `pipeline/bs_irc.py` | Endpunktprüfung — **verworfen**, falsch-negativ |
