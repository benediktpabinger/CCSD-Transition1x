# Übergangszustände bei Multireferenzcharakter

*Fassung 2, begonnen 16.08.2026, Stand 18.08.2026. Neu aufgebaut gegenüber `chapter_mr_transition_states.md`:
Story voran, danach neun Abschnitte mit jeweils denselben fünf Blöcken.
Rücknahmen und korrigierte Zahlen stehen gesammelt in Anhang A, die
Reproduktion in Anhang B.*

---

## Änderungsprotokoll

**Basislinie: 17.08.2026, dieser Stand ist eingespeist.** Alles Spätere steht
hier und ist im Text zusätzlich an Ort und Stelle markiert mit

```
[GEÄNDERT TT.MM.]   an der geänderten Stelle
[NEU TT.MM.]        bei einem neu hinzugekommenen Absatz oder Block
```

| Datum | Abschnitt | Was |
|---|---|---|
| 17.08. | Anhang B | Trenntest: Schritt 2+3 auf den Climbing Images der Baseline. **5 von 5** konvergieren und bestehen alle drei Stufen, **3 von 5** auf demselben Punkt. n = 5. |
| 17.08. | Anhang A.12 | dieselbe Aussage nachgezogen: beide Unterschiede wirken, aber für Verschiedenes |
| 17.08. | To-do Punkt 3 | Zwischenstand des Trenntests statt „läuft" |
| 17.08. | To-do Punkt 8 | **offen:** TS-Optimierungen von UMA-M-Geometrien vollständig ins Kapitel ziehen — bisher nur die zehn Triage-Läufe in §6 |
| 18.08. | §8 **ganz** | von 398 auf 200 Zeilen. Neue Rahmung: *geht es überhaupt, und wäre ein Modell, das es kann, ein Gewinn?* Statt einer Methode drei austauschbare Startpunkte. Neue **Sattelmatrix** mit Energieversatz. |
| 18.08. | §8 Aussage | **15 von 19** am Zielniveau, aber kein Verfahren — und in vier Reaktionen liegen zwei gültige Sattelpunkte 194 bis 892 meV auseinander |
| 18.08. | §8 Zahlen | Startpunkt A 11/10, B 9/9, C 18/12, mit Aufwand je Reaktion |
| 18.08. | §8 neu | Absatz zur **prinzipiellen Grenze** des Modells: Kraftfehler zur Laufzeit weder reduzierbar noch von innen erkennbar |
| 18.08. | §8 neu | *Und trotzdem* — die Modellgeometrie ist der billigste Startpunkt und findet dreimal den tieferen Punkt |
| 18.08. | Anhang B | neuer Abschnitt **Wie die Sattelpunkte beschafft wurden**, ~200 Zeilen: die drei Startpunkte im Einzelnen, NEB-TS gegen NEB-CI, der Trenntest, die Frequenzformel, sechs Sackgassen |
| 18.08. | §9 | drei überholte Punkte ersetzt: *kein tieferer Sattelpunkt*, *neun nie versucht*, *prinzipiell nicht ausgeschlossen*. Der Gegenbeleg-Absatz ist zurückgenommen. |
| 18.08. | §0 und global | **Produktionsniveau → Zielniveau**, **Prüfstand / billiges Niveau → Screening-Niveau**, einmal in §0 begründet |
| 18.08. | Überblick | Fix-Absatz und *was offen bleibt* auf den neuen Stand |
| 18.08. | Anhang A.9 | Nachtrag: die Frage ist inzwischen beantwortet, nur mit einer anderen Messung |
| 18.08. | **§10 neu** | Ausblick: lohnt es sich, die Kräfte zu verbessern? Als Startpunktgeber nein, als Vorhersager ja — aber der Hebel ist die Abdeckung der Nahtregion, nicht die Modellgröße |
| 18.08. | To-do | erneuert — erledigt abgetrennt, vier offene Punkte |
| 20.08. | §0 neu | **Kalibrierung der Stufe-1-Schwelle**: 0.15 gegen ORCA-Default 0.015 und ASE 0.05; Restgradient konvergierter TS-Opt 0.002–0.031, Median 0.012, n = 37 → Faktor 13 zum Median, 4.8 zum ungünstigsten Fall |
| 20.08. | §0 neu | **Empfindlichkeitsanalyse**: AUC 0.764–0.840 und stille Ausfälle 13–21 % über eine Dekade Schwellenvariation; 19 von 122 Zeilen liegen aber innerhalb ±0.03 um die Schwelle |
| 20.08. | §0 Rücknahme | die Kalibrierungszahl **0.006–0.011** stammte aus einem veralteten Skript-Docstring und ist zu eng. Richtig ist 0.002–0.031 |
| 20.08. | §2 Vorbehalte | Verweis auf die neue Kalibrierung; ausdrücklich vermerkt, dass Stufe 2 und 3 dieselbe Prüfung noch **nicht** haben |
| 20.08. | §1 Vorbehalte | **N_FOD-Aussage abgeschwächt**: der Vorsprung der Instabilitätsanalyse existiert erst ab Schwelle 0.10 und ist vom ΔAUC-CI nicht abgesichert |
| 20.08. | neu im Repo | `paper_methods_thresholds.md`, `pipeline/threshold_sensitivity.py`, `pipeline/saddle_residuals.py`, `results/threshold_sensitivity.txt`, `results/saddle_residuals.csv` |

## Wie dieses Kapitel gelesen wird

Jeder der neun Abschnitte hat dieselben fünf Blöcke, immer in dieser Reihenfolge:

| Block | Was darin steht |
|---|---|
| **Aussage** | was behauptet wird, in ein bis drei Sätzen |
| **Methode** | wie gerechnet wurde — Eingaben wörtlich, Definitionen, Herkunft der Dateien |
| **Zahlen** | das Ergebnis, immer mit der Kontrollspalte daneben |
| **Kontrollen** | was geprüft wurde, damit die Zahl nicht bloß plausibel ist |
| **Vorbehalte** | was die Aussage *nicht* trägt |

**Die Blöcke stehen auch dann da, wenn sie leer sind.** Wo keine Kontrolle
existiert, steht „keine" — nicht nichts. So ist auf einen Blick sichtbar, welche
Aussage ungeprüft ist.

Ein zweites Prinzip zieht sich durch: **jede Zahl steht neben ihrem Gegenstück
aus dem Satz, in dem die restringierte Lösung stabil ist.** Ohne diese zweite
Spalte wäre keine der Aussagen belastbar — sie würde eine Eigenschaft von
Multireferenzsystemen behaupten, ohne zu zeigen, dass sie anderswo nicht auch
gilt.

### Benennung

**RKS-TS** statt „Referenz", durchgehend.

> Die Struktur, gegen die der Benchmark misst, ist ein Sattelpunkt der
> restringierten Fläche. Wo die restringierte Lösung stabil ist, ist das
> dieselbe Fläche wie die richtige, und der RKS-TS ist eine gültige Referenz.
> Wo sie instabil ist, sind es zwei Flächen — und das Wort „Referenz" trägt dann
> eine Annahme, die nicht mehr gilt.

**BS-TS-Opt** statt „unsere Struktur". Gemeint ist die Struktur aus einer
gebrochen-symmetrischen TS-Optimierung, die am RKS-TS startet — eines von
mehreren Suchverfahren im Kapitel, nicht der Maßstab. Der interne Name
unterstellte genau das.

**BS-Fläche** für die Fläche der gebrochen-symmetrischen Lösung, **RKS-Fläche**
für die der restringierten. Wo beide zusammenfallen, weil die restringierte
Lösung stabil ist, steht **Grundzustandsfläche**.

---

# Überblick — der Bogen

Multireferenzcharakter ist als Problem bekannt. Was fehlt, ist ein Weg, die
betroffenen Reaktionen zu finden, **bevor** man rechnet — und eine Antwort auf
die Frage, was dort eigentlich schiefgeht.

**Es gibt einen billigen Detektor.** Eine Stabilitätsanalyse am RKS-TS,
Minuten pro Reaktion, sagt vorher, ob ein Modell überhaupt einen
gültigen Übergangszustand liefern wird. Sie trifft in 84 von 100 Fällen die
richtige Reaktion, der etablierte Deskriptor N_FOD in 78. Und sie trennt noch
*innerhalb* der Gruppe, die N_FOD gleich behandelt: 89 gegen 44 Prozent gültige
Stationärpunkte. Sie misst also nicht dasselbe genauer, sondern etwas anderes.

**Bevor irgendein Ergebnis gezeigt wird, steht die Regel.** Ein Punkt gilt erst
dann als Übergangszustand, wenn er stationär ist, genau eine imaginäre Mode hat,
und diese Mode die Bindungen *dieser* Reaktion bewegt. Drei Stufen. Sie hat zwei
Urteile umgedreht, je eines gegen jede Seite — das ist der Grund, ihr zu trauen.

**Die Modelle scheitern nicht einzeln, sondern gemeinsam.** Wo die restringierte
Lösung stabil ist, liegen die drei Modelle 0.0045 Å auseinander und ihre
Barrieren streuen um 0.3 meV. Wo sie instabil ist, 13.9 meV, im Extrem 4.4 eV.
Der naheliegende Schluss wäre, sie seien uneinig, *welcher* Sattelpunkt es ist.
Das ist falsch: in **7 von 8** Reaktionen, in denen mehrere Modelle einen
Sattelpunkt treffen, ist es derselbe. Die größten Streuungen kommen aus
Reaktionen, in denen *keines* einen findet.

**Prüft man ihre Strukturen, sind es keine Sattelpunkte.** 96 Prozent gegen 46.

**Die Ursache steht in einer einzigen Zahl.** Das Modell meldet in beiden Gruppen
dieselbe Restkraft, 0.032 eV/Å. Tatsächlich wirken dort 0.067 gegen 0.163. Es
merkt nicht, dass es in Schwierigkeiten steckt — und hält deshalb an, wo es nicht
anhalten dürfte.

**Dann das Scharnier.** Der RKS-TS ist keine schlechte Rechnung. Er ist ein
sauberer Sattelpunkt: 18 von 19 unterschreiten auf seiner eigenen Fläche
0.15 eV/Å. Auf der Fläche, auf der die Reaktion abläuft, ist **keiner von 19**
ein Stationärpunkt, mit Faktoren von 4 bis 63 zwischen beiden Spalten. Eine gute
Antwort auf die falsche Frage. Daraus folgt unmittelbar, dass es nicht genügt,
UKS-Energien auf RKS-Pfade zu setzen.

**Und der Plottwist: es liegt nicht an den Modellen.** Auch ORCA scheitert, auf
denselben Reaktionen. Die eigenen BS-NEB-Läufe enden mit Gradienten von 0.68 bis
2.55 eV/Å und zwei imaginären Moden. Von 19 UKS-NEB-Läufen konvergieren 15, und nur 8 liefern
einen gültigen Sattelpunkt. Die TS-Optimierung ist mit 13 von 19 die beste Methode im
Feld — und liefert trotzdem keine verlässliche Antwort, weil ihr Ergebnis eine
Funktion des Startpunkts ist. Vier unabhängige Werkzeuge, dasselbe Muster,
derselbe Reaktionssatz: **das ist eine Aussage über die Fläche, nicht über eine
Werkzeugklasse.**

**Der Fix.** Ein Schlüsselwort — und damit zugleich viermal strengere
Bandschwellen und eine andere Nachbehandlung.
Statt ORCA die Verfeinerung selbst zu überlassen (`NEB-TS`), endet der Lauf am
Climbing Image (`NEB-CI`); dessen Geometrie geht in eine eigene TS-Optimierung
mit Krümmungsinformation, und der am Climbing Image gefundene elektronische
Zustand wird über `MORead` weitergereicht statt bei jedem SCF neu hergeleitet.
**Von 0 auf 15 von 19** am Zielniveau — aber aus drei Anläufen mit
verschiedenen Startpunkten zusammengesetzt, nicht aus einem Verfahren. Und
wo zwei Startpunkte dieselbe Reaktion lösen, tun sie es nicht immer mit
demselben Punkt: viermal liegen zwei gültige Sattelpunkte 194 bis 892 meV
auseinander.

**Was offen bleibt, steht ausdrücklich da.** Warum Bandverfahren an dieser
Schwelle scheitern, wissen wir nicht — drei Kandidaten, keiner von den anderen
getrennt. Welcher von zwei gültigen Sattelpunkten der relevante ist, kann das
Kapitel nicht entscheiden — die dreistufige Regel prüft Gültigkeit, nicht
Optimalität. Und es gibt noch keine Barrierenhöhe auf der gebrochenen Fläche.

**In einem Satz.** Ein billiger Test sagt vorher, wo es klemmt; dort scheitern
erst die Modelle und dann auch die etablierten Rechenverfahren, alle auf dieselbe
Weise — weil dort zwei Flächen liegen, wo jedes dieser Werkzeuge eine annimmt.
Sucht man den Sattelpunkt erst nach dem Band, mit festgehaltenem Zustand und
mit Krümmung, wird die richtige Fläche zugänglich.

---

# §0 · Aufbau, der für alles gilt

## Der Datensatz

45 Reaktionen aus Transition1x, für die Edukt, Produkt und ein
RKS-Übergangszustand vorliegen. Die Aufteilung entsteht **nicht** durch Auswahl,
sondern durch Messung — externe Stabilitätsanalyse der restringierten Lösung am
RKS-TS:

```
RKS stabil    26 Reaktionen   λ_min ≥ 0, die restringierte Lösung ist ein
                              Minimum im Raum der Wellenfunktionen
RKS instabil  19 Reaktionen   λ_min < 0, ORCA rotiert in eine tiefere,
                              gebrochen-symmetrische Lösung
```

Die 26 sind die **Kontrollgruppe** und nicht bloß ein Anhängsel: fast jede
Aussage dieses Kapitels ist eine Differenz zwischen diesen beiden Spalten.

## Wie die 45 Reaktionen zusammengestellt wurden

Nicht zufällig und nicht nach Ergebnis, sondern **geschichtet nach N_FOD** —
dem etablierten Deskriptor für Multireferenzcharakter. Aus 279 Kandidaten der
FOD-Rangliste:

```
Schicht                              n   Ränge     N_FOD          MR   Kontrolle
oberste 26 nach N_FOD               26     1–26    0.684–1.146    18       8
zehn über die Rangliste verteilt     9   40–269    0.017–0.566     1       8
unterste 10                         10  270–279    0.003–0.014     0      10
------------------------------------------------------------------------------
gesamt                              45    1–279    0.003–1.146    19      26
```

**Der Zweck der Schichtung** ist, das Verhalten über den *ganzen* Bereich des
Multireferenzcharakters zu sehen und nicht nur an den Rändern. Die mittlere
Schicht ist dünn besetzt, deckt aber Ränge 40 bis 269 ab; von den zehn
gezogenen liegt für neun ein Ergebnis vor.

**Die Instabilität fällt monoton mit N_FOD:**

```
oben     18 von 26 instabil     69 %
Mitte     1 von  9              11 %
unten     0 von 10               0 %
```

Das ist erwartungsgemäß und zugleich der Ausgangspunkt für §1: **acht der
obersten 26 sind trotz hohem N_FOD stabil.** Dort trennt die
Stabilitätsanalyse weiter, wo N_FOD nicht mehr trennt.

## Um welche Reaktionen es geht

```
                Summenformeln
MR (19)         C5H5NO 10,  C3H5NO2 9
Kontrolle (26)  C3H5NO2 13, C5H5NO 6, C3H8N2O 4, C2H5NO2 2, C2H3N3O2 1
```

**Der gesamte multireferenzielle Satz besteht aus Umlagerungen von zwei
Molekülen** — C5H5NO mit 12 Atomen und C3H5NO2 mit 11. Die Kontrollgruppe ist
breiter, teilt aber 19 ihrer 26 Reaktionen mit denselben beiden Summenformeln;
der Gruppenvergleich ist insofern chemisch fair. Was fehlt, ist Breite auf der
MR-Seite (§9).

**Grobe Einteilung nach dem, was sich ändert** — aus den zwei Paaren mit der
größten Abstandsänderung: brechen beide, heißt es Fragmentierung; bilden sich
beide, Ringschluss; ist ein Wasserstoff beteiligt, H-Wanderung; sonst
Umlagerung.

```
             Umlagerung  H-Wanderung  Fragmentierung  Ringschluss
MR (19)          13           4             2             0
Kontrolle (26)    6          10             9             1
```

> **Das ist ein Störfaktor und gehört benannt.** Die beiden Gruppen
> unterscheiden sich nicht nur in der Stabilität der restringierten Lösung,
> sondern auch im Reaktionstyp: rechts dominiert die Umlagerung ohne
> Wasserstoff, links die H-Wanderung. Ein Teil der Gruppendifferenz könnte
> daran hängen statt an der Multireferenz. Die Einteilung ist außerdem grob —
> sie kennt nur die zwei stärksten Abstandsänderungen und keine Ringgrößen,
> Ladungen oder Übergangszustandsgeometrien.


**Was in den 19 passiert.** Die veränderten Bindungen sind nach derselben Regel
bestimmt, die Stufe 3 verwendet: die zwei Paare mit dem größten
|d_Produkt − d_Edukt|, beschränkt auf Paare, die auf mindestens einer Seite
gebunden sind. Abstände in Å, Edukt → Produkt.

```
rxn       Formel    was sich ändert
rxn0346   C3H5NO2   C6-H11 bricht 1.09→2.78 | C3-C6  bricht 1.50→2.68
rxn0894   C3H5NO2   C5-H9  bildet 4.01→1.10 | C1-H9  bricht 1.09→3.99
rxn1147   C3H5NO2   C2-C3  bricht 1.54→3.32 | C2-O6  bildet 2.54→1.43
rxn1283   C3H5NO2   C5-O6  bricht 1.42→3.85 | O3-O6  bildet 3.61→1.43
rxn1320   C3H5NO2   C3-H7  bildet 2.89→1.09 | O1-H7  bricht 0.96→2.56
rxn3107   C3H5NO2   C3-O4  bricht 1.41→2.65 | C3-N6  bildet 2.62→1.56
rxn4113   C3H5NO2   O1-C4  bildet 3.48→1.43 | N3-C4  bricht 1.45→3.48
rxn4518   C3H5NO2   N1-O6  bildet 3.33→1.43 | N1-C2  bricht 1.44→2.92
rxn4522   C3H5NO2   O4-C5  bricht 1.37→3.34 | N1-O4  bildet 3.29→1.43

rxn5690   C5H5NO    C4-H9  bildet 3.47→1.09 | C2-C5  bricht 1.49→3.03
rxn5691   C5H5NO    O1-N7  bildet 2.83→1.44 | C5-N7  bricht 1.46→2.33
rxn6196   C5H5NO    C3-C6  bricht 1.47→4.12 | C3-H11 bricht 1.09→3.42
rxn7060   C5H5NO    O1-C2  bricht 1.19→5.08 | O1-C6  bildet 4.48→1.17
rxn7949   C5H5NO    C4-C6  bildet 2.54→1.47 | C5-C6  bricht 1.44→2.46
rxn7957   C5H5NO    C2-H8  bricht 1.11→4.35 | C6-H8  bildet 2.70→1.08
rxn8827   C5H5NO    N1-C6  bildet 3.89→1.45 | C5-C6  bricht 1.48→2.57
rxn8832   C5H5NO    C2-C7  bildet 2.56→1.48 | C2-C3  bricht 1.47→2.47
rxn8837   C5H5NO    N1-C7  bildet 3.58→1.46 | C5-C7  bricht 1.51→2.34
rxn8885   C5H5NO    C2-O3  bricht 1.42→2.67 | C2-N7  bildet 2.64→1.57
```

Das Muster ist einheitlich: **eine Bindung bricht, eine andere bildet sich** —
Umlagerungen, bei denen am Barrierenkamm zwei Zentren ungepaarte Dichte tragen.
Genau dort sitzt die Symmetriebrechung, und genau dazu passt der Befund aus §6,
dass die Endpunkte geschlossenschalig sind und nur der Kamm betroffen ist.

**Eine Reaktion fällt heraus: rxn6196.** Dort brechen *zwei* Bindungen und es
bildet sich keine — eine Fragmentierung. Sie ist zugleich die Reaktion, deren
RKS-TS als einzige auch auf der eigenen Fläche nicht sauber konvergiert (§6),
und die auf dem Screening-Niveau einen Sattelpunkt liefert, den es auf
Zielniveau nicht gibt (§8).

**Erzeugt von:** `pipeline/reaction_table.py` aus
`orca_neb_results/<rxn>/{reactant,product}.xyz`.

## Niveau der Theorie

```
Zielniveau         ωB97M-V/def2-TZVP  def2/J  RIJCOSX  TightSCF
Screening-Niveau   ωB97X/6-31G(d)     TightSCF
Modelle            OMol25-Familie: UMA-S, UMA-M, eSEN
                   trainiert gegen ωB97M-V/def2-TZVPD
```

**Zielniveau** heisst es, weil OMol25 dagegen trainiert ist und alle Aussagen
dieses Kapitels dort gelten sollen. **Screening-Niveau** benennt den Zweck des
zweiten: Methodenentwicklung, nicht Strukturaussage.

Das zweite Niveau existiert, weil ein BS-NEB am Zielniveau 7 bis 45 Stunden
pro Reaktion braucht und dort jede Hesse-Matrix numerisch ist — ωB97M-V trägt
einen VV10-Term ohne analytische zweite Ableitungen. ωB97X hat kein VV10 und
liefert analytische Hesse-Matrizen. Wofür das Screening-Niveau validiert ist
und wofür nicht, steht in §8 und Anhang B.

## Die drei Stufen, operationalisiert

```
Stufe 1   stationär       max|F| < 0.15 eV/Å auf der Grundzustandsfläche
Stufe 2   ein Übergang    genau eine imaginäre Mode
Stufe 3   der richtige    Modenanteil ≥ 0.10 UND Bindungsrate ≥ 0.05
                          auf dem reaktiven Paar DIESER Reaktion
```

**Das reaktive Paar** ist nicht vorgegeben, sondern aus den Endpunkten
abgeleitet: die zwei Atompaare mit dem größten |d_Produkt − d_Edukt|, beschränkt
auf Paare, die auf mindestens einer Seite gebunden sind. Ohne diese
Beschränkung gewinnen Paare, die sich nur durch eine Drehung des ganzen Moleküls
scheinbar verändern.

**Massengewichtet.** Der Modenanteil ist die Summe der quadrierten Auslenkungen
der reaktiven Atome, geteilt durch die Summe über alle Atome, in
massengewichteten Koordinaten. Eine kartesische Auswertung gewichtet Wasserstoff
zu stark und liefert andere Urteile (Anhang A.3).

**Die Bindungsrate** ist |d(r_ij)/dQ| entlang der imaginären Mode, in Å pro
Einheit der normierten Normalkoordinate. Sie unterscheidet eine Mode, die die
reaktiven Atome *bewegt*, von einer, die sie nur mitschwingen lässt.

**Woher die Modenzahl kommt.** Aus ORCAs eigener Ausgabe, nicht aus einer
erneuten Diagonalisierung. Die Projektion lässt Restrotationen bis etwa
−24 cm⁻¹ stehen, die Zählschwelle liegt bei −20 — ein sauberer Sattelpunkt
erster Ordnung käme dann als zweiter Ordnung heraus. Bei rxn1320 druckt ORCA
sechs exakte Nullen und eine Mode bei −404.3, die Nachdiagonalisierung dagegen
−403.9 **und** −23.6. Für den Eigen*vektor* bleibt die Projektion verlässlich,
und nur den braucht Stufe 3. Also: **Zahl von ORCA, Richtung aus der Hesse.**


**[NEU 20.08.] Woher die 0.15 kommt und was sie aushält.** Sie ist gesetzt, aber
kalibriert. Zum Vergleich die Konvergenzkriterien, gegen die sie sich abgrenzt:
ORCAs Voreinstellung für Geometrieoptimierung liegt bei `TolMAXG` 3.0e-04 Eh/Bohr
= 0.0154 eV/Å, die ASE-Konvention bei 0.05. **Die Prüfschwelle ist zehnmal
lockerer als ORCAs Default — absichtlich.** Ein Konvergenzkriterium beantwortet
„ist die Optimierung fertig"; Stufe 1 beantwortet „steht dieser Punkt überhaupt
in der Nähe eines Stationärpunkts". Für die zweite Frage muss man locker sein,
sonst verwirft man bloß schlecht auskonvergierte Strukturen und der Befund wird
trivial.

Gemessen am Restgradienten der konvergierten TS-Optimierungen dieser Arbeit,
alle drei Startpunkte am Zielniveau:

```
n = 37   Median 0.0116 eV/Å   Spanne 0.0018 bis 0.0314

Startpunkt A   n=11   Median 0.0064   0.0018 – 0.0175
Startpunkt B   n= 9   Median 0.0132   0.0031 – 0.0314
Startpunkt C   n=17   Median 0.0131   0.0068 – 0.0180

0.15 liegt 13-fach über dem Median, 4.8-fach über dem ungünstigsten Fall.
```

**Und was daran hängt.** Variiert man die Schwelle über eine ganze Dekade,
bleibt AUC(−λ_min_ext) zwischen 0.764 und 0.840 und der Anteil stiller Ausfälle
zwischen 13 und 21 %. Zwei Dinge kippen aber: unterhalb von 0.10 ist N_FOD
gleich gut wie die Instabilitätsanalyse, und **19 von 122 Zeilen liegen
innerhalb von ±0.03 um die Schwelle** — für die ist das Urteil eine Münze
(rxn4513 liefert bei 0.146, 0.150 und 0.154 zweimal „gültig" und einmal
„Ausfall"). Deshalb ruhen die Kernaussagen dieses Kapitels auf schwellenfreien
Größen: den Mediansvergleichen 0.0315/0.0316 gegen 0.0675/0.1626, der
Rangkorrelation +0.58 und dem Median 1.697 eV/Å in §6.

Vollständige Empfindlichkeitsanalyse in `paper_methods_thresholds.md` und
`results/threshold_sensitivity.txt`, erzeugt von
`pipeline/threshold_sensitivity.py` und `pipeline/saddle_residuals.py`.

> **Zurückgenommen.** Der Docstring von `pipeline/model_saddle_stats.py` nannte
> als Kalibrierung 0.006 bis 0.011 eV/Å, und diese Zahl ist von dort in die
> Argumentation gewandert. Sie stammt aus der Zeit vor den Produktionsläufen.
> Am Zielniveau gemessen ist die Spanne 0.002 bis 0.031 — der Sicherheitsabstand
> zum ungünstigsten Fall ist damit 4.8-fach, nicht 15- bis 25-fach.

## Die ORCA-Kette und warum sie dreiteilig ist

Jede Auswertung an einer gegebenen Geometrie läuft in derselben Reihenfolge:

```
1a  Einzelpunkt mit Stabilitätsanalyse       → Orbitale des Grundzustands
    ! UKS <METHODE> TightSCF
    %scf
      STABPerform true
      STABRestartUHFifUnstable true
      MaxIter 300
    end

1b  Gradient auf eben diesen Orbitalen
    ! UKS <METHODE> TightSCF EnGrad MORead
    %moinp "bs_start.gbw"

1c  Hesse, wo gebraucht
    ! UKS <METHODE> TightSCF NumFreq MORead        (Zielniveau, wegen VV10)
    %freq
      CentralDiff true
      Increment 0.005
    end
```

**Die Dreiteilung ist erzwungen, nicht gewählt.** ORCA erlaubt `STABPerform` nur
mit `RunTyp SinglePoint`; ein Lauf, der zugleich die Stabilität prüft und einen
Gradienten liefert, bricht nach zwei Sekunden ab mit

```
WARNING: Only RunTyp == SinglePoint possible with Stability Analysis!
  ===> : Skipping actual calculation
```

Und ohne `MORead` würde das SCF des Gradientenlaufs neu konvergieren —
womöglich auf die restringierte Lösung, womit der Gradient zur falschen Fläche
gehörte.

**Einheiten.** ORCA druckt dE/dx in Eh/Bohr. Eine Kraft ist das Negative davon;
der Umrechnungsfaktor nach eV/Å ist 51.42208.

## `BrokenSym 1,1` — was es tatsächlich tut

`BrokenSym` ist **zustandslos**. Es leitet den gebrochenen Startzustand bei jedem
SCF neu her: erst wird der Hochspin-Triplett konvergiert, dann ein Spin
umgeklappt und erneut konvergiert. Zwei Folgen, die im Kapitel mehrfach
gebraucht werden:

- In ⟨S²⟩-Statistiken taucht der Zwischenschritt mit ⟨S²⟩ ≈ 2.0 auf. Mit einer
  Schwelle bei 1.8 abtrennen.
- In einem iterativen Verfahren — einem Band, einer Optimierung — wird an jedem
  Schritt neu entschieden, welche Lösung genommen wird. Der Zustand kommt nie zur
  Ruhe. Das ist der Grund, aus dem in §8 stattdessen `MORead` verwendet wird.

## Dichtematrix, nicht Orbitale

Die Symmetriebrechung sitzt in der Spindichte, nicht in einzelnen Orbitalen. Ein
Vergleich einzelner Orbitalenergien über zwei Rechnungen hinweg ist deshalb
nicht aussagekräftig. Die im Kapitel verwendeten Größen sind **⟨S²⟩** und
**ΔE_BS**, die Energiedifferenz zwischen der gebrochenen und der restringierten
Lösung an derselben Geometrie.

---

# §1 · Es gibt einen billigen Detektor

## Aussage

Eine externe Stabilitätsanalyse am RKS-TS — ein Einzelpunkt, Minuten pro
Reaktion — sagt vorher, ob ein Modell für diese Reaktion überhaupt einen
gültigen Übergangszustand liefern wird. Sie ist dem etablierten
Multireferenz-Deskriptor N_FOD deutlich überlegen, und sie misst nachweislich
etwas anderes als er.

## Methode

**Der Prädiktor.** Ein Einzelpunkt ωB97M-V/def2-TZVP am RKS-TS mit interner und
externer Stabilitätsanalyse. Berichtet werden `int_stable`, `ext_stable` und
`lmin_ext` — der kleinste Eigenwert der externen Stabilitätsmatrix. Ablage in
`stab_pipeline/<rxn>/result.json`, Eintrag mit `source == 'RKS-ref'`.

**Die Zielgröße ist referenzfrei.** Gefragt wird nicht, wie weit ein Modell von
einer Vergleichsstruktur entfernt liegt, sondern ob seine eigene Vorhersage ein
Stationärpunkt ist:

```
Positiv (= Problem)   max|F| ≥ 0.15 eV/Å an der Modellgeometrie selbst,
                      gemessen auf der Grundzustandsfläche
```

Der Gradient kommt aus derselben Zweistufenkette wie in §5: `STABPerform` für die
Orbitale, dann `EnGrad MORead` an der unveränderten Modellstruktur. Es geht
**kein Vergleichspunkt** ein — weder der RKS-TS noch eine andere Struktur.

**Warum das wichtig ist.** Eine frühere Fassung nahm den RMSD zur RKS-TS-Struktur
als Zielgröße. Das ist auf der instabilen Seite nicht haltbar, weil der RKS-TS
dort selbst kein Stationärpunkt der richtigen Fläche ist (§6): „Abweichung vom
RKS-TS" wäre ein Abstand, kein Fehler. Und der Test wäre beinahe zirkulär —
Prädiktor und Zielgröße handelten beide von der restringierten Lösung. Die
Einzelheiten des Wechsels stehen in Anhang A.7.

**Das Maß.** Mann-Whitney-AUC, Implementierung wörtlich übernommen aus der
Vorfassung. Lesart: nimm ein Paar aus einer problematischen und einer
unauffälligen Zeile — die AUC ist die Wahrscheinlichkeit, dass der Prädiktor auf
die problematische zeigt. 0.5 ist ein Münzwurf, 1.0 wäre perfekt.

**Die Stichprobe.** 122 Zeilen = 3 Modelle × 42 Reaktionen, jede Zeile eine
Modellvorhersage. Die stabile Seite stammt aus der FOD-geschichteten Auswahl von
`force_error_at_ts.py` (die obersten 26 nach N_FOD, zehn über die Rangliste
verteilt, die untersten zehn); die instabile sind die 19.

## Zahlen

```
von 100 Paaren -- je eine problematische und eine unauffällige Reaktion --
wie oft zeigt der Deskriptor auf die richtige?

  Instabilitätsanalyse   ████████████████▌      84
  N_FOD                  ██████████████         78
  --------------------------------------------------
  Münzwurf               █████████              50
```

```
AUC für „kein Stationärpunkt"        (n = 122, Positive 29 = 24 %)

  -λ_min_ext (kontinuierlich)   0.836
  ext_stable (binär)            0.829
  N_FOD      (kontinuierlich)   0.776

je Modell, AUC von -λ_min_ext
  UMA-S    0.842    n = 41,  davon 12 nicht stationär
  UMA-M    0.862    n = 41,  davon  6
  eSEN     0.837    n = 40,  davon 11
```

**Die Trennung, ohne den Umweg über die AUC:**

```
                   Strukturen   Stationärpunkte   Median max|F| [eV/Å]
  RKS stabil           78        74     95 %           0.067
  RKS instabil         44        19     43 %           0.163
```

**Das Argument gegen „das misst doch dasselbe".** Nimm nur die Strukturen, die
N_FOD ohnehin als verdächtig meldet, und teile sie noch einmal:

```
  N_FOD > 0.5,  RKS stabil       27 Strukturen     89 % Stationärpunkte
  N_FOD > 0.5,  RKS instabil     41                44 %
```

N_FOD hätte beide gleich behandelt; die Stabilitätsanalyse trennt sie um den
Faktor zwei. Zwei Fälle zum Anfassen: **rxn7945** und **rxn7937** haben hohes
N_FOD und trotzdem stabile Übergangszustände — bei ihnen sitzt die Brechung am
Produkt, nicht am Barrierenkamm. N_FOD sieht sie, kann aber nicht sagen, *wo*
sie sitzt, und genau darauf kommt es an.

## Kontrollen

- **Zwei unabhängige Wege auf dieselbe Trennung.** Die 95 % / 43 % oben sind
  allein aus Gradienten gerechnet, ohne eine einzige Hesse-Matrix. Die 96 % /
  46 % in §4 stammen aus der vollen dreistufigen Kette an einer anderen
  Stichprobe. Beide Zahlenpaare stimmen innerhalb weniger Prozentpunkte überein.
- **Der Prädiktorwechsel ist gegengerechnet**, nicht bloß behauptet: dieselben
  Prädiktoren, dieselbe AUC-Implementierung, nur die Zielgröße getauscht.
- **Modellweise Streuung geprüft.** 0.842 / 0.862 / 0.837 — die Vorhersagekraft
  hängt nicht an einem einzelnen Modell.

## Vorbehalte

- **Die 42 Reaktionen sind nicht der Benchmark-Satz.** Die stabile Seite ist
  FOD-geschichtet ausgewählt, also bewusst auf schwierige Fälle hin; das macht
  die Kontrollgruppe strenger, aber nicht repräsentativ für beliebige
  Reaktionen.
- **MACE und MACE+delta fehlen**, weil für sie keine Gradienten an den
  Modellgeometrien vorliegen. In der alten, referenzbasierten Fassung fielen sie
  aus dem Muster (AUC 0.70, bei N_FOD mit 0.47 schlechter als Raten). Ob das
  referenzfrei ebenso wäre, ist ungeprüft.
- **Der Vorsprung des kontinuierlichen Eigenwerts über sein binäres Abbild
  verschwindet** — 0.836 gegen 0.829. Er war ein Artefakt der referenzbasierten
  Zielgröße. Praktisch heißt das: **„instabil ja/nein" genügt**, der Zahlenwert
  muss nicht berichtet werden.
- **Die Schwelle 0.15 eV/Å** in der Zielgröße ist die Stufe-1-Schwelle aus §0 und
  damit nicht unabhängig gewählt; eine andere Schwelle verschöbe die Zahl der
  Positiven und damit die AUC. **[GEÄNDERT 20.08.] Nachgerechnet:** über cut
  0.05 bis 0.50 bleibt AUC(−λ_min_ext) zwischen 0.764 und 0.840, die Zahl der
  Positiven fällt dabei von 99 auf 8. Die Höhe der AUC ist also robust.
- **[NEU 20.08.] Der Vorsprung gegenüber N_FOD ist es nicht.** Bei cut 0.05 und
  0.08 ist N_FOD gleich gut oder minimal besser (0.764 gegen 0.764, 0.771 gegen
  0.772); erst ab 0.10 zieht die Instabilitätsanalyse davon. Zusammen mit dem
  ΔAUC-Bootstrap-CI von [−0.034, +0.166], das die Null enthält, heißt das:
  **„Instabilität schlägt N_FOD" ist nicht abgesichert.** Belegt ist, dass sie
  mindestens ebenbürtig ist und zusätzlich innerhalb der N_FOD-Flags trennt
  (89 % gegen 44 %). Einzelheiten in `paper_methods_thresholds.md`.
- **Es ist eine Vorhersage über Modelle, nicht über Rechenverfahren.** Dass die
  Instabilität auch für ORCA-NEB und TS-Optimierung Schwierigkeiten vorhersagt,
  wird in §7 gezeigt, aber nicht als AUC quantifiziert.

**Erzeugt von:** `pipeline/predictor_reffree.py`. Die alte, referenzbasierte
Fassung bleibt als `pipeline/sep_analysis.py` erhalten
(`stability_vs_fod_separation.txt`).

---

# §2 · Das Instrument, bevor irgendein Ergebnis kommt

## Aussage

Ein Punkt gilt in diesem Kapitel erst dann als Übergangszustand einer Reaktion,
wenn er drei Prüfungen besteht: stationär, genau eine imaginäre Mode, und die
Mode bewegt die Bindungen *dieser* Reaktion. Die Regel ist festgelegt, bevor
Ergebnisse gezeigt werden, und sie wird auf **alle** Kandidaten gleich
angewandt — auf Modellvorhersagen wie auf eigene Rechnungen.

## Methode

```
Stufe 1   stationär       max|F| < 0.15 eV/Å
                          Gradient auf der Grundzustandsfläche:
                          RKS wo die restringierte Lösung extern stabil ist,
                          BS wo nicht
Stufe 2   ein Übergang    genau eine imaginäre Mode
                          Zählung aus ORCAs Ausgabe, Schwelle -1.0 cm⁻¹
Stufe 3   der richtige    Modenanteil ≥ 0.10 UND Bindungsrate ≥ 0.05
                          massengewichtet, auf dem reaktiven Paar
```

**Reaktives Paar.** Aus den beiden Endpunktgeometrien: die zwei Atompaare mit dem
größten |d_Produkt − d_Edukt|, beschränkt auf Paare, die auf mindestens einer
Seite gebunden sind.

**Modenanteil.** Summe der quadrierten Auslenkungen der reaktiven Atome durch die
Summe über alle Atome, in massengewichteten Koordinaten.

**Bindungsrate.** |d(r_ij)/dQ| entlang der imaginären Mode, für das reaktive
Paar, in Å pro Einheit der normierten Normalkoordinate.

**Arbeitsteilung zwischen ORCA und Nachauswertung.** Die *Zahl* der imaginären
Moden kommt aus ORCAs Ausgabe, die *Richtung* aus der nachdiagonalisierten und
projizierten Hesse. Grund in §0: die Projektion lässt Restrotationen bis −24 cm⁻¹
stehen, während die Zählschwelle bei −20 liegt.

**Symmetrische Anwendung.** Dieselbe Regel, dieselben Schwellen, derselbe Code
für Modellstrukturen, RKS-TS, BS-TS-Opt, NEB-Ergebnisse. Es gibt keinen
Kandidaten, der von einer Stufe ausgenommen ist.

## Zahlen

Die Regel selbst produziert keine Zahlen — sie erzeugt Urteile, die in §4, §7 und
§8 auftauchen. Was hier gehört, ist ihr Wirkungsnachweis:

```
Urteile, die die Regel gegenüber der naiven Auswertung umgedreht hat:   2
  eines zugunsten der Modelle    ein Punkt, den wir verworfen hätten
  eines zu ihren Lasten          ein Punkt, den wir akzeptiert hätten
```

Beide Fälle sind in §7 im Einzelnen aufgeführt. Eine Regel, die nur in eine
Richtung korrigiert, wäre verdächtig; diese korrigiert in beide.

## Kontrollen

- **Zwei Spiegelfälle**, je einer pro Richtung — der Beleg, dass die Regel nicht
  auf ein gewünschtes Ergebnis hin gebaut ist.
- **Die Modenrichtung ist robust gegen die numerische Ungenauigkeit der Hesse.**
  Nachgerechnet mit zentralen statt vorwärts Differenzen (§8) ändern sich
  Modenanteil und Bindungsrate erst in der dritten Stelle: 0.70/1.072 → 0.68/1.039,
  0.97/1.296 → 0.97/1.276, 0.97/1.389 → 0.97/1.390.
- **Kreuzvalidierung der Hesse-Matrizen** zwischen ORCA und PySCF, siehe
  Anhang B.

## Vorbehalte

- **Die Schwellen sind gesetzt, nicht hergeleitet.** 0.15 eV/Å, 0.10 und 0.05
  sind Konventionen. Sie sind vor der Auswertung festgelegt und nirgends
  nachträglich verschoben worden, aber sie sind nicht aus einer Theorie
  abgeleitet. **[GEÄNDERT 20.08.]** Für 0.15 liegt inzwischen eine Kalibrierung
  und eine Empfindlichkeitsanalyse vor (§0): 13-fach über dem Restgradienten
  konvergierter TS-Optimierungen, und die Aussagen halten über eine Dekade
  Schwellenvariation. Für die Stufe-3-Schwellen 0.10 und 0.05 sowie für die
  −20 cm⁻¹ in Stufe 2 steht dieselbe Prüfung **aus**.
- **Stufe 3 prüft die Richtung, nicht die Verbindung.** Die eigentlich richtige
  Prüfung — verbinden zwei konkurrierende Sattelpunkte dieselben Minima? — wäre
  ein IRC von beiden aus. Der ist versucht und **verworfen** worden, weil er
  falsch-negativ ausfiel (Anhang A.10). Die Modenanalyse ist ein Ersatz.
- **Stufe 1 misst auf der Grundzustandsfläche.** Welche das ist, entscheidet die
  Stabilitätsanalyse. Damit hängt Stufe 1 von §1 ab; eine falsche Einstufung
  würde den Gradienten auf der falschen Fläche messen.
- **Ein Punkt, der alle drei Stufen besteht, ist damit nicht der *tiefste*
  Sattelpunkt** der Reaktion. Die Regel prüft Gültigkeit, nicht Optimalität
  (§9).

**Erzeugt von:** `pipeline/verdict_final.py` (die Regel), `pipeline/imag_mode.py`
(Modenanalyse), `pipeline/stage3_new.py` (Stufe 3 über mehrere
Ergebnisverzeichnisse), `pipeline/mode_compare.py` (Vergleich zweier
konkurrierender Sattelpunkte).

---

# §3 · Die Modelle scheitern gemeinsam, nicht einzeln

## Aussage

Wo die restringierte Lösung instabil ist, hören die Modelle auf, sich einig zu
sein — **in der Energie**. In der Struktur bleiben sie einig: wo mehrere von
ihnen überhaupt einen Sattelpunkt treffen, ist es in 7 von 8 Fällen derselbe.
Die großen Energiestreuungen stammen aus Reaktionen, in denen *keines* einen
findet. Der Mechanismus ist also nicht Uneinigkeit, sondern **gemeinsames
Verfehlen** — die Modelle hören auf, mit der Referenz und miteinander in der
Energie übereinzustimmen, weil keines von ihnen auf einem Stationärpunkt landet.

## Methode

**Was verglichen wird, ohne Referenz.** Zwei Größen, die keinen Vergleichspunkt
brauchen:

```
Modelle untereinander     paarweiser Kabsch-RMSD der drei Modellvorhersagen
Barrierenstreuung         Spannweite der drei Barrierenhöhen, jede aus der
                          eigenen Vorhersage des jeweiligen Modells
```

**Die Strukturfrage.** Für jede Reaktion werden nur die Modellpunkte betrachtet,
die Stufe 1 **und** Stufe 2 bestehen. Liefern mindestens zwei Modelle einen
solchen Punkt, wird gefragt, ob es derselbe ist:

```
paarweiser Kabsch-RMSD über alle Atome
paarweiser RMSD über die reaktiven Atome allein
```

Die zweite Zahl ist die schärfere: zwei Strukturen können sich durch eine
Methylrotation um 0.05 Å unterscheiden und trotzdem denselben Übergang
beschreiben. Weichen dagegen die *reaktiven* Abstände ab, sind es verschiedene
Punkte.

**Zählregel.** Reaktionen mit nur einem gültigen Modellpunkt oder ohne einen
gehen nicht in die 7-von-8 ein, sondern in die Bilanz darunter.

## Zahlen

```
                             RKS stabil     RKS instabil
Modelle untereinander          0.0045 Å        0.051 Å
Barrierenstreuung              0.3 meV        13.9 meV     (Extremwert 4.4 eV)
```

Vier Größenordnungen in der Barrierenstreuung.

**Die Strukturfrage, aufgeschlüsselt:**

```
rxn      Sattelpunkte  paarweiser RMSD   reaktive Abweichung   Urteil
rxn7949        2       0.039             0.006                 gleicher Sattel
rxn8832        2       0.051             0.003                 gleicher Sattel
rxn1320        3       0.009 - 0.016     0.002 - 0.007         gleicher Sattel
rxn6196        3       0.006 - 0.012     0.001 - 0.033         gleicher Sattel
rxn4522        3       0.012 - 0.016     0.003 - 0.021         gleicher Sattel
rxn1147        3       0.018 - 0.032     0.001 - 0.005         gleicher Sattel
rxn7957        3       0.011 - 0.019     0.003 - 0.021         gleicher Sattel
rxn1283        2       0.064             0.208                 *** VERSCHIEDEN ***

  7  gleicher Sattel          6  kein Modell ist ein Sattel
  5  nur ein Modell           1  verschiedene Sattelpunkte
```

**Und der Zusammenhang, der daraus folgt.** Ohne einen einzigen gültigen
Modell-Sattelpunkt sind rxn8837, rxn4113, rxn8885, rxn0894, rxn7060 und
rxn0346 — **fünf davon sind zugleich die fünf größten Streuungen im Satz**
(Ränge 1, 2, 3, 4 und 6).

Die Streuung misst also gemeinsames Danebenliegen, nicht konkurrierende
Übergangszustände.

## Kontrollen

- **Beide Größen sind referenzfrei.** Weder die Modellstreuung noch die
  Barrierenstreuung braucht den RKS-TS. Das ist der Grund, sie rechts überhaupt
  berichten zu dürfen, wo eine Referenz fehlt.
- **Zwei Abstandsmaße statt einem.** Der reaktive RMSD trennt rxn1283 (0.208)
  klar von den übrigen (0.001 bis 0.033), obwohl sein Gesamt-RMSD mit 0.064 nur
  wenig über rxn8832 (0.051) liegt. Ein einzelnes Maß hätte den Fall nicht
  gefunden.
- **Die Zählregel ist explizit**, und die nicht gezählten Fälle stehen mit ihrer
  Zahl daneben (5 + 6), statt still wegzufallen.

## Vorbehalte

- **Einigkeit ist kein Beleg für Richtigkeit.** Die drei Modelle teilen
  Trainingsdaten, Zielniveau und Architekturfamilie; ihre Übereinstimmung kann
  geerbt sein. Das gilt auch für die 0.0045 Å auf der stabilen Seite. Hierhin
  gehört das Argument, es könne einen tieferen Sattelpunkt geben, den alle drei
  verfehlen — zur Einigkeit, nicht zur Uneinigkeit.
- **n = 8** für die Strukturfrage. Sieben zu eins ist ein deutliches Verhältnis,
  aber es ruht auf acht Reaktionen.
- **Die Kausalrichtung ist nicht bewiesen.** Dass die fünf größten Streuungen
  Reaktionen ohne gültigen Modellpunkt sind, ist eine Korrelation. Sie ist
  konsistent mit der Deutung in §5, aber sie ersetzt sie nicht.
- **Der Extremwert 4.4 eV** stammt aus einer einzelnen Reaktion und ist kein
  typischer Wert; berichtet wird deshalb der Median.

**Erzeugt von:** `pipeline/mode_compare.py` (Strukturvergleich),
`pipeline/model_spread.py` und `pipeline/barrier_spread.py` (Streuungen),
`pipeline/model_saddle_stats.py`.

---

# §4 · Prüft man die Strukturen, sind es keine Sattelpunkte

## Aussage

Unterwirft man die Modellvorhersagen der dreistufigen Prüfung aus §2, bestehen
sie im einreferenziellen Kontrollsatz fast durchweg — und im
multireferenziellen weniger als die Hälfte.

## Methode

**Die Kette je Struktur.** Für jede Modellvorhersage läuft dieselbe dreiteilige
ORCA-Kette aus §0: Einzelpunkt mit Stabilitätsanalyse, Gradient mit `MORead`,
Hesse. Als Array-Job über eine Aufgabenliste, ein Verzeichnis je
Struktur-Modell-Paar.

```
job_orca_freq_sweep.sh   →   orca_freq/<rxn>_<Modell>/
                              bs_sp.out   Stabilität und Orbitale
                              engrad.out  Gradient   → Stufe 1
                              numfreq.out Hesse      → Stufe 2 und 3
```

**Aufgabenlisten aus vorhandenen *Strukturen*, nicht aus vorhandenen
*Hesse-Matrizen*.** Andernfalls kann eine Lücke nie geschlossen werden: eine
Reaktion ohne Hesse taucht in der nächsten Liste wieder nicht auf. Diese Falle
ist einmal zugeschnappt und steht in Anhang B unter den Fallstricken.

**Die Bilanz** wird aus den Verzeichnissen gezogen, nicht aus einer
Zwischendatei, damit sie nicht gegenüber den Rohdaten driftet.

## Zahlen

```
                       geprüft   stationär   +1 imag   = Sattelpunkt   Anteil
einreferenziell            26          26        25              25     96 %
multireferenziell          57          30        50              26     46 %
```

Die 57 sind 19 Reaktionen × 3 Modelle. Bemerkenswert ist die mittlere Spalte:
**50 von 57 haben genau eine imaginäre Mode**, aber nur 30 sind überhaupt
stationär. Die Strukturen *sehen aus* wie Übergangszustände — sie sind nur
keine.

## Kontrollen

- **Gegengerechnet aus einer zweiten Richtung.** §1 kommt allein über Gradienten,
  ohne eine einzige Hesse-Matrix und an einer anderen Stichprobe, auf 95 % gegen
  43 %. Zwei unabhängige Wege, dieselbe Trennung.
- **Der Kontrollsatz ist nicht geschönt.** Er enthält die FOD-Verdachtsfälle
  (rxn7945, rxn7937, rxn1150, rxn0896 mit N_FOD zwischen 0.84 und 0.90) und ist
  dadurch strenger als eine zufällige Auswahl.

## Vorbehalte

- **Beide Zahlen sind Stufe 1 UND 2, nicht Stufe 3.** 96 % und 46 % zählen
  Strukturen, die stationär sind *und* genau eine imaginäre Mode haben. Die
  Frage, ob die Mode zur Reaktion gehört, ist nur für den multireferenziellen
  Satz beantwortet — `sweep_summary` führt Stufe 3 ausdrücklich als
  *multireference only*. Für die 26 links ist damit offen, wie viele auch die
  dritte Stufe bestünden; die Zahl kann dort nur sinken.
- **Keine eigene Kontrolle für diesen Abschnitt.** Was hier gemessen wird, ist
  die Anwendung von §2 auf einen Datensatz; die Kontrollen sitzen in §2 (Regel)
  und §1 (unabhängiger zweiter Weg). Einen eigenen Gegentest gibt es nicht.
- **n = 26 gegen n = 57.** Die Gruppen sind ungleich groß, weil links pro
  Reaktion nicht alle drei Modelle gerechnet wurden.

**Erzeugt von:** `pipeline/job_orca_freq_sweep.sh` (Array-Job),
`pipeline/make_freq_list*.py` (Aufgabenlisten),
`pipeline/model_saddle_stats.py` → `model_saddle_stats.txt`,
`pipeline/status_matrix.py` → `status_matrix.md`.

---

# §5 · Die Ursache: der Kraftfehler an der eigenen Vorhersage

## Aussage

Ein NEB hält an, wenn **seine** Kraft klein ist. Die Modelle melden in beiden
Gruppen dieselbe Restkraft — 0.032 eV/Å. Tatsächlich wirken dort 0.067
beziehungsweise 0.163. Das Modell merkt nicht, dass es in Schwierigkeiten
steckt, und hält deshalb an einem Punkt an, an dem die echte Kraft noch wirkt.
Das ist der Mechanismus hinter §4.

## Methode

Beide Zahlen lagen längst vor und waren nie nebeneinandergelegt worden. Weil die
Aussage des Abschnitts vollständig an diesem Vergleich hängt, steht der
Rechenweg hier ausgeschrieben.

**Modellseite.** `<modeldir>/<rxn>/transition_state.xyz`, vom ASE-NEB an seinem
Climbing Image geschrieben:

```
Properties=species:S:1:pos:R:3:forces:R:3  charge=0 spin=1
energy=-8793.900387278647  free_energy=...  stress=...
N   2.91270324  0.59558217 -0.85689469   -0.08230502  0.05081359 -0.23120899
    └────── Position [Å] ──────┘         └────── Kraft [eV/Å] ──────┘
```

Spalten 4 bis 6 sind die Kräfte des Rechners, **nicht** die projizierte
NEB-Kraft — erkennbar daran, dass `energy`, `free_energy` und `stress` im selben
Kopf stehen, also ein gewöhnlicher ASE-Rechner-Dump vorliegt.

**DFT-Seite, zwei Läufe an derselben unveränderten Geometrie.** Die
Modellstruktur wird nach `start.xyz` kopiert und nirgends nachoptimiert.

```
bs_sp.inp
  ! UKS wB97M-V def2-TZVP def2/J RIJCOSX TightSCF
  %pal nprocs 8 end
  %maxcore 4500
  %scf
    STABPerform true
    STABRestartUHFifUnstable true
    MaxIter 300
  end
  * xyzfile 0 1 start.xyz
                                    → bs_sp.gbw, kopiert nach bs_start.gbw

engrad.inp
  ! UKS wB97M-V def2-TZVP def2/J RIJCOSX TightSCF EnGrad MORead
  %moinp "bs_start.gbw"
  %pal nprocs 8 end
  %maxcore 4500
  %scf MaxIter 300 end
  * xyzfile 0 1 start.xyz
                                    → CARTESIAN GRADIENT
```

**Warum zwei Läufe.** `STABPerform` erlaubt ORCA nur mit `RunTyp SinglePoint`;
Stabilitätsanalyse und Gradient sind in einem Lauf nicht zu haben. `MORead`
stellt sicher, dass der Gradient zu der Lösung gehört, die die Stabilitätsanalyse
gefunden hat — der Gradient liegt damit auf der gebrochenen Fläche, wo eine
gebrochene Lösung existiert, und auf der restringierten, wo nicht.

**Umrechnung.** ORCA druckt dE/dx in Eh/Bohr; eine Kraft ist das Negative davon,
Faktor 51.42208 nach eV/Å. Die Modellkräfte liegen bereits in eV/Å vor.

**Definitionen.** ΔF = F_Modell − F_DFT, komponentenweise über alle 3N
Komponenten. MAE ist der Mittelwert von |ΔF|, „max Komp." das größte |ΔF|.
**|F| ist die größte Betragskomponente des jeweiligen Kraftvektors, nicht die
Norm.** Berichtet werden Mediane über die Gruppe.

**Wie die Kontrollgruppe vervollständigt wurde.** Eine erste Fassung hatte links
nur 26 Struktur-Modell-Paare — genau ein Modell je Reaktion, verteilt als UMA-M
18×, UMA-S 6×, eSEN 2×. Der Gruppenvergleich war davon unberührt, die
modellweise Zeile stand für eSEN aber auf zwei Strukturen. Die 52 fehlenden
Paare sind mit demselben Rezept nachgerechnet worden, wörtlich kopiert aus
`orca_freq/rxn0101_UMA-M`.

## Zahlen

```
                     MAE   max Komp.  |F| Modell  |F| DFT     (eV/Å, Mediane)
  RKS stabil    (78) 0.013     0.058      0.032     0.067
  RKS instabil  (44) 0.031     0.142      0.032     0.163

je Modell, MAE stabil → instabil
  UMA-S    0.013 → 0.036    2.7×
  UMA-M    0.011 → 0.023    2.1×
  eSEN     0.017 → 0.038    2.3×
```

**Die entscheidende Spalte ist die letzte.** Das Modell meldet in beiden Gruppen
dieselbe Restkraft, 0.032 eV/Å. Was dort wirklich wirkt, unterscheidet sich um
den Faktor 2.4.

```
                  Modell glaubt   tatsächlich   Verhältnis
RKS stabil            0.032          0.067          2.1
RKS instabil          0.032          0.163          5.1
```

> Ein NEB hält an, wenn **seine** Kraft klein ist. Ist die Kraft um 0.14 eV/Å
> falsch, hält er an einem Punkt an, an dem die echte Kraft noch wirkt — und
> meldet dabei Konvergenz.

**Und die Verbindung zu §4:** die DFT-Spalte ist genau die Größe, die Stufe 1
prüft. Die beiden Zeilen liegen auf verschiedenen Seiten der Schwelle von
0.15 eV/Å.

**Die schlimmsten Einzelfälle:**

```
rxn0894  UMA-M  MAE 0.372   max 1.361   glaubt 0.314   hat 1.320
rxn7060  eSEN   MAE 0.204   max 1.140   glaubt 0.028   hat 1.126
rxn0894  UMA-S  MAE 0.193   max 0.771   glaubt 0.112   hat 0.776
rxn8837  UMA-S  MAE 0.169   max 0.745   glaubt 0.046   hat 0.749
rxn7060  UMA-S  MAE 0.159   max 0.890   glaubt 0.019   hat 0.884
rxn0894  eSEN   MAE 0.116   max 0.799   glaubt 0.029   hat 0.794
rxn8837  UMA-M  MAE 0.115   max 0.438   glaubt 0.752   hat 0.757
rxn7060  UMA-M  MAE 0.076   max 0.337   glaubt 0.002   hat 0.335
```

**Zwei Fehlerarten in derselben Spalte**, und die Unterscheidung fällt gratis
heraus:

```
rxn7060  eSEN   glaubt 0.028, hat 1.126     Faktor 40 — Genauigkeitsproblem
rxn8837  UMA-M  glaubt 0.752, hat 0.757     Kraft richtig, trotzdem gestoppt
                                            — Konvergenzabbruch
```

Der zweite Fall ist **kein Modellfehler**. Er gehört ins Kapitel, weil er zeigt,
dass „das Modell liegt daneben" mindestens zwei verschiedene Dinge bedeuten
kann.

**Vorweg: es liegt nicht an den Trainingslabels.** Das ist die Erklärung, die
ein Leser hier von selbst einsetzt, und sie ist geprüft und **falsch**.

Die Vermutung war: die Modelle sind auf Transition1x-Labels trainiert, die
restringiert gerechnet wurden, hätten also die falsche Fläche gelernt. Der
Einwand dagegen ist richtig — in OMol25 wurde mit UKS neu gerechnet, und eine
UKS-Rechnung kollabiert dort zur RKS-Lösung, wo diese stabil ist. Es gibt für
das Modell nur *eine* Fläche zu lernen.

*Der Test.* An jeder Modellgeometrie werden drei Barrieren verglichen: die des
Modells, eine RKS-Barriere und eine BS-Barriere, alle vom selben Edukt aus.
Welche der beiden DFT-Varianten die Modellbarriere trifft, sagt, welcher Fläche
das Modell folgt.

```
folgt RKS                                          1
folgt BS                                          39
keine Unterscheidung möglich (Blätter < 50 meV)   16

eingeschränkt auf die 33 Fälle, in denen sich die beiden
Hypothesen um mehr als 300 meV unterscheiden:   RKS 1, BS 32

rxn        Modell   Modell     RKS       BS   |m−RKS|  |m−BS|
rxn4518    UMA-S      3.91     6.78     3.93     2.87    0.02
rxn8885    eSEN       3.26     6.31     3.29     3.05    0.03
rxn1283    UMA-M      4.82     6.73     4.86     1.91    0.04
rxn4522    UMA-M      3.85     6.04     3.85     2.19    0.00
```

**Die Modelle sind auf der richtigen Fläche**, und zwar in der Energie sehr
genau: |Modell − BS| liegt typisch bei 0.01 bis 0.04 eV, während |Modell − RKS|
bis 3.05 eV erreicht. Sie können auch gar nicht auf dem falschen Blatt
landen — ein gelerntes Potential ist per Konstruktion eine eindeutige Funktion
der Geometrie, es gibt dort kein SCF, das sich an jedem Punkt neu entscheidet.

*Kontrolle, ohne die der Test nichts wert wäre:* die Modell-Edukte liegen im
Median 0.0005 Å (max 0.0207) von den Referenz-Edukten entfernt. Der Nullpunkt
der Barriere ist praktisch dieselbe Struktur und bevorzugt keine der beiden
Hypothesen.

*Der eine Gegenfall* ist rxn0894/eSEN und ist einer: |m−RKS| 1.30 gegen |m−BS|
2.69, bei einer Modellbarriere von 7.12 eV — eine Geometrie, an der eSEN ohnehin
um 0.8 Å danebenliegt.

**Damit steht der Kraftfehler als eigenständiger Befund:** das Modell hat die
richtige Fläche gelernt und trifft ihre Energie, aber nicht ihre Ableitung.

**Erzeugt von:** `pipeline/which_sheet_did_models_learn.py` → `which_sheet.txt`.

**Wo der Fehler sitzt.** Beide Größen lassen sich an derselben Stelle messen —
`stab_pipeline` führt einen Eintrag je Modellgeometrie, also Brechungstiefe und
Kraftfehler am selben Punkt. Der Zusammenhang ist da, aber nicht monoton:

```
Spearman gegen |F|_DFT an der Modellgeometrie   (n = 122)
  |ΔE_BS| dort            0.465
  ⟨S²⟩ dort               0.465
  −λ_min_ext dort         0.615

Brechungstiefe      n    |F|_DFT   |F|_Modell   Fehler
stabil, ΔE = 0     82     0.069      0.031      0.035
1 – 50 meV         11     0.163      0.023      0.124   ← Maximum
50 – 200 meV        6     0.163      0.114      0.089
über 200 meV       23     0.141      0.034      0.069
```

Der Fehler ist **nicht** dort am größten, wo am tiefsten gebrochen wird,
sondern wo die Brechung flach ist — wo die beiden Lösungen also nahezu entartet
sind. Bei über 200 meV ist er fast halbiert.

Das ist verträglich mit einer einfachen Deutung: an der Kreuzung der beiden
Blätter hat die Grundzustandsfläche einen Knick, und eine glatte Funktion kann
einen Knick zwar in der Energie mitteln — daher die 0.01 bis 0.04 eV
oben — nicht aber in der Ableitung. Wo eines der Blätter deutlich tiefer liegt,
ist die Fläche wieder glatt und das Modell wieder gut.

> **Belegt ist die Korrelation, nicht der Mechanismus.** Die mittleren Gruppen
> sind klein (n = 11 und 6). |F|_DFT ist außerdem die Kraft, die das Modell
> hätte wegoptimieren sollen — ein Zusammenhang mit der Instabilität könnte
> auch heißen, dass die Fläche dort schlicht steiler ist. Die reine
> Fehlerspalte korreliert schwächer, 0.375 gegen λ_min_ext statt 0.615. Die
> Alternative „dort ist alles schwierig" ist damit nicht ausgeschlossen.

## Kontrollen

**Es ist derselbe Punkt.** `orca_freq/<rxn>_<Modell>/start.xyz` ist eine Kopie
von `transition_state.xyz`. Über 44 Paare geprüft, größte Abweichung **exakt
0.00 Å**. Das ist keine Erkenntnis, sondern eine Installationskontrolle — sie
hätte gefangen, wenn die Aufgabenliste die Referenzstruktur statt der
Modellstruktur genommen oder die Atomreihenfolge getauscht hätte.

**Die gespeicherten Kräfte gehören zu den gespeicherten Koordinaten.** Geometrie
und Kräfte stammen aus demselben ASE-Snapshot, aber das stand nur zu vermuten.
Gegentest: dasselbe Modell noch einmal als reiner Einzelpunkt auf dieselbe
Geometrie.

```
UMA-S   median max|ΔF| 3.28e-06   größte 1.07e-05 eV/Å
UMA-M                  1.53e-06            1.94e-05
eSEN                   2.10e-06            1.32e-05
```

57 von 57 ohne Abweichung, vier Größenordnungen unter dem berichteten Fehler.

**Die Kontrollgruppe ist wirklich einreferenziell.** Jede der 52 Nachrechnungen
protokolliert ihr ⟨S²⟩ aus Stufe 1a: Median 0.0000, Minimum 0.0000, Maximum
0.3118; zwei Werte über 0.05 (0.070 und 0.312), die übrigen exakt null. An den
Modellgeometrien der stabilen Reaktionen findet die Stabilitätsanalyse also
nichts zu brechen — die Einstufung stammte bis dahin allein aus der Analyse am
RKS-TS.

**52 von 52 ohne Abbruch**, mit Sperren gegen jede der drei Arten, auf die ein
solcher Lauf still scheitern kann: ORCA nicht ausführbar, Lauf nicht normal
beendet, keine Gradientenausgabe. Vorhandene Gradienten wurden nicht
überschrieben.

**Kreuzprobe an der modellweisen Zeile.** UMA-M, das als einziges schon vorher
eine ordentliche Kontrollseite hatte (18 Strukturen), ändert sich durch die
Nachrechnung **gar nicht** — 0.011 vorher wie nachher. eSEN, das auf zwei
Strukturen stand, verschiebt sich von 0.010 auf 0.017.

## Vorbehalte

- **Basissatz.** Die Modelle sind gegen ωB97M-V/**def2-TZVPD** trainiert
  (OMol25), gerechnet wird def2-TZVP. Ein Teil der Differenz ist Basissatz und
  nicht Modellfehler. Für die Kernaussage trägt das, weil sie von der
  Größenordnung lebt und nicht von der dritten Stelle.
- **n = 44 von 57** auf der instabilen Seite. Dreizehn Modellgeometrien haben
  keinen DFT-Gradienten: rxn7949, rxn5691 und rxn4522 je alle drei Modelle,
  rxn1147 (UMA-M, eSEN), rxn7957 (UMA-S, eSEN). rxn7949 fehlt damit ausgerechnet
  in dieser Tabelle, obwohl es andernorts im Kapitel diskutiert wird.
- **|F| ist die größte Komponente**, nicht der Betrag. 0.032 und 0.163 sind
  Maximalkomponenten.
- **Vier der 19 Modell-NEBs sind gar nicht konvergiert.** Für diese gilt der
  Satz „ein NEB hält an, wenn seine Kraft klein ist" nicht — sie sind in die
  Wandzeit gelaufen.
- **Der Effekt ist kleiner, als eine frühere Fassung auswies.** Mit
  unvollständiger Kontrollseite standen dort Faktoren von 2.1 bis 3.7; mit
  vollständiger sind es 2.1 bis 2.7 (Anhang A.8). Dafür ist die Aussage jetzt
  einheitlich: alle drei Modelle verschlechtern sich um gut das Doppelte, keines
  fällt aus dem Rahmen.

**Erzeugt von:** `pipeline/force_error_at_ts.py` → `force_error_at_ts.txt`;
`pipeline/job_orca_grad_gap.sh` über `grad_gap_tasks.txt` für die 52
Nachrechnungen; `pipeline/model_sp_recheck.py` für den Gegentest. Die Fassung
vor der Vervollständigung liegt als `force_error_at_ts_vor_luecke.txt` daneben.

---

# §6 · Das Scharnier: der RKS-TS ist gut, nur auf der falschen Fläche

## Aussage

Der RKS-TS ist keine schlechte Rechnung. Er ist ein sauberer Sattelpunkt der
restringierten Fläche — 18 von 19 unterschreiten dort 0.15 eV/Å. Auf der Fläche,
auf der die Reaktion abläuft, ist **keiner von 19** ein Stationärpunkt. Eine
gute Antwort auf die falsche Frage. Damit verschiebt sich die Diagnose weg von
den Modellen und hin zu der Fläche, auf der alle Beteiligten rechnen.

## Methode

**Dieselbe Geometrie, zwei Flächen.** Für jede Reaktion wird am RKS-TS zweimal
ein Gradient bestimmt:

```
rks_grad.max_evang        restringierte Rechnung
bs.bs_grad.max_evang      gebrochen-symmetrische Rechnung
```

Beide stehen in `stab_pipeline/<rxn>/result.json` im selben Eintrag
(`source == 'RKS-ref'`). Die zweite Spalte existiert nur, wo die restringierte
Lösung extern instabil ist — sonst ist `bs` gleich `None`, und beide Flächen
fallen zusammen.

> **Warum die beiden Zahlen leicht zu verwechseln sind.** Sie heißen
> `rks_grad.max_evang` und `bs.bs_grad.max_evang` und stehen im selben Eintrag.
> Wer die erste liest, wo die zweite gemeint ist, bekommt einen RKS-TS, der
> überall gültig aussieht — die genaue Umkehrung des Befunds. Beim Bauen dieser
> Tabelle ist das einmal passiert (Anhang A.2).

**Die Endpunktprämisse.** Damit eine Barriere überhaupt einen Sinn hat, muss der
Nullpunkt stimmen. Geprüft wird deshalb dieselbe Stabilitätsanalyse an Edukt und
Produkt jeder Reaktion.

## Zahlen

```
rxn         auf RKS    auf BS   Faktor   Lesart
---------------------------------------------------------------------------
rxn7949       0.105     1.686      16x   stationär auf RKS, nicht auf BS
rxn8832       0.142     2.733      19x   stationär auf RKS, nicht auf BS
rxn1320       0.059     2.073      35x   stationär auf RKS, nicht auf BS
rxn4113       0.079     0.386       5x   stationär auf RKS, nicht auf BS
rxn8885       0.042     2.637      62x   stationär auf RKS, nicht auf BS
rxn6196       0.179     0.638       4x   auf beiden nicht stationär
rxn0346       0.052     2.613      50x   stationär auf RKS, nicht auf BS
rxn4518       0.068     2.949      43x   stationär auf RKS, nicht auf BS
rxn3107       0.063     1.646      26x   stationär auf RKS, nicht auf BS
rxn8837       0.057     1.697      30x   stationär auf RKS, nicht auf BS
rxn7060       0.033     1.766      53x   stationär auf RKS, nicht auf BS
rxn5691       0.041     1.419      35x   stationär auf RKS, nicht auf BS
rxn1283       0.038     2.386      63x   stationär auf RKS, nicht auf BS
rxn8827       0.026     1.128      43x   stationär auf RKS, nicht auf BS
rxn4522       0.098     1.875      19x   stationär auf RKS, nicht auf BS
rxn1147       0.065     1.840      28x   stationär auf RKS, nicht auf BS
rxn0894       0.062     1.350      22x   stationär auf RKS, nicht auf BS
rxn7957       0.026     0.901      34x   stationär auf RKS, nicht auf BS
rxn5690       0.037     0.162       4x   stationär auf RKS, nicht auf BS

Median auf der RKS-Fläche   0.059 eV/Å    18 von 19 stationär
Median auf der BS-Fläche    1.697 eV/Å     0 von 19 stationär
```

Der Faktor zwischen den beiden Spalten läuft von 4 bis 63.

**Die Eingrenzung gehört unmittelbar daneben**, damit die Aussage nicht größer
klingt, als sie belegt ist:

> Alle 45 Edukte sind geschlossenschalig, 40 von 45 Produkten auch. Jede
> Vorwärtsbarriere steht damit auf einem korrekten Nullpunkt; betroffen sind
> Reaktionsenergien und Rückbarrieren von fünf Reaktionen, um 2 bis 84 meV.
> **Das Problem sitzt ausschließlich am Barrierenkamm.**

## Was daraus folgt

**Relabeling reicht nicht.** Ein UKS-Einzelpunkt auf einem RKS-Pfad ist eine
Energie an einem Punkt, an dem noch 1.70 eV/Å Kraft wirken. Die Barriere daraus
ist keine Barriere. Die Geometrie muss auf der unrestringierten Fläche
**optimiert** werden, nicht nur dort ausgewertet.

**Die Triage.** Aus dem Kraftfehler (§5) folgt etwas unmittelbar Brauchbares:
ob eine TS-Optimierung von einer Modellgeometrie aus gelingt, lässt sich am
DFT-Gradienten an dieser Geometrie ablesen.

```
Gradient am Modell   Lauf                    Ergebnis                 Stufe
────────────────────────────────────────────────────────────────────────────
0.05 – 0.25 eV/Å     tsopt_rxn0346_UMA-M     ν -1295, Rate 1.031        c
                     tsopt_rxn1147_UMA-M     ν  -230, Rate 0.073        c
                     tsopt_rxn3107_UMA-M     ν -1484, Rate 0.598        c
                     tsopt_rxn7957_UMA-M     ν  -624, Rate 0.987        c
                     tsopt_rxn8827_UMA-M     ν  -592, Rate 1.390        c
                     tsopt_rxn8832_UMA-M     ν  -652, Rate 1.217        c
                     tsopt_rxn7949_UMA-M     ν   -69, Rate 0.008        b  ←
                     → 6 von 7 bestehen alle drei Stufen

0.33 – 1.32 eV/Å     tsopt_rxn0894_UMA-S     Minimum, keine imag. Mode  a
                     tsopt_rxn8837_UMA-M     ν   -59, Rate 0.054        b
                     tsopt_rxn7060_UMA-M     nicht konvergiert, 1.71    –
                     → 0 von 3
```

Die Schwelle 0.25 eV/Å liegt in der Lücke zwischen der obersten Zeile der ersten
Gruppe (0.25) und der untersten der zweiten (0.33). Sie ist **abgelesen, nicht
angepasst**, und die Vorhersage „unten gelingt es, oben nicht" wurde vor dem
Start der zweiten Gruppe notiert.

## Kontrollen

- **Die Vorhersage der Triage war vorab festgelegt** und ist mit 9 von 9
  eingetroffen; die Negativkontrolle rxn0894 (1.32 eV/Å) ist wie angekündigt
  gescheitert. Eine im Nachhinein gelegte Grenze hätte diesen Charakter nicht.
- **Der Nebenbefund als zusätzliche Probe:** keiner der zehn Läufe hat einen
  Sattelpunkt gefunden, der nicht schon bekannt war. Gegen den stehenden
  Vorbehalt — eine lokale Suche findet nur, was unter ihrem Startpunkt liegt —
  ist das der stärkste verfügbare Hinweis.
- **Die Endpunktprämisse ist geprüft**, nicht angenommen: alle 45 Edukte
  geschlossenschalig, 40 von 45 Produkten.

## Vorbehalte

- **rxn6196 ist der Ausreißer, und er gehört dazu.** Mit 0.179 eV/Å ist es die
  einzige Reaktion, bei der der RKS-TS auch auf seiner *eigenen* Fläche nicht
  ganz konvergiert ist. Die Tabelle sagt es, statt es zu glätten.
- **Die Triage-Schwelle beruht auf zehn nicht zufällig gewählten Punkten**, neun
  davon von UMA-M. Sie ist abgelesen, nicht bestimmt.
- **rxn7949 zeigt, dass ein niedriger Gradient nicht genügt.** Der Lauf startet
  bei 0.051 eV/Å, liegt also klar im unteren Bereich, und landet auf einem
  Torsionssattel im **Eduktbecken**: −69 cm⁻¹, Bindungsrate 0.008, beide
  reaktiven Bindungen auf Eduktwerten. Ein sauber konvergierter Stationärpunkt
  mit genau einer imaginären Mode, 0.53 eV unter dem BS-TS-Opt-Übergangszustand,
  und trotzdem der falsche Punkt. Die Triage sagt voraus, ob eine Optimierung
  *konvergiert* — nicht, ob sie am richtigen Ort ankommt.
- **Zwei Reaktionen liefern tiefere Stationärpunkte, die Minima sind:** rxn0894
  und rxn8885, 180 beziehungsweise 425 meV unter dem BS-TS-Opt-Sattelpunkt, beide
  ohne imaginäre Mode. Die Falle aus der Dreistufenregel ist kein Einzelfall,
  sondern wiederkehrend in der diradikalischen Region.

**Erzeugt von:** `pipeline/saddle_matrix.py` → `saddle_matrix.txt`;
`pipeline/gradient_comparison.py` für die beiden Gradientenspalten;
`pipeline/endpoint_report.py` → `endpoint_report.txt`;
`pipeline/job_bs_tsopt_umam_missing.sh` für die Triage-Läufe, bewertet mit
`pipeline/verdict_final.py`.

---

# §7 · Der Plottwist: auch die Rechenverfahren scheitern

## Aussage

Wenn nur die Modelle scheiterten, ließe sich das als Unreife abtun. Aber ORCA
scheitert auf denselben Reaktionen — ein ausgewachsenes Quantenchemieprogramm mit
gebrochener Symmetrie, gebaut genau für diesen Fall. Vier voneinander
unabhängige Werkzeuge, dasselbe Muster, derselbe Reaktionssatz. **Das ist eine
Aussage über die Fläche, nicht über eine Werkzeugklasse** — und es ist der
Grund, warum die Instabilitätsanalyse als Prädiktor überhaupt etwas taugt: sie
sagt nicht „hier wird das Modell schlecht sein", sondern „hier wird alles
schwierig".

## Methode

**Der eigene BS-NEB.** 19 Reaktionen, ωB97M-V/def2-TZVP, `BrokenSym 1,1`,
Endpunkte aus `orca_neb_results/<rxn>/{reactant,product}.xyz`. Das Ergebnis
durchläuft dieselbe dreistufige Prüfung wie jede Modellvorhersage — Gradient auf
der BS-Fläche, Hesse, Modenanalyse.

**Der Bandzustand, Bild für Bild.** Für jedes Bild eines konvergierten Bandes
wird ⟨S²⟩ nachgemessen: ein Einzelpunkt mit `TightSCF MORead` auf den
gespeicherten Orbitalen des jeweiligen Bildes, der von einer konvergierten
Wellenfunktion aus in einem Zyklus fertig ist und auf derselben Lösung bleibt.
Damit lässt sich unterscheiden:

```
verfügbar    existiert an dieser Geometrie überhaupt eine gebrochene Lösung?
             → Einzelpunkt mit STABPerform
genommen     hatte das Band sie?
             → MORead auf den gespeicherten Bildorbitalen
```

**Die TS-Optimierung vom RKS-TS aus** ist das dritte Verfahren: `OptTS` auf der
BS-Fläche, Start an der restringierten Struktur, Ergebnis wieder durch die drei
Stufen.

## Zahlen

**Die eigenen BS-NEB-Läufe, dreistufig geprüft:**

```
nebts_rxn0346   Gradient 2.553 eV/Å   2 imaginäre Moden   kein Stationärpunkt
nebts_rxn8827            1.074        2                   kein Stationärpunkt
nebts_rxn6196            0.683        2                   kein Stationärpunkt
```

**Und sie sind dabei nicht an der Konvergenz gescheitert.** Das ist der
wichtigste Einzelbefund dieses Abschnitts, und er widerspricht der
naheliegenden Vermutung:

```
NEB konvergiert (nach ORCAs eigenen Kriterien)     15 von 19
Wandzeit erreicht                                   4 von 19
                                 rxn7949, rxn8885, rxn4518, rxn7060

mit NEB-TS_converged.xyz                           13
nur mit CI-Datei                                    3   rxn3107, rxn5691, rxn1283
Laufzeiten der konvergierten                        5 h bis 46 h
```

Fünfzehn Läufe melden Konvergenz und liefern eine Struktur ab, die auf der
Grundzustandsfläche 0.68 bis 2.55 eV/Å trägt. Ein Lauf, der bei 0.103 eV/Å
konvergiert meldet und bei Nachmessung 2.55 zeigt, hat nicht ungenau gerechnet —
**er hat auf einer anderen Lösung gerechnet als der, gegen die geprüft wird.**
Innerhalb der NEB-Maschinerie entscheidet `BrokenSym` bei jedem SCF neu, welches
Blatt genommen wird; das Verfahren konvergiert dann sauber gegen etwas, das
keine durchgehende Fläche ist.

Dasselbe Bild von der Modellseite: der Kraftfehler der Modelle hat sein
Maximum dort, wo die beiden Lösungen nahezu entartet sind (§5). Beide
Werkzeugklassen stolpern an derselben Stelle, in verschiedener Gestalt — das
SCF entscheidet sich neu, das Modell mittelt darüber hinweg.

**Die Bilanz über alle 19:**

```
Band bricht die Symmetrie irgendwo      14 von 19
höchstes Bild gebrochen                 11 von 19
UKS-NEB liefert gültigen Sattelpunkt     8 von 19
```

**Wo das höchste Bild landet, entscheidet mit:**

```
                      n   Grad auf der BS-Fläche   stationär   RMSD zum RKS-TS
Gipfel gebrochen     11         0.011 eV/Å          7 von 7        0.529 Å
Gipfel restringiert   8         1.074               1 von 5        0.055
```

Faktor 100 im Gradienten, Faktor 10 im Abstand. Ein Band, dessen oberstes Bild
restringiert bleibt, liefert eine Struktur, die praktisch **auf dem RKS-TS**
sitzt — dieselbe Beziehung wie in §6: stationär auf der einen Fläche, nicht auf
der anderen.

**Die TS-Optimierung vom RKS-TS aus:**

```
gültige Sattelpunkte                     13 von 19
Reaktionen mit konkurrierendem Sattel    10 von 19
```

Sie ist damit die beste Methode im Feld — und liefert trotzdem keine
verlässliche Antwort, weil ihr Ergebnis eine Funktion des Startpunkts ist. Bei
rxn1320 und rxn4518 konvergiert sie sauber, mit genau einer imaginären Mode und
tiefer als der RKS-TS, und ist trotzdem nicht verwertbar: „besseren Sattel
gefunden" lässt sich nicht von „falschen Sattel gefunden" unterscheiden. Beide
fallen durch Stufe 3 (Modenanteil 0.00 beziehungsweise 0.03).

**Laufzeiten**, für die Planung eines Nachfolgeexperiments:

```
rxn1320   7 h 26 m        rxn8827   9 h 49 m
rxn8837   9 h 54 m        rxn7949   kein TOTAL RUN TIME → Walltime-Abbruch
```

## Kontrollen

- **Die Bandmessung hat eine eingebaute Positivkontrolle.** Das Skript bricht ab
  (Exit 3 beziehungsweise 4), wenn sein Rezept ein bereits bekanntes ⟨S²⟩ nicht
  reproduziert. Ohne diese Sperre wäre eine stille Fehlmessung nicht
  aufgefallen — sie war es vorher einmal (Anhang A.1).
- **Die Zahlen sind auf die 19 beschränkt.** `bs_uks_neb_results` enthält 22
  Läufe, drei davon aus dem einreferenziellen Satz (rxn1150, rxn7936, rxn7945).
  Eine frühere Fassung hat über das Verzeichnis iteriert statt über die Liste
  (Anhang A.4).
- **rxn7949 ist als Walltime-Abbruch identifiziert**, nicht als inhaltliches
  Scheitern — das Log hat kein `TOTAL RUN TIME`. Es stand zwischenzeitlich
  falsch als Methodenversagen im Text (Anhang A.5).
- **Dieselbe dreistufige Regel** wie für die Modelle, ohne Ausnahme. Der Twist
  wäre wertlos, wenn die eigenen Rechnungen milder bewertet würden.

## Vorbehalte

- **Der Zusammenhang zwischen Gipfelbild und Ergebnis ist eine Korrelation.**
  Warum bei sechs Bändern das höchste Bild restringiert bleibt, ist nicht
  erklärt. Eine Erklärung über einen Energieversatz an der Naht zwischen den
  Blättern wurde geprüft und **widerlegt**.
- **Der eine Stationärpunkt in der rechten Spalte ist rxn5690** (Gradient
  0.004) — der Grenzfall des Satzes: N_FOD 0.433, ΔE_BS −1.3 meV, ⟨S²⟩ 0.068 am
  RKS-TS. Faktisch einreferenziell, nur nicht so etikettiert. Ohne ihn stünde
  dort 0 von 4.
- **Vier Werkzeuge, aber nicht vier unabhängige Codes.** UMA-S, UMA-M und eSEN
  teilen Trainingsdaten und Architekturfamilie; wirklich unabhängig ist nur der
  Gegensatz Modelle ↔ ORCA. Das Argument trägt trotzdem, weil ORCA weder
  Trainingsdaten noch Architektur mit den Modellen teilt.
- **Es ist keine erschöpfende Suche.** Dass vier Verfahren scheitern, zeigt
  nicht, dass jedes Verfahren scheitern muss. §8 ist das Gegenbeispiel.

**Erzeugt von:** `pipeline/job_bs_uks_neb18.sh` → `bs_uks_neb_results/`,
ausgewertet in `bs_neb_check.txt`; `pipeline/job_orca_band_s2.sh` und
`job_orca_band_s2_cheap.sh` für die Bandmessung; `pipeline/tsopt_null.py` für
die Startpunktabhängigkeit.

---

# §8 · Sattelpunkte sind doch zu haben — aber es ist kein Verfahren

## Aussage

Am Ende von §7 steht eine Fläche, auf der **kein einziger** der bekannten
Punkte stationär ist. Damit ließe sich nichts weiterrechnen: keine Barriere,
keine Frequenz, kein Vergleichswert für irgendein Modell.

Die Frage dieses Abschnitts ist deshalb nicht, welche Methode die beste ist,
sondern ob es überhaupt geht. **Fände ein Rechenverfahren die richtigen
Sattelpunkte verlässlich, wäre das der Maßstab, an dem ein Modell gemessen
werden könnte** — und ein Modell, das ihn erreicht, wäre ein unmittelbarer
Gewinn: Stunden statt Tagen, für jede Reaktion.

Die Antwort ist ein eingeschränktes Ja. Fünfzehn der neunzehn Reaktionen haben
am Ende einen Sattelpunkt, der alle drei Stufen besteht. Aber sie stammen aus
drei Anläufen mit verschiedenen Startpunkten, und acht Bänder liefen in die
Wandzeit. **Vor allem aber führen verschiedene Startpunkte nicht immer zu
demselben Sattelpunkt:** in vier Reaktionen liegen zwei gültige Punkte
derselben Reaktion 194 bis 892 meV auseinander, in zweien davon gehören sie
sogar zu verschiedenen elektronischen Lösungen. **Ein Verfahren, das man
jemandem in die Hand geben könnte, ist das nicht.**

## Methode

Der eine methodische Griff: ORCA verfeinert den Sattelpunkt **innerhalb** der
NEB-Maschinerie weiter, mit `BrokenSym`, das den gebrochenen Zustand bei jedem
SCF neu herleitet. Wir hören am Climbing Image auf und übergeben an eine eigene
Optimierung.

```
1  Band            NEB-CI statt NEB-TS   →  Climbing Image
2  Einzelpunkt     STABPerform           →  Orbitale, Zustand eingefroren
3  TS-Optimierung  OptTS NumFreq MORead  →  Sattelpunkt, dann Stufe 1-3

alles auf ωB97M-V/def2-TZVP def2/J RIJCOSX, Hesse numerisch wegen VV10
```

**Schritt 2 ist der entscheidende.** Ohne ihn müsste Schritt 3 wieder mit
`BrokenSym` arbeiten und in jedem Optimierungsschritt neu entscheiden, welche
der beiden Lösungen er nimmt. Mit `MORead` steht der Zustand fest, und die
Suche läuft auf *einer* Fläche.

**Schritt 1 ist dagegen austauschbar**, und genau das wurde geprüft. Drei
Startpunkte, danach dieselben Schritte 2 und 3:

```
A  neues Band          NEB-CI, eigens gerechnet
B  vorhandenes Band    das Climbing Image der Läufe aus §7,
                       deren eigenes Ergebnis dort durchgefallen ist
C  Modellgeometrie     die Vorhersage von UMA-M, ohne jedes Band
```

Skripte, Schwellenwerte und die gescheiterten Ansätze stehen in Anhang B.

## Zahlen

```
Startpunkt              geprüft  gültig   Aufwand je Reaktion
A  neues Band              11      10     7 bis 45 h; 8 in die Wandzeit
B  vorhandenes Band         9       9     1 bis 2 h
C  Modellgeometrie         18      12     ~1 h
```

**Welcher Startpunkt welchen Sattelpunkt gefunden hat.** Die Zahl hinter dem
Urteil ist der Energieabstand in meV zum tiefsten *gültigen* Punkt derselben
Reaktion; alle drei Wege enden in einer konvergierten Optimierung auf demselben
Niveau, die Energien sind also direkt vergleichbar.

```
              A neues Band        B vorhandenes Band   C Modellgeometrie
rxn0346       g  0.60    —        g  0.33   +14        g  0.60    —
rxn0894       ·                   g  1.03    —         ·
rxn1147       g  0.46  +238       ·                    g  0.00    —
rxn1283       ·                   ✗                    n  1.00
rxn1320       m  1.02             g  0.00  +637        g  0.69    —
rxn3107       ·                   g  0.07    —         g  0.17  +194
rxn4113       g  0.97    —        ·                    g  1.01   +22
rxn4518       ·                   ·                    n  1.01
rxn4522       ·                   g  0.95    —         m  1.01
rxn5690       g  0.00    —        g  0.00    —         g  0.25   +14
rxn5691       ·                   g  1.02    —         g  1.02    —
rxn6196       g  0.49    —        g  0.50    +1        g  0.49    —
rxn7060       ·                   ·                    s  0.36
rxn7949       ·                   ·                    m  0.99
rxn7957       g  0.72  +892       ·                    g  0.70    —
rxn8827       g  1.02    —        g  1.02    —         g  1.02    —
rxn8832       g  1.01    —        ·                    g  1.00   +13
rxn8837       g  1.04    —        ·                    m  1.01
rxn8885       g  0.15    —        ·                    g  0.15    —

g  gültig, alle drei Stufen        s  nicht stationär (Stufe 1)
n  falsche Zahl imaginärer Moden   m  Mode gehört nicht zu dieser Reaktion
✗  Optimierung nicht konvergiert   ·  nicht versucht oder kein Ergebnis
—  der tiefste gültige Punkt dieser Reaktion
Zahl nach g/s/n/m ist ⟨S²⟩ am Punkt
```

Drei Dinge stehen darin:

**Übereinstimmung.** Wo zwei oder drei Wege einen gültigen Punkt liefern,
liegen sie in acht Reaktionen innerhalb von 1 bis 22 meV — rxn0346, rxn4113,
rxn5690, rxn5691, rxn6196, rxn8827, rxn8832, rxn8885. Das ist derselbe
Sattelpunkt, erreicht von Startpunkten, die weder Verfahren noch Geometrie
teilen.

**Uneinigkeit.** Vier Reaktionen haben zwei gültige Sattelpunkte mit deutlichem
Abstand:

```
rxn7957   892 meV      rxn1320   637 meV
rxn1147   238 meV      rxn3107   194 meV
```

Bei rxn1147 und rxn1320 unterscheiden sich die beiden auch in ⟨S²⟩ — 0.46 gegen
0.00 und 0.00 gegen 0.69. Dort sind es nicht nur zwei Geometrien, sondern zwei
verschiedene elektronische Lösungen.

**Abdeckung.** Kein Startpunkt allein kommt über zwölf hinaus. Vier
Reaktionen bleiben auf allen Wegen offen: rxn1283, rxn4518, rxn7060,
rxn7949.

## Kontrollen

Dasselbe Rezept auf drei einreferenziellen Reaktionen, ausgewählt nach dem
Abstand zur Instabilität (`lmin_int` 0.919 / 0.327 / 0.224):

```
rxn1061   ⟨S²⟩  0.000   ν  -471.1 cm⁻¹   0.012 Å vom RKS-TS
rxn0101   ⟨S²⟩  0.000   ν   -68.0        0.142
rxn0896   ⟨S²⟩ -0.000   ν -1533.3        0.026
```

⟨S²⟩ ist exakt null, jede hat genau eine imaginäre Mode, und alle drei landen
auf dem RKS-TS. **Das Verfahren bricht keine Symmetrie, wo keine zu brechen
ist.** Ohne diese drei Zeilen könnten die fünfzehn ein Artefakt der Methode
sein.

rxn0896 ist dabei der härteste Fall: der geringste Abstand zur
Instabilitätsschwelle, und zugleich eine der sieben Reaktionen, die auf dem
Screening-Niveau instabil erscheinen und erst am Zielniveau stabil sind.

## Der Unterschied, auf den es hinausläuft

Dass wir kein verlässliches Verfahren gefunden haben, liegt an der Methodik —
Wandzeit, Optimierereinstellungen, Wahl des Startpunkts. Das ist behebbar: mehr
Rechenzeit, engere Protokolle, ein besserer Startpunkt.

Beim Modell ist es das nicht.

> Beim Rechenverfahren ist Ungenauigkeit eine Frage des Aufwands: engere
> Schwellen, längere Läufe, ein anderer Startpunkt. Beim fertigen Modell ist
> sie es nicht. Der Kraftfehler ist zur Laufzeit weder reduzierbar noch von
> innen erkennbar — das Modell meldet an einem falschen Punkt dieselbe
> Restkraft wie an einem richtigen, 0.032 gegen 0.032 eV/Å, während dort 0.163
> beziehungsweise 0.067 wirken (§5). **Das ist eine prinzipielle Grenze, keine
> Frage der Rechenzeit.**
>
> Sie gilt für ein gegebenes Modell. Ob gezieltes Nachtrainieren in der
> Nahtregion sie aufhebt, ist offen (§10).

Damit fehlt dem Modell nicht nur Genauigkeit, sondern auch ein
Abbruchkriterium: man kann einer Vorhersage nicht ansehen, ob ihr zu trauen
ist. Genau deshalb braucht die Triage in §6 einen externen Maßstab — eine
einzelne DFT-Gradientenrechnung.

## Und trotzdem

**Die Modellgeometrie ist der billigste und breiteste Startpunkt.** Sie deckt
alle neunzehn Reaktionen ab, liefert zwölf gültige Sattelpunkte in je etwa
einer Stunde — und findet in drei Fällen den *tieferen* Punkt als das Band.

Die Vorhersage selbst ist kein Sattelpunkt. Aber sie liegt im richtigen Becken,
oft im besseren. Der Kraftfehler verhindert, dass das Modell dort ankommt; er
verhindert nicht, dass es hinzeigt.

**Das Modell ersetzt die Rechnung nicht — es macht sie bezahlbar.**

## Vorbehalte

- **Warum der ungeteilte Lauf scheitert, ist nicht erklärt.** Belegt ist, dass
  die Trennung hilft; der Mechanismus dahinter nicht.
- **Vier Reaktionen bleiben auf allen drei Wegen offen** — rxn1283, rxn4518,
  rxn7060, rxn7949.
- **Die Wege sind sich nicht immer einig.** rxn1320 liefert über das neue Band
  einen ungültigen Punkt, über die beiden anderen je einen gültigen, und diese
  beiden liegen 637 meV auseinander.
- **Die Energievergleiche gelten nur innerhalb einer Reaktion.** Über
  Reaktionen hinweg sagt die Matrix nichts; die Spalte ist ein Abstand zum
  jeweils tiefsten Punkt derselben Zeile.
- **Das Screening-Niveau überträgt Geometrien nicht.** In allen vier prüfbaren
  Fällen liegt die dort gefundene Struktur über 0.18 Å von der am Zielniveau
  entfernt (Anhang B).
- **Die dreistufige Regel prüft Gültigkeit, nicht Optimalität.** Wo zwei
  gültige Punkte einer Reaktion vorliegen, sagt sie nicht, welcher der
  relevante ist.

**Erzeugt von:** `pipeline/job_orca_nebci_split.sh`,
`pipeline/job_orca_sep_step23.sh`, `pipeline/job_bs_tsopt_umam_missing.sh` und
`pipeline/job_orca_umam_eval.sh`; ausgewertet mit `pipeline/stage3_new.py`.
Einzelheiten zu allen drei Startpunkten in Anhang B.

---

# §9 · Grenzen — was dieses Kapitel nicht zeigt

## Aussage

Drei Dinge, die zwischen den Zeilen mitgelesen werden könnten und nicht belegt
sind: dass wir wüssten, *warum* Bandverfahren scheitern; dass ein tieferer
Sattelpunkt gefunden wurde; und dass es nun Barrierenhöhen auf der gebrochenen
Fläche gäbe. Keines davon trifft zu.

## Methode

Der Höhenvergleich ist **versucht** worden, und der Rechenweg gehört
mitgedruckt, weil das Scheitern selbst ein Befund ist.

**Der Aufbau.** Für jede der 16 Reaktionen mit neuem Sattelpunkt wurden alle
Kandidaten mit **einer** Methodenzeile gerechnet — der neue Punkt, BS-TS-Opt,
RKS-TS, UKS-NEB, UMA-M, TSoptM — je Geometrie zwei SCF-Lösungen, die tiefere
zählt:

```
plain   ! UKS wB97X 6-31G(d) SP TightSCF SlowConv
        %scf STABPerform true / STABRestartUHFifUnstable true end
bs      ! UKS wB97X 6-31G(d) SP TightSCF SlowConv
        %scf BrokenSym 1,1 end
```

**Danach der Gradient an denselben Geometrien**, `EnGrad` mit `MORead` aus dem
gespeicherten `sp.gbw`, damit er zu genau der Lösung gehört, deren Energie in
der Tabelle steht. `STABPerform` ist dort bewusst nicht gesetzt — ORCA erlaubt
es nur mit `RunTyp SinglePoint`.

**Die Auswertungsregel, vorab festgelegt:** eine Energiedifferenz zählt nur,
wenn **beide** Punkte auf diesem Niveau stationär sind.

## Zahlen

```
max|F| in eV/Å auf ωB97X/6-31G(d), an der Geometrie der jeweiligen Struktur

rxn         neu  | BS-TS-Opt  RKS-TS  UKS-NEB   UMA-M  TSoptM
rxn0346   0.002  |      1.42    4.62     4.57    1.58    1.44
rxn0894   0.002  |      0.68    2.56     0.58    1.88       —
rxn1147   0.004  |      0.56    3.61     0.54    0.98    0.95
rxn1283   0.002  |      0.43    5.25        —       —       —
rxn1320   0.002  |      0.38    3.04     3.06    0.37       —
rxn3107   0.004  |      3.53    5.00     3.41    3.41    3.53
rxn4113   0.003  |      0.58    1.84     0.58    0.68       —
rxn4518   0.004  |      0.67    3.33        —    0.68       —
rxn4522   0.002  |      2.24    2.90     1.42    0.61       —
rxn5690   0.011  |         —    1.91     0.91    1.99       —
rxn6196   0.006  |      1.86    3.34     3.58    1.74       —
rxn7949   0.003  |      1.38    2.38        —    1.23    1.22
rxn7957   0.002  |      1.41    3.13     1.39    1.35    1.26
rxn8832   0.002  |      1.13    3.61     1.13    1.24    1.13
rxn8837   0.005  |      0.91    2.07     0.91    1.45    1.21
rxn8885   0.015  |      4.83    5.97        —    4.88       —

65 Vergleichsgeometrien:  Minimum 0.37   Median 1.44   Maximum 5.97
   unter 0.15 eV/Å:  0        unter 0.50:  3        unter 1.00:  18
```

**Keine einzige Energiedifferenz ist auswertbar.** Eine Energie an einem Punkt,
der 1.4 eV/Å vom Stationären entfernt liegt, ist ein Hangwert und keine
Sattelhöhe.

**Der Grund:** Strukturen, die auf ωB97M-V/def2-TZVP optimiert wurden, überstehen
den Niveauwechsel nicht. Energievergleiche über Niveaugrenzen hinweg sind hier
**gegenstandslos, nicht bloß ungenau**.

**Was innerhalb der Tabelle trotzdem erwartungstreu ist:** der RKS-TS hat in
jeder Zeile den größten Gradienten (1.84 bis 5.97), BS-TS-Opt und UKS-NEB die
kleinsten. Das ist §6 an einer unabhängigen Messung wiederholt.

## Kontrollen

- **Die Bezugsseite ist sauber.** Alle 16 Einzelpunkte reproduzieren die
  Energie, auf der die zugehörige TS-Optimierung geendet hat — das Rezept trifft
  dieselbe elektronische Lösung. Ohne diesen Nachweis wäre die ganze Tabelle
  wertlos, weil dann zwei verschiedene SCF-Lösungen verglichen worden wären.
- **Der eigene Gradient der neuen Punkte liegt bei 0.002 bis 0.015 eV/Å**, also
  drei Größenordnungen unter den Vergleichsgeometrien. Sie sind auf diesem
  Niveau stationär, wie sie müssen, da sie hier optimiert wurden.
- **Die Auswertungsregel stand vor den Zahlen fest**, nicht danach.

## Vorbehalte

Die Liste, die gedruckt gehört und nicht in eine Fußnote:

- **Warum Bandverfahren an der Climbing-Image-Schwelle scheitern, ist
  unbekannt.** Drei
  Kandidaten, keiner von den anderen getrennt (§8).
- **Zwei gültige Sattelpunkte, keine Rangordnung.** In vier Reaktionen liegen
  die Punkte zweier Startpunkte 194 bis 892 meV auseinander, und beide
  bestehen alle drei Stufen (§8). Welcher der relevante ist, sagt die
  dreistufige Regel nicht — sie prüft Gültigkeit, nicht Optimalität. Eine
  Entscheidung bräuchte einen IRC von beiden aus, und der ist verworfen (A.10).
- **Keine Barrierenhöhe auf der BS-Fläche.** Das Kapitel liefert Sattelpunkte,
  aber noch keine Zahl, die in eine Tabelle mit Aktivierungsenergien gehörte.
  Dafür fehlen die Endpunktenergien auf derselben Fläche.
- **Alle BS-TS-Opt-Läufe starteten am RKS-TS.** Eine lokale Suche findet nur,
  was unter ihrem Startpunkt liegt.
- **Bei 10 von 19 Reaktionen existieren konkurrierende Sattelpunkte** (6 nach
  Stufe 3), und in vier davon ist der Abstand jetzt gemessen (§8). Ob es
  jeweils noch einen tieferen gibt, bleibt offen.
- **Die Prüfung „verbinden beide dieselben Minima?" ist nicht direkt
  durchgeführt**; die Modenanalyse ist ein Ersatz. Der IRC wurde versucht und
  verworfen (Anhang A.10).
- **Die Triage-Schwelle 0.25 eV/Å ist abgelesen, nicht bestimmt**, und beruht auf
  zehn nicht zufällig gewählten Punkten.
- **Die Modellroute deckt alle 19 ab, aber nur 12 davon gültig.** Von 19
  TS-Optimierungen ab UMA-M-Geometrie konvergieren 17; sechs der bewerteten
  landen auf einem Minimum, einem Sattelpunkt zweiter Ordnung, einem anderen
  Prozess oder sind gar nicht stationär (Anhang B).
- **Das Screening-Niveau ist für Geometrien nicht validiert** (§8, Anhang B).
- **Der multireferenzielle Satz umfasst nur zwei Summenformeln** — C5H5NO und
  C3H5NO2, zehn und neun Reaktionen (§0). Alle Aussagen dieses Kapitels über
  „Multireferenzreaktionen" sind streng genommen Aussagen über Umlagerungen
  dieser beiden Moleküle. Ob der Befund auf größere Systeme, andere Elemente
  oder andere Reaktionstypen überträgt, ist nicht geprüft. Die Kontrollgruppe
  ist mit fünf Summenformeln breiter, teilt aber 19 ihrer 26 Reaktionen mit
  denselben beiden — der Gruppenvergleich ist damit fair, die Verallgemeinerung
  nicht gedeckt.

> **Was hier bis zum 18.08. stand, gilt nicht mehr.** Der Abschnitt führte zwei
> Gegenbelege dafür, dass kein tieferer Sattelpunkt existiert: zehn Suchen von
> Modellgeometrien aus hätten nichts Neues gefunden, und die aufgeteilte Suche
> ebenso wenig. Beides ist überholt — mit den Energien aus §8 liegen in vier
> Reaktionen zwei gültige Punkte 194 bis 892 meV auseinander, und in dreien
> davon findet die Modellgeometrie den tieferen.

**Erzeugt von:** `pipeline/job_orca_sp_samelevel.sh` (Einzelpunkte),
`pipeline/job_orca_grad_samelevel.sh` (Gradienten).

---

# §10 · Ausblick — lohnt es sich, die Kräfte zu verbessern?

**[NEU 18.08.]**

§5 zeigt, dass die Modelle an genau den Geometrien danebenliegen, die sie selbst
ansteuern, und §8, dass ihre Vorhersagen trotzdem brauchbare Startpunkte sind.
Daraus folgt eine Frage, die das Kapitel beantworten sollte, weil sie darüber
entscheidet, wohin Arbeit fließt.

## Als Startpunktgeber: die Kräfte reichen bereits

```
UMA-M-Geometrie + DFT-Verfeinerung   12 von 18 gültig
                                     alle 19 Reaktionen abgedeckt
                                     ~1 Stunde statt 7 bis 45
                                     dreimal der tiefere Sattelpunkt
```

Für diese Verwendung ist der Kraftfehler gleichgültig: verfeinert wird mit DFT,
das Modell liefert nur den Ausgangspunkt. Bessere Kräfte würden ein oder zwei
der sechs Fehlschläge retten, mehr nicht.

## Als eigenständiger Vorhersager: nicht durch Modellgröße

Ohne Verfeinerung ist die Modellvorhersage in mehr als der Hälfte der
multireferenziellen Fälle kein Sattelpunkt (§4), und man sieht ihr das nicht an
(§8). Hier lohnt eine Verbesserung — aber vermutlich nicht auf dem üblichen Weg.

Der Befund aus §5 sagt, wo der Fehler sitzt: **maximal bei flacher Brechung**,
also dort, wo die beiden Lösungen nahezu entartet sind und die
Grundzustandsfläche eine Kante hat. Bei tiefer Brechung, wo ein Blatt klar
dominiert, ist der Fehler halb so groß.

Formal hat eine glatte Funktion an einer Kante nie die richtige Ableitung.
Praktisch ist die Kante eine Nullmenge, und was zählt, ist ihre Umgebung — die
ist lernbar, und mit genug Kapazität schrumpft der betroffene Bereich.

**Das verschiebt den Hebel von der Modellgröße zur Datenabdeckung.** OMol25
sampelt Gleichgewichtsstrukturen und Reaktionspfade; die Nahtregion ist ein
schmaler Bereich, den ein Pfad rasch durchquert und der dort vermutlich dünn
besetzt ist. Gezieltes Nachsampeln genau dort wäre die naheliegendste Maßnahme
— und die billigste.

> **Das ist eine Vermutung über OMol25, keine Messung.** Sie ließe sich prüfen,
> indem man zählt, wie viele Trainingspunkte bei kleinem |ΔE_BS| liegen. Dafür
> braucht es Zugriff auf die Trainingsverteilung, nicht nur auf die 45
> Reaktionen hier.

## Die strukturelle Alternative

Ein Modell, das **⟨S²⟩ oder die Spindichte mitlernt**, könnte die beiden Blätter
unterscheiden, statt über sie hinwegzumitteln. Die Zielfunktion wäre dann pro
Blatt eindeutig und ohne Kante — das Problem verschwände, statt feiner
approximiert zu werden.

Das ist der Unterschied zwischen mehr Daten und einer anderen Zielgröße. Welcher
Weg trägt, ist mit den Daten dieses Kapitels nicht zu entscheiden.

## Was daraus für die Bewertung folgt

```
mehr Parameter                 adressiert das Problem vermutlich nicht
mehr Daten in der Nahtregion   naheliegend, billig, ungeprüft
Spinzustand als Zielgröße      adressiert die Ursache, aufwendiger
DFT-Verfeinerung akzeptieren   funktioniert heute -- 12 von 18
```

Und ein Nebenbefund, der in dieselbe Richtung zeigt: **die Instabilitätsanalyse
sagt die Ausfälle vorher** (§1, AUC 0.84). Wer ein Modell einsetzt, kann die
kritischen Reaktionen für Minuten je Reaktion vorab markieren und nur dort
verfeinern — unabhängig davon, ob das Modell je besser wird.

---

# Was als Nächstes zu tun ist

*Stand 18.08.2026.*

```
ERLEDIGT seit dem 17.08.
   Kontrolle zu §8                         3 von 3, ⟨S²⟩ = 0
   Baender am Zielniveau                   11 von 19 mit CI, 10 gueltig
   Startpunkt B                            9 von 9 gueltig
   Startpunkt C                            18 bewertet, 12 gueltig
   Trenntest                               5 von 5, in Anhang B
   Hoehenvergleich innerhalb einer Reaktion  moeglich und gemacht (Sattelmatrix)

OFFEN
1  Vier Reaktionen ohne gueltigen Punkt    rxn1283, rxn4518, rxn7060, rxn7949
                                            auf allen drei Wegen gescheitert
2  Frequenzen zentral nachrechnen           ~30 min je Reaktion; aendert
                                            keine Urteile, nur die Zahlen
3  Endpunktenergien auf der BS-Flaeche      nicht begonnen -- erst dann
                                            gibt es Barrierenhoehen
4  Nahtregion in OMol25                     wie dicht ist sie abgedeckt?
                                            Braucht Zugriff auf die
                                            Trainingsverteilung, nicht nur
                                            auf die 45 Reaktionen hier
```

Zwei weitere Punkte, notiert und nicht begonnen:

- **Stabilitätsanalyse an den Originalgeometrien des Datensatzes**
  (`data/Transition1x.h5`) — sitzt der Datensatz selbst auf dem falschen Blatt?
- **Die Zwei-Blatt-Hypothese direkt prüfen**, indem die Hesse aus 6N
  verschobenen Gradienten selbst gebaut und ⟨S²⟩ je Verschiebung protokolliert
  wird. ORCAs `NumFreq` schreibt diese Zwischenrechnungen nach
  `tsopt2.lastscf`, das jede folgende überschreibt — aus der vorhandenen
  Ausgabe ist es nicht zu beantworten.

---

# Anhang A · Revisionen

Was hier steht, sind Zahlen und Deutungen, die in früheren Fassungen dieses
Kapitels oder in Vorträgen anders lauteten. Sie stehen gesammelt und nicht im
Fließtext, damit der Haupttext lesbar bleibt — aber sie stehen, aus zwei
Gründen: eine Zahl, die einmal falsch war, kursiert womöglich noch anderswo; und
eine Fehlerart, die einmal aufgetreten ist, tritt wieder auf.

Jeder Eintrag nennt die falsche Aussage, die richtige, und **warum** es
schiefging.

---

## A.1 · „22 von 22 Bandphasen durchgehend restringiert"

**Stand bis 14.08.** Daraus war abgeleitet, `BrokenSym` verliere den gebrochenen
Zustand flächendeckend.

**Richtig ist:** 14 von 19 Bändern halten die Brechung irgendwo, 11 am höchsten
Bild.

**Warum es schiefging.** Die Zahl stammte aus dem Hauptlog des NEB — und ORCA
protokolliert die Band-SCFs dort gar nicht. Gezählt wurden die
Endpunktrelaxationen der PREOPT-Phase, wo ⟨S²⟩ = 0 die richtige Antwort ist.
Eine Zählprobe hätte es gefangen: rxn4113 hat 100 Iterationen, aber nur 36
SCF-Läufe im Log.

**Folge.** Die gesamte Diagnose in §8 beruhte zwischenzeitlich auf dieser Zahl.
Seither hat jede Bandmessung eine eingebaute Positivkontrolle, die abbricht,
wenn das Rezept ein bekanntes ⟨S²⟩ nicht reproduziert.

---

## A.2 · `nebts_` als RKS-TS gelesen

**Falsch.** Die Einträge `nebts_<rxn>` in `sweep_summary` sind das eigene
BS-NEB-Ergebnis dieser Arbeit, nicht der RKS-TS.

**Folge, wäre es stehen geblieben.** Der RKS-TS wäre in 8 Reaktionen als gültiger
Sattelpunkt der BS-Fläche erschienen — die genaue Umkehrung des Befunds in §6.

**Warum es schiefging.** Die beiden Größen heißen `rks_grad.max_evang` und
`bs.bs_grad.max_evang` und stehen im selben JSON-Eintrag. Behoben, indem die
RKS-TS-Gradienten aus `stab_pipeline` gezogen und zusätzlich gegen
`saddle_matrix.txt` abgeglichen werden.

---

## A.3 · Kartesische statt massengewichteter Normalmoden

**Falsch.** Stufe 3 wurde zunächst mit kartesischen Auslenkungen ausgewertet.

**Folge.** Sechs von 30 Urteilen wichen von der massengewichteten Auswertung ab.
Kartesisch gewichtet Wasserstoff zu stark.

**Behoben,** indem die Funktionen aus `sweep_summary.py` ausgeführt statt
nachgebaut werden — so kann die Definition nicht ein zweites Mal driften.

---

## A.4 · 21 statt 19 Reaktionen in der Bandstatistik

**Falsch.** Eine Fassung iterierte über das Verzeichnis `bs_uks_neb_results`
statt über die Liste der 19. Das Verzeichnis enthält 22 Läufe, drei davon aus dem
einreferenziellen Satz (rxn1150, rxn7936, rxn7945).

**Folge.** Zwei einreferenzielle Läufe landeten in der Gruppe „Gipfel
restringiert" — dort, wo ein restringierter Gipfel die *richtige* Antwort ist.
Der Median verschob sich von 1.074 auf 0.683, die Stationärquote von 1 von 5 auf
3 von 7.

---

## A.5 · rxn7949 als Methodenversagen

**Falsch.** rxn7949 stand als „keine konvergierte Struktur" und damit als
inhaltliches Scheitern im Text.

**Richtig ist:** das Log hat kein `TOTAL RUN TIME` — es ist ein
Walltime-Abbruch.

---

## A.6 · „7 von 7" bei der Triage

**Falsch.** Die untere Gruppe der Triage stand mit 7 von 7 bestandenen Läufen.

**Richtig ist 6 von 7.** `tsopt_rxn7949_UMA-M` startet bei 0.051 eV/Å, liegt also
klar im unteren Bereich, und landet auf einem Torsionssattel im Eduktbecken
(−69 cm⁻¹, Bindungsrate 0.008, beide reaktiven Bindungen auf Eduktwerten).

**Warum es zählt.** Der Fall zeigt, dass die Triage vorhersagt, ob eine
Optimierung *konvergiert* — nicht, ob sie am richtigen Ort ankommt.

---

## A.7 · Der Prädiktortest gegen den RMSD zum RKS-TS

**Frühere Fassung.** 225 Zeilen (5 Modelle × 45 Reaktionen), Zielgröße „RMSD der
Modellvorhersage über 0.3 Å", 34 Positive.

```
  -λ_min_ext (kontinuierlich)   0.8496
  ext_stable (binär)            0.7710
  N_FOD      (kontinuierlich)   0.6963
```

**Warum das nicht haltbar ist.** Auf der instabilen Seite ist der RKS-TS kein
Stationärpunkt der Fläche, auf der die Reaktion abläuft (§6). „Abweichung vom
RKS-TS" ist dort ein *Abstand*, kein *Fehler*. Und der Test wird beinahe
zirkulär: Prädiktor und Zielgröße handeln beide von der restringierten Lösung.
Ein Gutachter könnte einwenden — natürlich weicht ein Verfahren, das der wahren
Fläche folgt, vom restringierten Sattelpunkt ab, wenn die restringierte Lösung
instabil ist. Das sagt nichts über Modellfehler.

**Was den Wechsel überlebt:**

```
                              alt (RMSD zum RKS-TS)   neu (referenzfrei)
  Instabilität schlägt N_FOD        0.850 / 0.696        0.836 / 0.776   ✓
  kontinuierlich schlägt binär      0.850 / 0.771        0.836 / 0.829   ✗
```

**Zurückgenommen:** dass der kontinuierliche Eigenwert sein binäres Abbild um
0.08 AUC schlägt und deshalb die zu berichtende Fassung sei. Referenzfrei sind
es 0.007 — ein Gleichstand. **„Instabil ja/nein" genügt.**

Die alten Zahlen bleiben in `stability_vs_fod_separation.txt` und sind dort als
*Abstand* zu lesen, nicht als Abweichung.

---

## A.8 · Unvollständige Kontrollgruppe im Kraftfehler

**Frühere Fassung.**

```
                     MAE   max Komp.  |F| Modell  |F| DFT
  RKS stabil    (26) 0.011     0.045      0.021     0.050
  RKS instabil  (44) 0.031     0.142      0.032     0.163

  UMA-S 3.4×      UMA-M 2.1×      eSEN 3.7×
```

**Das Problem.** Die 26 waren **ein Modell je Reaktion** — UMA-M 18×, UMA-S 6×,
eSEN 2×. Die Zeile `eSEN 0.010` stand damit für einen Median über zwei
Strukturen.

**Nach Nachrechnung der 52 fehlenden Paare:**

```
  RKS stabil    (78) 0.013     0.058      0.032     0.067
  UMA-S 2.7×      UMA-M 2.1×      eSEN 2.3×
```

**Was sich ändert.** Der Effekt ist kleiner — Faktoren 2.1 bis 2.7 statt 2.1 bis
3.7. Die alten Zahlen haben den Befund geschmeichelt. Dafür ist die Aussage jetzt
einheitlich, und die Kernzeile wird schärfer: das Modell meldet in beiden Gruppen
**dieselbe** Restkraft, 0.032, statt vorher 0.021 gegen 0.032.

**Kreuzprobe.** UMA-M, das als einziges schon vorher eine ordentliche
Kontrollseite hatte, ändert sich nicht.

---

## A.9 · „Der neue Sattelpunkt liegt tiefer"

**Zwischenzeitlich behauptet.** Auf ωB97X/6-31G(d) schienen die beiden
geometrisch am weitesten entfernten neuen Punkte tiefer zu liegen als alles
Bekannte — rxn6196 mit +323 bis +413 meV gegen alle vier Vergleichsstrukturen,
rxn3107 mit +110 bis +316 gegen alle fünf. Das schien die These zu belegen, dass
Modelluneinigkeit kein guter Prädiktor sei, weil ein tieferer Sattelpunkt
existieren könne.

**Das gilt nicht.** Die Übereinstimmung über vier beziehungsweise fünf Strukturen
hinweg ist kein Zeichen von Konsistenz: **ein Sattelpunkt liegt systematisch
tiefer als beliebige Punkte am Hang um ihn herum**, und genau solche Punkte sind
die Vergleichsstrukturen auf diesem Niveau. rxn6196s Vergleichswerte stehen bei
1.74 bis 3.58 eV/Å, rxn3107s bei 3.41 bis 5.00 (§9).

**Und geometrisch löst es sich ebenfalls auf.** rxn6196 lag auf dem Screening-
Niveau 1.04 Å von jeder bekannten Struktur entfernt und bestand alle drei
Stufen. Auf Zielniveau liegt derselbe Lauf 0.008 Å von BS-TS-Opt. Der
weit entfernte Punkt war ein Sattelpunkt der Screening-Fläche, den es oben nicht
gibt.

> **Die Frage selbst ist inzwischen beantwortet, nur anders.** Die Rücknahme oben
> betrifft den Vergleich über Niveaugrenzen hinweg, der gegenstandslos war. Am
> Zielniveau, wo alle Kandidaten konvergierte Sattelpunkte derselben Rechnung
> sind, ist der Vergleich möglich — und in vier Reaktionen liegen zwei gültige
> Punkte 194 bis 892 meV auseinander (§8). Es gibt also tiefere; sie waren nur
> mit der damaligen Messung nicht zu zeigen.

---

## A.10 · Der IRC als Endpunktprüfung — verworfen

**Idee.** Von zwei konkurrierenden Sattelpunkten je einen IRC starten und
prüfen, ob sie dieselben Minima verbinden. Das wäre die eigentlich richtige
Prüfung für Stufe 3.

**Verworfen**, weil falsch-negativ: der IRC bleibt auf der gebrochenen Fläche in
flachen Bereichen stehen, bevor er ein Minimum erreicht, und meldet dann
verschiedene Endpunkte für Punkte, die dieselbe Reaktion beschreiben.

**Ersatz.** Die Modenanalyse in Stufe 3. Sie prüft die Richtung, nicht die
Verbindung — das ist der Vorbehalt in §2.

---

## A.11 · „Unsere Struktur" als Name

**Interne Bezeichnung**, durchgehend ersetzt durch **BS-TS-Opt**.

**Warum der Name schädlich war.** „Unsere Struktur" liest sich wie die richtige.
An mindestens drei Stellen zeigt das Kapitel aber, dass sie es nicht ist: bei
rxn1320 mit Modenanteil 0.00 und rxn4518 mit 0.03 fällt sie durch Stufe 3, und
bei rxn0894 und rxn8885 liegt ein Minimum 180 beziehungsweise 425 meV darunter.
Ein Name, der den Maßstab unterstellt, macht solche Befunde schwerer sichtbar.

Dasselbe gilt für **RKS-TS** statt „Referenz" — aus dem Grund, der in der
Benennung am Anfang steht.

---

## A.12 · Zwei überholte Diagnosen zum NEB-Versagen

Die Erklärung, warum der UKS-NEB scheitert, hat zweimal gewechselt. Beide
Vorfassungen stehen hier, weil beide einmal im Text standen.

**Erste Fassung: `BrokenSym` ist zustandslos, das Band verliert die gebrochene
Lösung.** Widerlegt — 14 von 19 Bändern halten sie (A.1 und §7).

**Zweite Fassung: es entscheidet das oberste Bild.** Teilweise richtig: die
Korrelation in §7 ist real und gemessen. Aber sie erklärt das Versagen nicht,
weil offen bleibt, warum der Gipfel bei sechs Bändern restringiert bleibt — eine
Erklärung über einen Energieversatz an der Naht wurde geprüft und widerlegt.

**Dritte Fassung: es ist ein Konvergenzproblem an der Climbing-Image-Schwelle.**
Diese Fassung hat als einzige eine Vorhersage gemacht — die Trennung von Pfad
und Sattelpunkt müsse helfen — und die Vorhersage hat sich bestätigt. Die
*Begründung* ist trotzdem falsch, und zwar aus zwei Gründen, beide am 17.08.
nachgemessen:

```
Die alten Läufe sind nicht an der Konvergenz gescheitert
   15 von 19 melden Konvergenz, 4 laufen in die Wandzeit
   die 15 liefern Strukturen mit 0.68 bis 2.55 eV/A auf der
   Grundzustandsflaeche -- konvergiert, aber auf einer anderen Loesung

Die Aufteilung ueberspringt die Schwelle nicht
   NEB-CI treibt das hoechste Bild ebenfalls zur Konvergenz, mit einer
   VIERMAL STRENGEREN Schwelle als NEB-TS (5.00e-04 gegen 2.00e-03 Eh/Bohr)
```

Bis zum 17.08. stand in §8 die Formulierung „das Band liefert den Pfad auf der
groben Schwelle" und die Schwellenwerte „0.020 und 0.002 eV/Å" — letztere sind
die Werte der Baseline in Eh/Bohr, gelesen, als wären es eV/Å.

**Was stattdessen gilt.** **[GEÄNDERT 17.08.]** Der Unterschied besteht aus zwei
Teilen, viermal
strengere Bandschwellen und eine andere Nachbehandlung (eingefrorener Zustand
über `MORead`, Suche mit Krümmung). Der Trenntest in Anhang B zeigt, dass die
Nachbehandlung allein in fünf von fünf Fällen einen gültigen Sattelpunkt
liefert, auch vom lockeren Band aus, und in drei davon denselben. Was das
strengere Band darüber hinaus beiträgt, ist aus fünf Fällen nicht zu sagen. **Belegt ist, dass die
Aufteilung hilft; warum der ungeteilte Lauf scheitert, ist weiterhin offen.**

---

# Anhang B · Reproduktion

## Wie die Sattelpunkte beschafft wurden

Die Kurzfassung steht in §8. Hier die Einzelheiten — für jemanden, der es
nachbaut, und als Begründung dafür, warum es so und nicht anders gemacht wurde.

### Die drei Schritte, wörtlich

```
1  NEB-CI    ! UKS <METHODE> NEB-CI TightSCF SlowConv
             %neb Product "<produkt>.xyz" / NImages 8 / Preopt true end
             Endpunkte aus orca_neb_results/<rxn>/{reactant,product}.xyz
             → <name>_NEB-CI_converged.xyz

2  SP        ! UKS <METHODE> SP TightSCF SlowConv       am Climbing Image
             %scf STABPerform true / STABRestartUHFifUnstable true
                  MaxIter 500 end
             → bs.gbw

3  TS-Opt    ! UKS <METHODE> OptTS <Freq|NumFreq> TightSCF SlowConv MORead
             %moinp "bs.gbw"
             bei LEVEL=prod zusätzlich  %geom NumHess true end
```

`LEVEL` wählt zwischen `wB97M-V def2-TZVP def2/J RIJCOSX` (Zielniveau) und
`wB97X 6-31G(d)` (Screening-Niveau). Beide laufen durch dasselbe Skript, damit
die Rezepte nicht auseinanderdriften.

Am Zielniveau **muss** die Hesse numerisch sein: VV10 hat keine analytischen
zweiten Ableitungen, und `Calc_Hess true` scheitert dort mit
`ORCA_CPSCF: The CPSCF equations can not yet handle non-local correlation`.

### NEB-TS gegen NEB-CI — zwei Verhalten, ein Schlüsselwort

`NEB-TS` ist `NEB-CI` plus ORCAs eigene Verfeinerung im Anschluss. Bildzahl,
Vorrelaxation, Endpunkte und Niveau sind identisch, **beide durchlaufen eine
Climbing-Image-Phase.** Aber ORCA hängt an das Schlüsselwort auch seine
Voreinstellungen:

```
                        reguläre Bilder        Climbing Image
NEB-TS (§7)         2.00e-02 Eh/Bohr       2.00e-03 Eh/Bohr
                        = 1.028 eV/Å           = 0.103 eV/Å
NEB-CI (§8)         5.00e-03 Eh/Bohr       5.00e-04 Eh/Bohr
                        = 0.257 eV/Å           = 0.0257 eV/Å
```

**Der neue Lauf rechnet viermal genauer, nicht lockerer.** Das erklärt die
Laufzeiten — die alten Bänder waren nach 5 bis 46 Stunden fertig, die neuen
brauchen 7 bis 45 und acht erreichten die 48-Stunden-Wandzeit nicht. Und die
alte Climbing-Image-Schwelle von 0.103 eV/Å liegt bereits nahe an der
Stufe-1-Schwelle von 0.15: ein danach „konvergierter" Punkt konnte die Prüfung
gerade noch bestehen oder gerade nicht.

### Startpunkt A — neues Band

19 Läufe, 48 h Wandzeit, 12 Prozesse.

```
Climbing Image erreicht    11 von 19
TS-Optimierung konvergiert 11 von 11
besteht alle drei Stufen   10 von 11      der Ausfall ist rxn1320
Wandzeit erreicht           8 von 19
   rxn0894 rxn1283 rxn3107 rxn4518 rxn4522 rxn5691 rxn7060 rxn7949
```

Für rxn0346, rxn6196, rxn8827 und rxn1320 scheiterte Schritt 3 zunächst am
analytischen Hesse-Versuch (`error termination in SCF Hessian`) und wurde mit
`job_orca_tsopt_prod_resume.sh` und `NumHess true` nachgezogen; die Bandphase
blieb dabei erhalten.

### Startpunkt B — vorhandenes Band

Die Läufe aus §7 haben ihre Climbing Images hinterlassen. Deren *eigenes*
Ergebnis ist dort durchgefallen — Gradienten von 0.68 bis 2.55 eV/Å und zwei
imaginäre Moden. Verwendet wird nur die Geometrie des höchsten Bandbildes,
bevor ORCAs Verfeinerung darauf lief.

```
9 Läufe, alle gültig
   davon 4 Lücken aus Startpunkt A geschlossen
         rxn0894 rxn3107 rxn4522 rxn5691
   davon 1 Fehlurteil korrigiert
         rxn1320, das über A durch Stufe 3 fiel
   1 bis 2 Stunden je Reaktion
```

rxn1283 scheiterte in Schritt 2: das SCF konvergierte nicht in 500 Zyklen mit
`SlowConv`. Mit `VerySlowConv` und 1500 Zyklen ging es durch, danach lief die
TS-Optimierung in das Iterationslimit von 250 — bei einem Startgradienten von
0.0014, also bereits unter der Schwelle. Die Optimierung hat sich vom guten
Punkt entfernt.

**Der Trenntest.** Fünf Reaktionen wurden über Startpunkt B gerechnet, obwohl
Startpunkt A sie schon gelöst hatte — um zu sehen, welcher der beiden
Unterschiede zwischen §7 und §8 wirkt: die strengeren Schwellen oder die
Nachbehandlung.

```
             vom lockeren CI              vom strengen CI
        Zykl.   ⟨S²⟩    ν_imag       Zykl.   ⟨S²⟩    ν_imag
rxn8827   25   1.025   -583.6          23   1.024   -582.7
rxn5690   46   0.000   -405.9           4  -0.000   -389.8
rxn6196   41   0.496   -710.6          22   0.492   -730.6
rxn0346   16   0.328  -1705.6           —   0.598  -1244.3
rxn1320   18   0.000   -440.3           3   1.022   -248.2
```

Fünf von fünf konvergieren und bestehen alle drei Stufen, drei davon auf
demselben Punkt. **Die Nachbehandlung wirkt unabhängig davon, wie streng das
Band gerechnet wurde.** rxn6196 brauchte 41 statt 22 Zyklen und lief im ersten
Anlauf in das voreingestellte Iterationslimit — ein schlechterer Startpunkt
kostet Rechenzeit, nicht das Ergebnis. Wo es abweicht (rxn0346, rxn1320), hat
das Band entschieden.

> n = 5, und die dreistufige Regel prüft Gültigkeit, nicht Optimalität. Welcher
> Weg „besser" ist, sagen diese fünf Fälle nicht.

### Startpunkt C — Modellgeometrie

Die TS-Optimierungen starten direkt an der UMA-M-Vorhersage, ohne jedes Band.
Zwei Sätze: die zehn Triage-Läufe aus §6 und zehn weitere für die Reaktionen,
die dort fehlten (`job_bs_tsopt_umam_missing.sh`, PySCF). Bewertet wurden alle
mit derselben dreistufigen ORCA-Kette (`job_orca_umam_eval.sh`).

```
19 Reaktionen mit UMA-M-Start, keine Überschneidung der beiden Sätze
17 konvergiert         2 nicht: rxn7060 (Triage), rxn0894 (RKS-Stufe)
18 bewertet            darunter rxn7060, dessen Optimierung nicht konvergierte
12 gültig
```

Die sechs Fehlurteile verteilen sich auf alle drei Stufen:

```
rxn7060   max|F| 1.6 eV/Å       nicht stationär -- Stufe 1
rxn1283   0 imaginäre Moden     ein Minimum -- Stufe 2
rxn4518   2 imaginäre Moden     Sattelpunkt zweiter Ordnung -- Stufe 2
rxn4522   Rate 0.019            anderer Prozess -- Stufe 3
rxn7949   Rate 0.008            Torsionssattel im Eduktbecken -- Stufe 3
rxn8837   Rate 0.054            anderer Prozess -- Stufe 3
```

rxn7060 ist dabei der Fall, der zeigt, warum Stufe 1 auch hier geprüft werden
muss: seine Optimierung ist nie konvergiert, die Struktur wurde trotzdem
bewertet, und sie besteht Stufe 2 und 3 — aber sie trägt 1.6 eV/Å.

**Damit ist die Vorhersage aus dem Skriptkopf widerlegt.** Sie lautete: alle
neun Gradienten an den Modellgeometrien liegen unter 0.25 eV/Å, also erreichen
alle einen gültigen Sattelpunkt. Es sind sechs von neun. rxn1283, rxn4518 und
rxn4522 starteten bei 0.125, 0.055 und 0.083 — weit unter der Schwelle — und
landeten trotzdem falsch. Das Triage-Kriterium sagt **Konvergenz** voraus,
nicht den richtigen Ort; der Vorbehalt steht seit rxn7949 in §6 und hat jetzt
vier weitere Belege.

### Das Screening-Niveau überträgt Geometrien nicht

Auf ωB97X/6-31G(d) liefert die Aufteilung 16 von 19 Climbing Images und 15 von
16 gültige Sattelpunkte — deutlich mehr als am Zielniveau. Die Strukturen sind
dort aber andere:

```
rxn        RMSD Screening ↔ Ziel    ν Screening    ν Ziel
rxn5690           0.183 Å            -1771.7       -389.8
rxn0346           0.365              -1925.5      -1244.3
rxn7957           0.383                  —         -655.9
rxn6196           1.043               -401.4       -730.5
```

**In allen vier prüfbaren Fällen über 0.18 Å.** Der Screening-Niveau ist für die
Einstufung validiert — welche Reaktion bricht — nicht für die Frage, wo der
Sattelpunkt liegt. rxn6196 ist der deutlichste Fall: dort findet das
Screening-Niveau einen Sattelpunkt, der alle drei Stufen besteht und 1.04 Å von
jeder bekannten Struktur entfernt liegt; am Zielniveau existiert er nicht.

### Frequenzen: zentrale gegen vorwärts Differenzen

Die in `OptTS` eingebettete `NumFreq` nimmt **vorwärts**-Differenzen, 3N
Verschiebungen gegen einen Referenzpunkt, Fehler O(h). Jede eigenständige
`NumFreq` dieses Projekts nimmt zentrale, 6N und O(h²). Die Zeile
`Central differences ... used` / `... NOT used` steht in der Ausgabe.

```
           vorwärts   zentral    Stufe 3 Anteil/Rate
rxn0346    -1244.26  -1287.95    0.70/1.072 → 0.68/1.039
rxn6196     -730.55   -775.28    0.97/1.296 → 0.97/1.276
rxn8827     -582.68   -588.98    0.97/1.389 → 0.97/1.390
```

Kein Urteil ändert sich; die Frequenzen verschieben sich um 1.1 bis 6.1 %,
durchweg zu stärker negativ. Berichtet werden die zentralen Werte.

### Was nicht funktioniert hat

```
MORead direkt in der NEB-Eingabe        wird abgewiesen
NEB_Restart_GBWName, fremde Orbitale    fünf Segmentierungsfehler; der
                                        Parameter erwartet einen Basisnamen
                                        und liest <base>_im{N}.gbw je Bild,
                                        akzeptiert aber nur ORCA-eigene
                                        NEB-Orbitale
Rotate {HOMO,LUMO,...}                  braucht numerische Indizes, und
                                        scheitert auch damit im NEB
Calc_Hess true mit wB97M-V              CPSCF kann kein VV10
BrokenSym 2,2 statt 1,1                 sammelt kein zusätzliches Bild ein
                                        (rxn8837 identisch, rxn1320 verliert
                                        drei) und konvergiert bei WENIGER
                                        Brechung schneller
IRC als Endpunktprüfung                 falsch-negativ, siehe A.10
```


## Erzeugende Skripte

| Datei | Zweck | Ausgabe |
|---|---|---|
| **Detektor und Prädiktor** | | |
| `pipeline/stability_pipeline.py` | Stabilitätsanalyse, 45 Reaktionen × 4 Geometrien | `stab_pipeline/<rxn>/result.json` |
| `pipeline/predictor_reffree.py` | referenzfreier Prädiktortest, §1 | Konsole |
| `pipeline/sep_analysis.py` | Vorfassung gegen den RMSD zum RKS-TS, als Beleg für den Wechsel behalten (A.7) | `stability_vs_fod_separation.txt` |
| `pipeline/job_orca_cheap_stability.sh` + `cheap_stab_report.py` | Screening-Niveau: überträgt sich die Einstufung auf ωB97X/6-31G(d)? | `cheap_stab_report.txt` |
| **Die dreistufige Regel** | | |
| `pipeline/verdict_final.py` | die Regel, symmetrisch auf beide Seiten | Konsole |
| `pipeline/imag_mode.py` | Modenanalyse, Stufe 3 | Konsole |
| `pipeline/stage3_new.py` | Stufe 3 über mehrere Ergebnisverzeichnisse (`tsopt`, `tsopt2`, `numfreq`) | `stage3_new.txt` |
| `pipeline/mode_compare.py` | Vergleich zweier konkurrierender Sattelpunkte, §3 | Konsole |
| **Modelle** | | |
| `pipeline/job_orca_freq_sweep.sh` | Array-Job, dreistufige Kette je Struktur | `orca_freq/<label>/` |
| `pipeline/make_freq_list*.py` | Aufgabenlisten — **aus Strukturen, nicht aus Hesse-Matrizen** | `freq_tasks.txt` |
| `pipeline/model_saddle_stats.py` | 96 % / 46 % und die Stufenbilanz | `model_saddle_stats.txt` |
| `pipeline/status_matrix.py` | Zellmatrix je Reaktion × Kandidat | `status_matrix.md` |
| `pipeline/which_sheet_did_models_learn.py` | Labelhypothese | `which_sheet.txt` |
| `pipeline/force_error_at_ts.py` | Kraftfehler an der Modellgeometrie, §5 | `force_error_at_ts.txt` |
| `pipeline/job_orca_grad_gap.sh` | die 52 fehlenden Gradienten der Kontrollgruppe (A.8) | `orca_freq/<rxn>_<Modell>/` |
| `pipeline/model_sp_recheck.py` | Gegentest: gespeicherte gegen frisch gerechnete Modellkräfte | `model_sp_recheck/*.json` |
| `pipeline/model_spread.py`, `barrier_spread.py` | Modell gegen Modell, ohne Referenz | Konsole |
| `pipeline/job_bs_tsopt_umam_missing.sh` | TS-Optimierungen von Modellgeometrien, Triage | `orca_freq/tsopt_<rxn>_<Modell>/` |
| **Der RKS-TS auf beiden Flächen** | | |
| `pipeline/saddle_matrix.py` | eine Zeile je Reaktion | `saddle_matrix.txt` |
| `pipeline/gradient_comparison.py` | referenzfreier Abstand von der Stationarität | Konsole |
| `pipeline/lowest_saddle.py` | wer den tiefsten gültigen Sattel fand | `lowest_saddle.txt` |
| `pipeline/endpoint_report.py` | Stabilität an Edukt und Produkt | `endpoint_report.txt` |
| **Rechenverfahren** | | |
| `pipeline/job_bs_uks_neb18.sh` | BS-NEB, Baseline | `bs_uks_neb_results/` |
| `pipeline/job_orca_bs_neb_cheap.sh` | dieselbe Baseline am Screening-Niveau | `bs_uks_neb_cheap/` |
| `pipeline/job_orca_band_s2.sh`, `job_orca_band_s2_cheap.sh` | ⟨S²⟩ je Bandbild, mit eingebauter Positivkontrolle | `band_s2*/` |
| `pipeline/tsopt_null.py` | Nullmessung: Streuung der TS-Opt gegen die des NEB | Konsole |
| `pipeline/bs_freq.py`, `bs_freq2.py` | numerische BS-UKS-Hesse | `BSFREQ_OUT` |
| `pipeline/hess_compare.py` | ORCA gegen PySCF | `hess_cross_check.txt` |
| **Der Fix** | | |
| `pipeline/job_orca_nebci_split.sh` | **die Aufteilung**, §8 — `LEVEL=cheap\|prod`, `RXN_LIST`, `OUT_ROOT` | `bs_uks_nebci/`, `bs_uks_nebci_prod/` |
| `pipeline/job_orca_tsopt_prod_resume.sh` | Stufe 3 am Zielniveau mit `NumHess true` nachgezogen | `<rxn>/tsopt2.*` |
| `pipeline/job_orca_tsopt_from_broken.sh` | TS-Opt vom höchsten gebrochenen Bild | `tsopt_broken/` |
| `pipeline/job_orca_freq_central.sh` | Frequenz mit `CentralDiff true` an derselben Struktur | `freq_central/` |
| `pipeline/job_orca_sep_step23.sh` | Schritt 2+3 auf einem vorhandenen Climbing Image (Startpunkt B) | `sep_step23/` |
| `pipeline/job_orca_umam_eval.sh` | dreistufige Bewertung der Strukturen aus `bs_tsopt_umam` (Startpunkt C) | `orca_freq/tsopt_<rxn>_UMA-M/` |
| `pipeline/reaction_table.py` | Summenformeln und veränderte Bindungen der 45 Reaktionen | Konsole |
| `pipeline/job_orca_sep_step23.sh` | Trenntest: Schritt 2+3 auf den Climbing Images der Baseline | `sep_step23/` |
| **Grenzen** | | |
| `pipeline/job_orca_sp_samelevel.sh` | Einzelpunkte aller Kandidaten einer Reaktion auf **einem** Niveau, §9 | `sp_samelevel/` |
| `pipeline/job_orca_grad_samelevel.sh` | Gradient dazu, `EnGrad MORead` — entscheidet, welche Zeile auswertbar ist | `sp_grad/` |
| **Querschnitt** | | |
| `pipeline/checks.py` | Wachen: Positivkontrolle, Zählprobe, Sentinel, Abgleich | — |
| `pipeline/chapter_tables.py`, `chapter_tables2.py` | die Tabellen T0–T7 | `chapter_tables*.txt` |
| `pipeline/reliability_list.py` | vollständige Zuverlässigkeitsliste | Konsole |
| `pipeline/plot_saddle_landscape.py` | Hauptabbildung | `saddle_landscape.png` |
| `pipeline/plot_two_sheets.py` | Schema zweier Blätter | `two_sheets.png` |
| `pipeline/plot_spread.py`, `plot_spread_linear.py`, `plot_reliability_table.py` | die drei weiteren Abbildungen | `*.png` |

## Cluster

```
Host        slid.fysik.dtu.dk         Partition  xeon24el8
ORCA        5.0.4-gompi-2023a         Binärdatei unter $EBROOTORCA/bin/orca
Module      gompi/2023a, ORCA/5.0.4-gompi-2023a
Parallel    ORCA parallelisiert über MPI-Ränge
            → --ntasks=N --cpus-per-task=1, NICHT umgekehrt
```

## Kreuzvalidierung der Hesse-Matrizen

Die numerischen BS-UKS-Hesse-Matrizen sind in einem zweiten Code nachgerechnet
worden (PySCF gegen ORCA), damit weder ein Vorzeichenfehler noch eine falsche
Massengewichtung unbemerkt bleibt. Ergebnis in `hess_cross_check.txt`.

## Wachen — `pipeline/checks.py`

Jede Auswertung, die eine Zahl in dieses Kapitel liefert, ruft mindestens eine
davon auf. Die Datei prüft sich beim Import selbst.

```python
control(known, measured, what, tol=0.05)   Positivkontrolle: reproduziert das
                                           Rezept einen bekannten Wert?
expect(found, n, what)                     Zählprobe: sind es so viele, wie es
                                           sein müssen?
sentinel(values, what)                     schlägt an bei verdächtigen Werten
                                           (lauter Nullen, lauter gleiche)
crosscheck(mine, theirs, what, tol=1e-6)   Abgleich gegen eine zweite Quelle
header(script, inputs, note)               druckt Skript, Eingaben, Zeitstempel
orca_energy(path)                          gibt None zurück, wenn die Energie
                                           exakt 0.0 ist -- nichts gerechnet
```

## Bekannte Fallstricke

**ORCA-Aufruf und Sperren**

- **Den ORCA-Pfad nie zusammenbauen.** `$EBROOTORCA/orca` existiert nicht — die
  Binärdatei liegt in `bin/`. Der Aufruf scheitert dann pro Rechnung in
  Millisekunden, das umgebende Skript läuft weiter und meldet Erfolg über einem
  leeren Verzeichnis. `ORCA=$(which orca)` plus `[ -x "$ORCA" ] || exit`, und
  nach jeder Rechnung auf `ORCA TERMINATED NORMALLY` **und** eine Energie
  ungleich null prüfen.
- **`NoIter` rechnet nichts.** Ein Einzelpunkt mit `NoIter` überspringt die
  Eigenschaftsauswertung, gibt `FINAL SINGLE POINT ENERGY 0.000000000000` aus
  und druckt kein ⟨S²⟩. Ein fehlender Wert, der als Zahl gelesen wird.
- **`MaxIter 1` konvergiert nicht**, und ORCA hängt dann
  `(SCF not fully converged!)` an die Energiezeile, sodass das letzte Feld ein
  Wort ist. Zum Auslesen gespeicherter Orbitale eignet sich ein gewöhnlicher
  `TightSCF`-Einzelpunkt mit `MORead`: von einer konvergierten Wellenfunktion
  aus braucht er einen Zyklus und bleibt auf derselben Lösung.

**Was ORCA nicht kombiniert**

- **`STABPerform` verträgt nur `RunTyp SinglePoint`.** Mit `EnGrad` oder `Opt`
  bricht der Lauf nach zwei Sekunden ab mit
  `WARNING: Only RunTyp == SinglePoint possible with Stability Analysis!` und
  `Skipping actual calculation`. Wer den Gradienten *derselben* Lösung braucht,
  rechnet den Einzelpunkt mit Stabilitätsanalyse, speichert die Orbitale und
  hängt einen `EnGrad`-Lauf mit `MORead` daran.
- **`Calc_Hess true` mit ωB97M-V** scheitert an
  `ORCA_CPSCF: The CPSCF equations can not yet handle non-local correlation`.
  Stattdessen `NumHess true`.
- **`MORead` in einer NEB-Eingabe** wird abgewiesen. `NEB_Restart_GBWName`
  erwartet einen **Basisnamen** und liest `<base>_im{N}.gbw` je Bild, akzeptiert
  aber nur ORCA-eigene NEB-Orbitale — fremde führen zu Segmentierungsfehlern.

**Numerik**

- **Numerische Frequenzen: die Differenzenformel prüfen.** Eine eigenständige
  `NumFreq` nimmt hier zentrale Differenzen (6N Verschiebungen), die in eine
  `OptTS` eingebettete dagegen vorwärts (3N, Fehler O(h) statt O(h²), und ohne
  Gegenschubser, an dem eine abweichende SCF-Lösung auffiele). Die Zeile
  `Central differences ... used` / `... NOT used` steht in der Ausgabe; bei
  gebrochener Symmetrie gehört sie kontrolliert, nicht vorausgesetzt.
- **`BrokenSym` konvergiert zuerst den Hochspin-Triplett** — bei
  ⟨S²⟩-Statistiken mit Schwelle 1.8 abtrennen.
- **Normalmoden massengewichtet auswerten**, nicht kartesisch (A.3).
- **Die Modenzahl von ORCA nehmen, die Richtung aus der Hesse** (§0).

**Buchführung**

- **Modell-NEBs prüfen, ob sie konvergiert sind.** Vier von 19 sind es nicht,
  und ihr letzter Schritt sieht in der Ausgabedatei aus wie jeder andere.
- **Aufgabenlisten aus vorhandenen *Strukturen* bauen, nicht aus vorhandenen
  *Hesse-Matrizen*** — sonst kann keine Lücke je geschlossen werden.
- **SLURM-Arrays erscheinen als *eine* Zeile in `squeue`** (`10737453_[4-44]`
  sind 41 wartende Aufgaben, keine abgestürzten). Mit `-r` aufschlüsseln.
- **Über die Reaktionsliste iterieren, nicht über das Verzeichnis** (A.4).
- **Ein Index aus einem Lauf gilt nicht im nächsten.** Der Index des höchsten
  Bildes ist bandspezifisch; ihn von einem Band am Zielniveau auf eine Messung am
  Screening-Niveau zu übertragen liefert stillschweigend das falsche Bild.

---

*Ende. Vorfassung als `chapter_mr_transition_states.md`, Commit e24f53f.*
