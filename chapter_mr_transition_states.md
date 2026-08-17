# Kapitel: Übergangszustände bei Multireferenzcharakter

Zwei Teile. **Teil A** ist die Story — die Gliederung, die Aussagen, die
Takeaways, ohne Beleg. **Teil B** ist dieselbe Gliederung noch einmal, jede
Aussage mit ihren Zahlen, dem Rechenweg und der erzeugenden Datei.

Wer das Kapitel schreibt, liest A. Wer eine Zahl belegen oder nachrechnen
muss, springt in B in denselben Abschnitt.

Stand: 2026-08-13. Alle Zahlen aus dem vollständigen Frequenz-Sweep
(106 Hesse-Matrizen) und der Endpunktprüfung (90 Einzelpunkte).

---
---

# TEIL A — DIE STORY

## Roter Faden

> Ein Übergangszustands-Benchmark ruht auf zwei Annahmen: dass die Struktur,
> gegen die gemessen wird, ein Übergangszustand ist, und dass eine konvergierte
> Rechnung bedeutet, was sie sagt. Bei einer vorab erkennbaren Teilmenge trifft
> keine von beiden zu — und die Modelle scheitern dort aus einem dritten, davon
> unabhängigen Grund.

## Bauprinzip

Jeder Abschnitt endet in derselben zweispaltigen Form, immer in derselben
Reihenfolge. Der Leser weiß dann an jeder Stelle, wo er steht.

```
                                  RKS stabil (26)     RKS instabil (19)
```

Die linke Spalte ist nicht nur Kontrast, sie ist die **Kontrolle**: sie zeigt,
dass Aufbau, Kriterien und Auswertung in Ordnung sind. Was rechts bricht,
bricht nicht am Handwerk.

## Benennung

**RKS-TS** statt „Referenz", durchgehend. Bei der ersten Erwähnung mit dem
Satz, der die Umbenennung zur Aussage macht:

> Die Struktur, gegen die der Benchmark misst, ist ein Sattelpunkt der
> restringierten Fläche. Wo die restringierte Lösung stabil ist, ist das
> dieselbe Fläche wie die richtige, und der RKS-TS ist eine gültige Referenz.
> Wo sie instabil ist, sind es zwei Flächen — und das Wort „Referenz" trägt
> dann eine Annahme, die nicht mehr gilt.

**BS-TS-Opt** statt „unsere Struktur". Gemeint ist die Struktur aus einer
gebrochen-symmetrischen TS-Optimierung, die am RKS-TS startet — eines von
mehreren Suchverfahren im Kapitel, nicht der Maßstab. Der interne Name
unterstellte genau das.

---

## §1 · Wo könnte es klemmen, und wie findet man es vorher

**Frage:** Multireferenzcharakter ist als Problem bekannt. Wie findet man die
betroffenen Reaktionen, bevor man rechnet?

**Antwort:** Die externe Stabilitätsanalyse an der Eduktgeometrie. Minuten pro
Reaktion.

**Der Test.** Nimm eine Reaktion, bei der ein Modell später eine ungültige
Struktur liefert, und eine, bei der es klappt. Wie oft zeigt der Deskriptor
vorher auf die richtige?

```
von 100 solchen Paaren:

  Instabilitätsanalyse   ████████████████▌      84
  N_FOD                  ██████████████         78
  --------------------------------------------------
  Münzwurf               █████████              50
```

**Ungültig heißt hier: die Modellstruktur ist kein Stationärpunkt** — max|F| ≥
0.15 eV/Å an ihrer eigenen Geometrie, gemessen mit DFT. Es geht kein
Vergleichspunkt ein. Das ist wichtig, weil ein Vergleich mit dem RKS-TS genau
dort nichts taugt, wo die Frage interessant wird (§3.0): dort ist der RKS-TS
selbst kein Sattelpunkt, und „Abstand zum RKS-TS" wäre dann Abstand, nicht
Fehler.

**Und so sieht die Trennung aus:**

```
                    Strukturen   davon Stationärpunkte
  RKS stabil            78            74     95 %
  RKS instabil          44            19     43 %
```

Eine Rechnung von Minuten an der Eduktgeometrie, und man weiß, ob man den
Modellen hier trauen kann.

**Das Argument gegen „das misst doch dasselbe".** Nimm nur die Reaktionen, die
N_FOD ohnehin als verdächtig meldet, und teile sie noch einmal:

```
  N_FOD > 0.5,  RKS stabil       27 Strukturen     89 % Stationärpunkte
  N_FOD > 0.5,  RKS instabil     41                44 %
```

N_FOD hätte beide gleich behandelt. Die Stabilitätsanalyse trennt sie um den
Faktor zwei. Dazu zwei Fälle zum Anfassen: rxn7945 und rxn7937 haben hohes
N_FOD und trotzdem stabile Übergangszustände — bei ihnen sitzt die Brechung am
Produkt, nicht am Barrierenkamm.

Die Paartabelle enthält nur, was **ohne** RKS-TS messbar ist:

```
Modelle untereinander        0.0045 Å            0.051 Å
Barrierenstreuung            0.3 meV             13.9 meV
λ_min_ext                    ≥ 0                 < 0
```

Der RKS-TS-Abstand kommt separat, und nur links, als Validierung:

> Im stabilen Satz reproduzieren alle drei Modelle den RKS-TS auf 0.005 Å. Das
> ist der Beleg, dass Modelle, Aufbau und Auswertung in Ordnung sind — und der
> Grund, dieselbe Auswertung rechts ernst zu nehmen.

**Übergang, der den fehlenden Vergleichspunkt zum Vorteil macht:**

> Im instabilen Satz steht dieselbe Zahl nicht zur Verfügung, weil der RKS-TS
> dort nicht die Antwort ist. Genau deshalb misst dieses Kapitel referenzfrei —
> und genau deshalb taugt die Stabilitätsanalyse als Detektor: sie braucht die
> Antwort nicht, die sie sucht.

**Takeaway:** Betroffene Reaktionen lassen sich vorab und billig
identifizieren. Rechts hören die Modelle auf, sich einig zu sein — nicht über
die Struktur, sondern über die Energie.

---

## §2 · Wie die Modelle scheitern

Vier Schritte, alle hier, keiner vertagt.

**(a) Den naheliegenden Einwand vorweg erledigen.**

> Die Modelle haben die richtige Fläche gelernt. Ihre Barriere trifft die
> Broken-Symmetry-Barriere an ihrer eigenen Vorhersagegeometrie auf 0.01 bis
> 0.04 eV; die restringierte verfehlt sie um bis zu 3 eV. 32 von 33
> entscheidbaren Fällen.

Ohne diesen Absatz liefert der Leser selbst eine Erklärung — „schlechte
Trainingsdaten" — und die ist widerlegt.

**(b) Die Form des Versagens.**

16 bis 17 von 19 Vorhersagen tragen genau eine imaginäre Mode. Nur 7 bis 13
sind stationär. *Die Lage geht verloren, nicht die Form.*

**(c) Die Ursache.**

> Das Modell glaubt, bei 0.032 eV/Å zu stehen; tatsächlich wirken 0.163. Ein
> NEB hält an, wenn *seine* Kraft klein ist — also hält er an der falschen
> Stelle an und meldet Erfolg.

**(d) Die Einordnung, ohne die es wie ein Verriss klingt.**

> 0.031 eV/Å ist für diese Modellklasse kein schlechter Kraftfehler. Er hat nur
> dieselbe Größe wie das Konvergenzkriterium — dann trägt das Kriterium keine
> Information mehr. Für eine Molekulardynamik wäre derselbe Fehler folgenlos.

```
gültige Sattelpunkte         96 %                46 %
Kraftfehler                  unkritisch          ≈ Konvergenzkriterium
```

**Takeaway:** Kein schlechtes Modell — eine Aufgabe, bei der genau dieser
Kraftfehler die Abbruchbedingung aushebelt.

---

## §3 · Auch die Rechnungen brechen

Eröffnung mit der stärksten Einzelzahl:

> Der RKS-TS ist bei **0 von 19** Reaktionen ein Stationärpunkt der Fläche, auf
> der die Reaktion abläuft. Median-Gradient 1.697 eV/Å gegen 0.043 im stabilen
> Satz.

Direkt danach die Eingrenzung, damit die Aussage nicht größer klingt, als sie
belegt ist:

> Alle 45 Edukte sind geschlossenschalig, 40 von 45 Produkten auch. Jede
> Vorwärtsbarriere steht damit auf einem korrekten Nullpunkt; betroffen sind
> Reaktionsenergien und Rückbarrieren von fünf Reaktionen, um 2 bis 84 meV. Das
> Problem sitzt ausschließlich am Barrierenkamm.

### §3a · Wo der UKS-NEB stehen bleibt

**Befund:** Von 19 Bändern halten **14** die Symmetriebrechung irgendwo. Was
darüber entscheidet, ob der NEB ein brauchbares Ergebnis liefert, ist aber
nicht, ob das Band irgendwo bricht, sondern ob **das höchste Bild** bricht —
denn aus ihm wird das Climbing Image und damit der Übergangszustand.

```
                      n   Grad auf der BS-Fläche   stationär   RMSD zum RKS-TS
Gipfel gebrochen     11         0.011 eV/Å          7 von 7        0.529 Å
Gipfel restringiert   8         1.074               1 von 5        0.055
```

Faktor 100 im Gradienten, Faktor 10 im Abstand. Ein Band, dessen oberstes Bild
restringiert bleibt, liefert eine Struktur, die praktisch **auf dem RKS-TS**
sitzt.

**Der eine Stationärpunkt in der rechten Spalte ist rxn5690** (Gradient 0.004)
— der Grenzfall des Satzes, N_FOD 0.433, ΔE_BS −1.3 meV, ⟨S²⟩ 0.068 am RKS-TS.
Faktisch einreferenzig, nur nicht so etikettiert. Ohne ihn steht dort 0 von 4.

**Und das ist kein Konvergenzversagen.** Der Gradient oben ist auf der
Grundzustandsfläche gemessen, nicht auf der, auf der der NEB gerechnet hat. Ein
Band mit restringiertem Gipfel konvergiert sauber — gegen den restringierten
Sattelpunkt. Auf der gebrochenen Fläche nachgemessen zeigt der dann 0.68 eV/Å.
Es ist dieselbe Beziehung wie beim RKS-TS in §3 Eröffnung: stationär auf der
einen Fläche, nicht auf der anderen. Die richtige Antwort auf die falsche Frage.

Das begriffliche Stück gehört hierhin: **zwei Blätter, nicht zwei Rechnungen.**
Ein Band kann teils auf dem einen, teils auf dem anderen liegen, und wo genau
der Gipfel fällt, entscheidet dann über das Ergebnis.

```
Band bricht irgendwo              14 von 19
Gipfel gebrochen                  11 von 19
UKS-NEB gültiger Sattelpunkt       8 von 19
```

> **Die Zahlen sind auf die 19 beschränkt.** `bs_uks_neb_results` enthält 22
> Läufe, drei davon aus dem einreferenzigen Satz (rxn1150, rxn7936, rxn7945).
> Eine erste Fassung hat über das Verzeichnis iteriert statt über die Liste und
> zwei davon in die rechte Spalte gezählt — dort, wo ein restringierter Gipfel
> die richtige Antwort ist. Das verschob den Median von 1.074 auf 0.683 und die
> Stationärquote von 1 von 5 auf 3 von 7.

**Takeaway:** Der NEB verliert die gebrochene Lösung nicht — sie ist da, in 14
von 19 Bändern. Er bekommt sie nur nicht bis zur geforderten Genauigkeit zur
Ruhe. Der Ansatzpunkt ist damit weder die Zustandsführung noch die Auswahl des
Gipfelbildes, sondern die Frage, ob man Pfad und Sattelpunkt überhaupt in
einem Lauf verlangen muss (§3d).

> **Wo die Läufe tatsächlich hängen bleiben.** ORCA fordert für das Band
> max|Fp| ≤ 0.020 eV/Å, für das Climbing Image aber max|FCI| ≤ **0.002** —
> zehnmal strenger. Bewegt wird das Climbing Image mit denselben Kräften
> erster Ordnung wie der Rest des Bandes, ohne jede Krümmungsinformation.
> Praktisch jeder abgebrochene Lauf steht an dieser einen Schwelle.

> **Zwei Erklärungen sind geprüft und ausgeschieden.** Erstens der flächige
> Spinkollaps: widerlegt, 14 von 19 Bändern halten den gebrochenen Zustand.
> Zweitens die Vermutung, mehr erzwungene Brechung helfe: `BrokenSym 2,2`
> sammelt kein zusätzliches Bild ein (rxn8837 identisch, rxn1320 verliert
> drei) und konvergiert bei *weniger* Brechung schneller.

> **Offen und ausdrücklich nicht beantwortet:** *warum* ein Bandverfahren an
> dieser Schwelle scheitert. Drei Kandidaten, die die vorliegenden Daten nicht
> voneinander trennen — die Zwei-Blatt-Struktur der Fläche mit einer Naht, die
> ein Band zwangsläufig kreuzt; `BrokenSym` als zustandsloses Verfahren, das
> den Startzustand bei jedem SCF neu herleitet; und das Fehlen zweiter
> Ableitungen im Bandverfahren. Ebenso offen bleibt, warum bei sechs Bändern
> der Gipfel restringiert bleibt; eine Erklärung über einen Energieversatz an
> der Naht wurde geprüft und **widerlegt** (§3.2). Was belegt ist: wo es
> hängt, und was es behebt. Nicht: warum es hängt.

> **Was hier vorher stand, war falsch.** Bis zum 14.08. behauptete dieser
> Abschnitt „22 von 22 Bandphasen durchgehend restringiert" und leitete daraus
> ab, `BrokenSym` verliere den Zustand flächendeckend. Diese Zahl stammte aus
> dem Hauptlog des NEB, das die Band-SCFs gar nicht enthält — gezählt wurden
> die Endpunktrelaxationen, wo ⟨S²⟩ = 0 richtig ist. Die Herleitung in §3.2.

### §3b · TS-Opt hängt vom Startpunkt ab

Das subtilste Versagen, weil die Rechnung **korrekt** ist. Sie startet an einer
ungültigen Geometrie und konvergiert sauber gegen den Sattelpunkt, der von dort
aus erreichbar ist.

```
rxn1320, rxn4518   konvergiert, genau eine imaginäre Mode, tiefer als der
                   RKS-TS — und trotzdem nicht verwertbar, weil sich
                   „besseren Sattel gefunden" nicht von „falschen Sattel
                   gefunden" unterscheiden lässt
```

Beleg für die Startpunktabhängigkeit: Suchen vom RKS-TS und von
Modellgeometrien aus enden verschieden. Bei 10 von 19 Reaktionen existieren
konkurrierende Sattelpunkte — was nach §3a genau das ist, was zwei Blätter
erwarten lassen.

```
gültige Sattelpunkte         konvergiert gegen    13 von 19
                             den RKS-TS
konkurrierende Sattel        keine                10 von 19
```

**Takeaway:** TS-Opt ist mit 13 von 19 die beste Methode im Feld — und liefert
trotzdem keine verlässliche Antwort, weil ihr Ergebnis eine Funktion des
Startpunkts ist.

### §3c · Bilanz und Dreistufenbewertung

```
gültige Sattelpunkte je 19   alles konvergiert    TS-Opt 13 · UMA-M 11
                             gegen dieselbe       UKS-NEB 8 · eSEN 7
                             Struktur             UMA-S 6
```

**Keine Methode über zwei Drittel.**

Die Dreistufenbewertung gehört hierhin, weil sie das Werkzeug ist, mit dem §3a
und §3b überhaupt auseinanderzuhalten sind:

1. stationär und tiefer
2. genau eine imaginäre Frequenz
3. die imaginäre Mode bewegt die Bindungen **dieser** Reaktion

> Eine imaginäre Frequenz beweist einen Sattelpunkt, nicht welchen. Stufe 3 hat
> zwei Urteile umgedreht, je eines gegen jede Seite — das ist der Grund, ihr zu
> trauen.

**Definition, hier eingeführt:**

> **Unabhängig bestätigter Sattelpunkt** — eine Struktur, die (a) alle drei
> Stufen besteht und (b) von mindestens zwei Suchen erreicht wird, die weder
> Startpunkt noch Optimierungsverfahren teilen. Übereinstimmung ist kein
> Beweis. Sie ist der stärkste Beleg, der ohne gültigen Vergleichspunkt zu
> haben ist.

**Abgrenzung, sonst liest §4 sich wie zweierlei Maß:**

> Die drei Modelle teilen Trainingsdaten, Zielniveau und verwandte
> Architekturen — ihre Übereinstimmung kann geerbt sein. BS-TS-Optimierung und
> NEB-TS teilen zwar das Verfahren am Ende, aber nicht den Startpunkt: die eine
> beginnt am RKS-TS, die andere am Climbing Image eines Bandes. Zwei weit
> auseinanderliegende Startpunkte, die im selben Becken landen, sind eine
> Aussage über das Becken — geteilte Trainingsdaten sind keine.

### §3d · Pfad und Sattelpunkt trennen

Bis hierher ist §3 Diagnose. Am Ende steht eine Fläche, auf der **kein
einziger** der bekannten Punkte stationär ist — und damit nichts, womit sich
weiterrechnen ließe: keine Barriere, keine Frequenz, kein Vergleichswert für
irgendein Modell. Dieser Abschnitt ist der einzige, der etwas herstellt.

**Die Idee.** Ein NEB-CI soll zwei Dinge zugleich leisten: ein Band, das den
Pfad beschreibt, und ein Bild, das exakt auf dem Sattelpunkt sitzt. Die
Genauigkeitsforderungen unterscheiden sich um den Faktor zehn, das Werkzeug
ist für beides dasselbe. Also verlangen wir es nicht mehr in einem Lauf: das
Band liefert den Pfad auf der groben Schwelle, sein Climbing Image geht als
Startpunkt in eine eigene BS-TS-Optimierung mit exakter Hesse und
Eigenvektorverfolgung, und die konvergierten Orbitale werden über `MORead`
weitergereicht, statt bei jedem Schritt neu hergeleitet zu werden.

**Was dabei herauskommt.**

```
                                       Climbing Image   TS-Opt   1 imag   alle 3 Stufen
NEB-CI aufgeteilt, wB97X/6-31G(d)         16 von 19      16/16    16/16     15 von 16
vom hoechsten gebrochenen Bild                     —      4/4      4/4       3 von 4
Produktion, wB97M-V/def2-TZVP                        3 von 3      3/3       3 von 3
```

Zusammengenommen haben **17 der 19** Reaktionen einen gebrochen-symmetrischen
Sattelpunkt, der alle drei Stufen besteht. Offen bleiben rxn7060, das auf
keinem Niveau ein Climbing Image erreicht, und rxn7949, dessen Punkt sauber
konvergiert, aber mit einem Modenanteil von 0.02 zu einer anderen Umlagerung
gehört. Zum Vergleich der Ausgangslage: **0 von 19**.

**Und der Teil, der es belastbar macht.** Auf Produktionsniveau liegen die
neuen Sattelpunkte auf Tausendstel Ångström auf Strukturen, die über einen
völlig anderen Weg gefunden wurden:

```
rxn      BS-TS-Opt   RKS-TS   UKS-NEB    UMA-M   TSoptM
rxn0346      0.002    0.173     0.177    0.010    0.001
rxn6196      0.008    0.101     0.137    0.070        —
rxn8827      0.006    0.355     0.371    0.132    0.003
```

Ein Verfahren, das nur Neues liefert, wäre verdächtig. Eines, das die
bekannten Punkte wiederfindet **und** dort liefert, wo vorher keiner war, ist
es nicht. Damit erfüllt jeder dieser drei Punkte die Definition des
*unabhängig bestätigten Sattelpunkts* aus §3c: dieselbe Struktur, erreicht von
zwei Suchen, die weder Startpunkt noch Verfahren teilen.

> **Was hier ausdrücklich nicht steht: ein neuer oder tieferer Sattelpunkt.**
> Auf dem billigen Niveau sahen drei Punkte danach aus, allen voran rxn6196 mit
> 1.04 Å Abstand zu jeder bekannten Struktur. Auf Produktionsniveau liegt
> derselbe Punkt 0.008 Å von der BS-TS-Opt-Struktur. Es war ein Artefakt des
> Niveaus. Die Methode reproduziert, sie entdeckt nicht — jedenfalls nicht in
> den bisher gerechneten Fällen (§4.4).

> **Kontrolle, läuft.** Dasselbe Rezept auf drei einreferenzigen Reaktionen,
> deren restringierte Lösung innen und außen stabil ist (rxn1061, rxn0101,
> rxn0896, ausgewählt nach dem Abstand zur Instabilität). Erwartung: ⟨S²⟩ ≈ 0
> und derselbe Punkt wie der RKS-TS. Käme dort eine gebrochene Lösung heraus,
> bräche das Verfahren Symmetrie, wo keine zu brechen ist — und der Befund an
> den 19 wäre entwertet. Ohne diese Kontrolle ist §3d nicht abgeschlossen.

**Erzeugt von:** `pipeline/job_orca_nebci_split.sh` (`LEVEL=cheap|prod`),
`pipeline/job_orca_tsopt_prod_resume.sh`, ausgewertet mit
`pipeline/stage3_new.py`.

---

## §4 · Thesen

**Relabeling reicht nicht.** Ein UKS-Einzelpunkt auf einem RKS-Pfad ist eine
Energie an einem Punkt, an dem noch 1.70 eV/Å Kraft wirkt. Die Barriere daraus
ist keine Barriere. Die Geometrie muss auf der unrestringierten Fläche
optimiert werden, nicht nur dort ausgewertet.

**Modell-Uneinigkeit, zweiseitig:**

```
Uneinigkeit  →  verlässliches Warnsignal. Die fünf größten Streuungen sind
                genau die Reaktionen, bei denen kein Modell einen gültigen
                Sattelpunkt findet.
Einigkeit    →  kein Beleg. Gemeinsame Trainingsdaten, gemeinsames Zielniveau,
                verwandte Architekturen — und ein tieferer Sattelpunkt, den
                alle drei verfehlen, bleibt möglich.
```

**RKS-Instabilität ist der beste Prädiktor** — AUC 0.84 gegen 0.78 für
N_FOD, referenzfrei gemessen. Sie sagt drei Dinge zugleich voraus: dass die
Modelle danebenliegen, dass der RKS-TS ungültig ist, und dass auch die
UKS-Rechnung schwierig wird.

**Reichweite, ehrlich:** Wir zeigen, dass der RKS-TS ungültig ist — nicht, was
an seine Stelle gehört. Alle Optimierungen starteten an eben dieser Struktur;
ein tieferer Sattelpunkt ist prinzipiell nicht ausgeschlossen. Der einzige
Gegenbeleg: zehn zusätzliche Suchen von Modellgeometrien aus haben nichts Neues
gefunden.

---

## §5 · Ausblick

Je ein Vorschlag pro Fehlerart, aus der Diagnose abgeleitet:

| Fehlerart | Vorschlag |
|---|---|
| Gipfel auf dem falschen Blatt (§3a) | Feinere Bänder, damit die gebrochene Region aufgelöst wird und das höchste Bild in sie fällt — der Mechanismus dahinter ist offen |
| Startpunktabhängigkeit (§3b) | Mehrfachstart; Triage über den DFT-Gradienten an der Modellgeometrie — unter 0.25 eV/Å gelingt die Verfeinerung 6/7, darüber 0/3 |
| Kraftfehler der Modelle (§2) | Modell als Startpunkt, DFT als Verfeinerung — nicht Modell als Antwort |

Prüfstand validiert: ωB97X/6-31G(d) enthält alle 19 als Teilmenge von 26, mit
tieferer Brechung und analytischen Hesse-Matrizen.

---

**Abbildungen:** `model_spread_linear.png` → §1 · `two_sheets.png` → §3a ·
`saddle_landscape.png` → §3c als Hauptabbildung

**Tabellen:** AUC-Vergleich → §1 · `which_sheet.txt`, `force_error_at_ts.txt`
→ §2 · `endpoint_report.txt` → §3 Eröffnung · `saddle_matrix.txt` → §3c

**Weglassen:** „Die BS-TS-Opt-Strukturen sind die richtigen Übergangszustände" (nicht
belegbar) und „die Modelle brechen bei Multireferenz ein" (referenzfrei
Faktor 2, nicht 25 — und nach §2 die falsche Beschreibung).

---
---

# TEIL B — DIESELBE STORY, UNTERFÜTTERT

Gleiche Nummerierung wie Teil A. Jeder Abschnitt: **Behauptung → Zahlen →
Rechenweg → Quelldatei → Vorbehalt.**

---

## §0 · Aufbau, der für alles gilt

### Der Datensatz

45 Reaktionen aus dem Transition1x-Benchmark, ausgewählt über den
N_FOD-Screen. Die Einteilung in die beiden Gruppen wurde **allein am
Übergangszustand** vorgenommen:

```
RKS stabil    26 Reaktionen   externe Stabilitätsanalyse am RKS-TS: λ_min ≥ 0
RKS instabil  19 Reaktionen   λ_min < 0, ORCA rotiert in eine gebrochene Lösung
```

Diese Einteilung ist an einer Stelle nachweislich zu grob, und das gehört ins
Kapitel: **rxn7945 und rxn7937** tragen hohe N_FOD-Werte (0.903, 0.877), haben
aber einen extern stabilen Übergangszustand und stehen deshalb links. Ihre
Symmetriebrechung sitzt am *Produkt*. Das Etikett stimmt für den
Übergangszustand und nicht für die Reaktion.

### Niveau der Theorie

| | |
|---|---|
| Produktion | ωB97M-V/def2-TZVP |
| Prüfstand (§5) | ωB97X/6-31G(d) |
| Programme | ORCA 5.0.4, PySCF |
| Modelle | UMA-S, UMA-M, eSEN (zusätzlich MACE und MACE+delta in §1) |

**Warum ωB97M-V teuer ist:** der VV10-Dispersionsterm (NLC) hat weder in PySCF
noch in ORCA 5.0.4 analytische zweite Ableitungen. Jede Hesse-Matrix ist
numerisch, 6N zentrale Differenzen. Das ist der Grund, warum der Prüfstand auf
ωB97X wechselt — dort gibt es kein VV10 und damit **analytische** Hesse-Matrizen.

### Die drei Stufen, operationalisiert

| Stufe | Kriterium | Zahlenwert |
|---|---|---|
| 1 | stationär | Gradientennorm < 0.15 eV/Å |
| 2 | Sattelpunkt erster Ordnung | genau eine imaginäre Frequenz |
| 3 | gehört zu dieser Reaktion | imaginäre Mode gegen die reaktiven Bindungen |

**Kalibrierung von Stufe 1:** die bestätigten BS-TS-Opt-Sattelpunkte liegen bei
0.006 bis 0.011 eV/Å. Die Schwelle 0.15 ist damit um mehr als eine
Größenordnung großzügiger als das, was eine konvergierte Optimierung liefert —
sie schließt nichts Grenzwertiges aus.

**Reaktive Bindungen** werden nicht von Hand gewählt: die beiden Atompaare mit
dem größten |d_Produkt − d_Edukt|.

**Marker für Stufe 3** (bewusst nicht automatisiert, siehe Vorbehalt unten):
Modenanteil auf den vier reaktiven Atomen (unter ~0.10 sitzt die Bewegung
woanders), Bindungsrate d/dQ (unter ~0.05 berührt die Mode die
Reaktionskoordinate nicht), und die Bindungslänge selbst (liegt eine reaktive
Bindung schon bei ihrem normalen Wert, ist die Reaktion dort abgeschlossen).

> **Vorbehalt zu Stufe 3.** Die Schwellen sind nicht automatisiert, weil sie an
> genau den zwei Fällen kalibriert wären, die sie entscheiden sollen. Die
> Skripte geben die Zahlen aus; das Urteil steht im Text und ist nachprüfbar.

### Die ORCA-Kette und warum sie dreiteilig ist

`STABPerform` lässt sich in ORCA 5.0.4 **nicht** mit `EnGrad`, `NEB` oder `Opt`
kombinieren:

```
Only RunTyp == SinglePoint possible with Stability Analysis
```

Vier Jobs sind daran in drei Sekunden gestorben, bevor das aufgefallen ist.
Die Kette ist deshalb dreistufig, mit `MORead` als Bindeglied:

```
1a   ! UKS wB97M-V def2-TZVP SP TightSCF     %scf STABPerform true
                                                  STABRestartUHFifUnstable true
1b   ! UKS wB97M-V def2-TZVP EnGrad          %moinp aus 1a   (MORead)
2    ! UKS wB97M-V def2-TZVP NumFreq         %moinp aus 1a
```

Stufe 1a entscheidet, auf welchem Blatt gerechnet wird; 1b und 2 erben es über
die Orbitale. Ohne diese Reihenfolge misst man den Gradienten auf der falschen
Fläche.

> **Praxishinweis:** `NumFreq` schreibt ein *reduziertes* Log. Die SCF jeder
> einzelnen Auslenkung landet in `numfreq.lastscf` und wird überschrieben. Wer
> ⟨S²⟩ über die Auslenkungen prüfen will, muss es zur Laufzeit abgreifen.

### `BrokenSym 1,1` — was es tatsächlich tut

Zwei Eigenschaften, die beide Konsequenzen haben:

1. **Es ist zustandslos.** Der gebrochene Startraten wird bei *jeder* SCF neu
   hergeleitet, nicht vom vorherigen Schritt übernommen. Das ist die Ursache in §3a.
2. **Es konvergiert zuerst den Hochspin-Triplett** (⟨S²⟩ ≈ 2.0) und flippt dann.
   Wer ⟨S²⟩ aus einem Log zählt, zählt diese Referenz mit. Eine frühere Auswertung
   von mir hat genau das getan und „5 von 11 kohärenten Bändern" gemeldet.

Alle Auswertungen in diesem Dokument trennen mit der Schwelle ⟨S²⟩ > 1.8 ab.

**Und eine zweite Trennung ist genauso nötig:** `NEB-TS` protokolliert ⟨S²⟩ in
*zwei* Phasen — Bandoptimierung und anschließende TS-Optimierung auf dem
Climbing Image. Wer beide zusammenzählt, schreibt der Bandphase eine Brechung
zu, die erst danach entsteht. Genau dieser Fehler steckte in der ersten Fassung
von §3.2. Teilungsmarke ist die Zeile `THE NEB OPTIMIZATION HAS CONVERGED`.

### Dichtematrix, nicht Orbitale

Wo ein Zustand über Geometrieschritte weitergereicht wird, geschieht das über
die **Dichtematrix** (`dm0`), nie über MO-Koeffizienten: Letztere sind nur
bezüglich der Überlappungsmatrix der Ausgangsgeometrie orthonormal.

---

## §1 · Wo könnte es klemmen — Zahlen und Rechenweg

### 1.1 Der Prädiktorvergleich

**Aufbau.** 122 Zeilen = 3 Modelle × 42 Reaktionen, jede Zeile eine
Modellvorhersage für einen Übergangszustand. **Zielgröße: die Modellstruktur
ist kein Stationärpunkt** — max|F| ≥ 0.15 eV/Å an ihrer eigenen Geometrie,
gemessen auf der Grundzustandsfläche (29 Positive, 24 %). Der Gradient stammt
aus derselben zweistufigen Rechnung wie in §2.3: `STABPerform` für die
Orbitale, dann `EnGrad MORead`. Beide Prädiktoren sind **vor** jeder teuren
Rechnung verfügbar.

```
AUC für die Vorhersage „kein Stationaerpunkt"   (n = 122, Positive 29)

  -λ_min_ext (kontinuierlich)   0.836
  ext_stable (binär)            0.829
  N_FOD      (kontinuierlich)   0.776

je Modell, AUC von -λ_min_ext
  UMA-S    0.842    n = 41,  davon 12 nicht stationär
  UMA-M    0.862    n = 41,  davon  6
  eSEN     0.837    n = 40,  davon 11
```

**Warum die Zielgröße gewechselt hat.** Eine frühere Fassung nahm den RMSD zur
RKS-TS-Struktur als Ziel (225 Zeilen, Schwelle 0.3 Å) und kam auf 0.8496 /
0.7710 / 0.6963. Das ist nicht haltbar: auf der instabilen Seite ist der
RKS-TS kein Stationärpunkt der Fläche, auf der die Reaktion abläuft (§3.0).
„Abweichung vom RKS-TS" ist dort ein Abstand, kein Fehler — und der Test wird
beinahe zirkulär, weil Prädiktor und Zielgröße beide von der restringierten
Lösung handeln. Die Zielgröße oben braucht keinen Vergleichspunkt.

**Was den Wechsel überlebt und was nicht.**

```
                              alt (RMSD zum RKS-TS)   neu (referenzfrei)
  Instabilität schlägt N_FOD        0.850 / 0.696        0.836 / 0.776   ✓
  kontinuierlich schlägt binär      0.850 / 0.771        0.836 / 0.829   ✗
```

Die Kernaussage bleibt fast unverändert. Der Vorsprung des kontinuierlichen
Eigenwerts über sein binäres Abbild verschwindet dagegen — 0.007 statt 0.08.
Er war ein Artefakt der referenzbasierten Zielgröße. **Für das Kapitel heißt
das: „instabil ja/nein" genügt**, der Zahlenwert muss nicht berichtet werden.

**Die Trennung, direkt statt über AUC:**

```
                    Strukturen   Stationärpunkte   Median max|F|
  RKS stabil            78        74     95 %          0.067
  RKS instabil          44        19     43 %          0.163
```

Das reproduziert die 96 % / 46 % aus §2.2 auf einem zweiten Weg, mit anderer
Stichprobe und ohne Hesse-Matrix.

> **Zur Stichprobe.** Die 42 Reaktionen sind nicht der Benchmark-Satz. Die
> stabile Seite stammt aus der FOD-geschichteten Auswahl von
> `pipeline/force_error_at_ts.py` (die obersten 26 nach N_FOD, zehn über die
> Rangliste verteilt, die untersten zehn), die instabile sind die 19. MACE und
> MACE+delta sind nicht enthalten, weil für sie keine Gradienten an den
> Modellgeometrien vorliegen; in der alten Fassung fielen sie mit AUC 0.70
> beziehungsweise 0.47 bei N_FOD aus dem Muster.

**Erzeugt von:** `pipeline/predictor_reffree.py`; die alte, referenzbasierte
Fassung bleibt als `pipeline/sep_analysis.py` bestehen.

### 1.2 Der Einwand „das misst doch dasselbe"

**Aufbau.** Nur die Strukturen, die N_FOD ohnehin als verdächtig meldet, dort
noch einmal nach Stabilität aufgeteilt. Wenn beide Deskriptoren dasselbe
messen, darf innerhalb dieser Gruppe nichts mehr zu trennen sein. Zielgröße
wie in §1.1: ist die Modellstruktur ein Stationärpunkt.

```
N_FOD > 0.5              Strukturen   Stationärpunkte
  RKS stabil                 27          24     89 %
  RKS instabil               41          18     44 %
```

**Sie trennt weiter.** Innerhalb der Gruppe, die N_FOD gleich behandelt, steht
89 gegen 44 Prozent. Die Stabilitätsanalyse misst also nicht dasselbe genauer,
sondern etwas anderes.

**Zwei Fälle zum Anfassen:** rxn7945 und rxn7937 haben hohes N_FOD und
trotzdem stabile Übergangszustände. Bei ihnen sitzt die Brechung am Produkt,
nicht am Barrierenkamm — N_FOD sieht sie, kann aber nicht sagen, *wo* sie
sitzt, und genau darauf kommt es an.

> **Was hier vorher stand.** Eine frühere Fassung teilte dieselbe Gruppe nach
> dem RMSD zur RKS-TS-Struktur (stabil 0.0383 Å, instabil 0.1996 Å, Faktor 5).
> Die Trennung war dieselbe, die Größe aber nicht haltbar: rechts ist der
> RKS-TS kein Sattelpunkt, der Abstand zu ihm also kein Fehler (§3.0). Die
> Zahlen stehen weiter in `stability_vs_fod_separation.txt`, Schritte 4 und 5,
> und sind als *Abstand* zu lesen, nicht als Abweichung.

**Erzeugt von:** `pipeline/predictor_reffree.py`.

### 1.3 Die referenzfreie Paartabelle

**Warum referenzfrei.** Die naheliegende Zeile „Modelle gegen RKS-TS" ist in
der rechten Spalte nicht auswertbar: dort ist der RKS-TS nicht die Antwort.
Eine frühere Fassung dieser Tabelle hat stattdessen gegen die BS-TS-Opt-Struktur
gemessen — das ist zirkulär, weil die BS-TS-Opt-Struktur genau das ist, was zur
Debatte steht. Beide Zeilen sind deshalb aus §1 entfernt.

Was bleibt, kommt ohne jeden Vergleichspunkt aus: Modell gegen Modell.

```
Modell gegen Modell, größte paarweise Differenz der drei

Gruppe                 n    TS-Energiestreuung [meV]    TS-Geometrie [Å]
                              median            max     median     max
  RKS stabil          26        0.3          513.5     0.0045  0.4585
  RKS instabil        19       13.9         4435.6     0.0509  2.5325

Edukt-Geometrie, zur Kontrolle              median     max
  RKS stabil          26                    0.0002  0.0003
  RKS instabil        19                    0.0002  0.0119
```

**Das Edukt kürzt sich heraus.** Die drei Modelle stimmen beim relaxierten
Edukt in *beiden* Gruppen auf 0.0002 Å überein. Die Streuung der TS-Energie
**ist** damit die Streuung der Vorwärtsbarriere — das Edukt fällt aus der
Differenz. Ohne diese Kontrollzeile wäre die Barrierenstreuung nicht
interpretierbar.

```
Wie viele Reaktionen überschreiten eine gegebene Uneinigkeit
  > 10 meV     stabil  2/26     instabil 10/19
  > 50 meV             1/26              5/19
  > 250 meV            1/26              4/19
  > 1 eV               0/26              2/19
```

Die einzelnen Reaktionen, nach Uneinigkeit sortiert:

```
rxn        N_FOD   TS-Energie [meV]   TS-Geometrie [Å]
rxn8837    0.798           4435.6           1.9683
rxn4113    0.960           1065.0           0.7300
rxn8885    0.923            345.6           1.4111
rxn0894    0.716            250.8           2.5325
rxn5691    0.778             62.7           0.2111
rxn7060    0.788             46.2           0.0469
rxn1283    0.769             34.6           0.0639
rxn8832    1.000             16.0           0.0509
rxn7949    1.146             15.3           0.0639
rxn8827    0.760             13.9           0.0620
rxn0346    0.847              7.0           0.0446
rxn5690    0.433              5.5           0.0258
rxn4522    0.731              3.5           0.0163
rxn1147    0.725              2.8           0.0319
rxn4518    0.833              2.7           0.0783
rxn3107    0.801              2.6           0.0410
rxn6196    0.869              1.5           0.0120
rxn1320    0.968              1.4           0.0161
rxn7957    0.684              0.3           0.0192
```

**Bemerkenswert und erwähnenswert:** die Reaktion mit dem *höchsten* N_FOD
(rxn7949, 1.146) liegt bei der Uneinigkeit im Mittelfeld, die Reaktion mit der
größten Uneinigkeit (rxn8837) bei N_FOD 0.798. Die beiden Größen sind nicht
dasselbe — was §1.2 quantitativ zeigt, ist hier zeilenweise sichtbar.

### 1.4 Die Validierung der linken Spalte

**Aufbau.** Zwei Maße nebeneinander, weil ein einzelnes die Fehlerarten
vermischt: die reaktive Koordinate (Schwelle 0.10 Å) beantwortet „derselbe
Übergangszustand?", der All-Atom-RMSD (0.30 Å) beantwortet „auch dieselbe
Konformation?".

```
RKS stabil, 26 Reaktionen, 130 Zeilen — gegen den RKS-TS
Modell        korrekt  RK ok, Konf.  RK daneben  falsch   RC med   RMSD med
  UMA-S            24            0           2       0    0.0054    0.0051
  UMA-M            24            0           2       0    0.0050    0.0050
  eSEN             24            0           1       1    0.0049    0.0054
  MACE             21            0           4       1    0.0195    0.0172
  MACE+delta       20            0           4       2    0.0376    0.0613
```

Das ist die Zahl „0.005 Å" aus Teil A. 24 von 26 exakt richtig, bei allen drei
starken Modellen. Diese Zeile trägt die Aussage, dass Modelle, Aufbau und
Auswertung funktionieren — und sie steht bewusst **ohne** Gegenstück rechts.

> **Vorbehalt.** Dieselbe Tabelle existiert für die BS-Gruppe, misst dort aber
> gegen die BS-TS-Opt-Struktur und ist damit zirkulär. Sie taugt zur Trennung von
> Konformations- und Chemiefehler, nicht als Leistungsurteil, und wird in §1
> nicht verwendet. Bei rxn1147 und rxn7957 zählt sie alle Modelle als Versager,
> obwohl bei rxn7957 die Modelle recht haben (§3c).

**Abbildung:** `model_spread_linear.png`.

---

## §2 · Wie die Modelle scheitern — Zahlen und Rechenweg

Drei Messungen, die zusammen eine Ursachenkette ergeben. Alle drei entstanden
erst, nachdem die naheliegende Erklärung widerlegt war — die Reihenfolge im
Kapitel ist deshalb auch die Reihenfolge der Untersuchung.

### 2.1 (a) Die widerlegte Labelhypothese

**Meine Vermutung war:** die Modelle sind auf Transition1x-Labels trainiert,
die restringiert gerechnet wurden; sie hätten also die falsche Fläche gelernt.

**Der Einwand dagegen war richtig:** in OMol25 wurde mit UKS neu gerechnet, und
eine UKS-Rechnung kollabiert dort zur RKS-Lösung, wo diese stabil ist. Es gibt
für das Modell nur *eine* Fläche zu lernen — die Grundzustandsfläche.

**Der Test.** An jeder Modellgeometrie werden drei Barrieren verglichen: die
des Modells, eine RKS-Barriere und eine BS-Barriere, alle vom selben Edukt aus.
Welche der beiden DFT-Varianten die Modellbarriere trifft, sagt, welcher Fläche
das Modell folgt.

```
follows RKS                                        1
follows BS                                        39
keine Unterscheidung möglich (Blätter < 50 meV)   16

eingeschränkt auf die 33 Fälle, in denen sich die beiden
Hypothesen um mehr als 300 meV unterscheiden:   RKS 1, BS 32
```

Die typische Abweichung |Modell − BS| liegt bei 0.01 bis 0.04 eV, während
|Modell − RKS| bis 3.05 eV erreicht (rxn8885/eSEN). Beispielzeilen:

```
rxn        Modell   Modell     RKS       BS   |m-RKS|  |m-BS|
rxn4518    UMA-S      3.91     6.78     3.93     2.87    0.02
rxn8885    eSEN       3.26     6.31     3.29     3.05    0.03
rxn1283    UMA-M      4.82     6.73     4.86     1.91    0.04
rxn4522    UMA-M      3.85     6.04     3.85     2.19    0.00
```

**Kontrolle, ohne die der Test nichts wert wäre:** die Modell-Edukte liegen im
Median 0.0005 Å (max 0.0207) von den Referenz-Edukten entfernt. Der Nullpunkt
der Barriere ist praktisch dieselbe Struktur und bevorzugt keine der beiden
Hypothesen.

**Der eine Gegenfall** ist rxn0894/eSEN und ist einer: |m−RKS| 1.30 gegen
|m−BS| 2.69, bei einer Modellbarriere von 7.12 eV — eine Geometrie, an der eSEN
ohnehin um 0.8 Å danebenliegt.

**Erzeugt von:** `pipeline/which_sheet_did_models_learn.py` →
`which_sheet.txt`.

> **Diese Widerlegung gehört ins Kapitel, nicht in eine Fußnote.** Sie ist die
> Erklärung, die der Leser sonst selbst einsetzt, und sie ist falsch.

### 2.2 (b) Die Form des Versagens

**Aufbau.** Vollständiger Frequenz-Sweep: für jede vorhandene Struktur die
dreistufige ORCA-Kette. 106 Rechnungen, keine Fehler.

```
                      geprüft  stationär  +1 imag.  = Sattelpunkt  Anteil
RKS-TS
  RKS instabil            19        0         —          —            0 %
  RKS stabil              26       25         —          —           96 %
BS-TS-Opt-Struktur
  RKS instabil            18       16        16         16           89 %
Modelle, RKS instabil
  UMA-M                   19       13        17         12           63 %
  eSEN                    19       10        17          8           42 %
  UMA-S                   19        7        16          6           32 %
Modelle, RKS stabil       26        —         —         25           96 %
```

**Die Diagonale ist die Aussage.** Bei jedem Modell tragen 16 oder 17 von 19
Vorhersagen genau eine imaginäre Mode — die Krümmungssignatur eines
Übergangszustands sitzt fast immer. Stationär sind nur 7 bis 13. **Der Engpass
ist ausschließlich Stufe 1.**

> **Die 89 % sind kein Qualitätsurteil.** Die Strukturen wurden darauf
> optimiert, Sattelpunkte zu sein; die Zeile ist eine Kontrolle der
> Optimierung. Sie darf nicht neben die Modellzahlen gestellt werden, als wäre
> es dieselbe Frage.
>
> **Zahlenstand.** `model_saddle_stats.txt` (12.08.) führt diese Zeile noch als
> 16 von 16 = 100 %, weil es vor den letzten beiden Hesse-Rechnungen entstand.
> Mit rxn1283 (zwei imaginäre Moden) und rxn4522 (Gradient 1.199, gar nicht
> stationär) sind es 18 geprüfte Strukturen und 16 bestandene. Maßgeblich ist
> `saddle_matrix.txt` (13.08.), das beide Fälle führt. Die Datei ist neu zu
> erzeugen.

**Für die 96 % links wurde keine Hesse am RKS-TS gerechnet** — Stufe 2 ist dort
unbekannt *by construction*. Die 96 % sind der Anteil stationärer Punkte, nicht
bestätigter Sattelpunkte. Das ist im Kapitel sauber zu benennen.

**Erzeugt von:** `pipeline/job_orca_freq_sweep.sh` (Array-Job über
`freq_tasks.txt`), ausgewertet mit `pipeline/model_saddle_stats.py` und
`pipeline/status_matrix.py` → `model_saddle_stats.txt`, `status_matrix.md`.

### 2.3 (c) Die Ursache: der Kraftfehler

**Aufbau.** Beide Zahlen lagen längst auf der Platte und waren nie
nebeneinandergelegt worden. Weil die Aussage des Abschnitts vollständig an
diesem Vergleich hängt, steht der Rechenweg hier ausgeschrieben.

*Datenquellen.* Auf der Modellseite `<modeldir>/<rxn>/transition_state.xyz`,
geschrieben vom ASE-NEB an seinem Climbing Image:

```
Properties=species:S:1:pos:R:3:forces:R:3  charge=0 spin=1
energy=-8793.900387278647  free_energy=...  stress=...
N   2.91270324  0.59558217 -0.85689469   -0.08230502  0.05081359 -0.23120899
    └────── Position [Å] ──────┘         └────── Kraft [eV/Å] ──────┘
```

Spalten 4 bis 6 sind die Kräfte des Rechners, **nicht** die projizierte
NEB-Kraft — erkennbar daran, dass `energy`, `free_energy` und `stress` im
selben Kopf stehen, also ein gewöhnlicher ASE-Rechner-Dump vorliegt. Auf der
DFT-Seite der `CARTESIAN GRADIENT` aus `orca_freq/<rxn>_<Modell>/engrad.out`,
Stufe 1b des Sweeps.

*Niveau und Fläche.* Stufe 1a erzeugt mit `STABPerform` und
`STABRestartUHFifUnstable` die Orbitale des Grundzustands; Stufe 1b rechnet den
Gradienten mit `EnGrad MORead` aus eben diesen Orbitalen. Der Gradient liegt
damit auf der gebrochenen Fläche, wo eine gebrochene Lösung existiert. Kontrolle,
dass es greift: bei rxn8837/UMA-M steht in Stufe 1a ⟨S²⟩ = 1.007.

*Umrechnung.* ORCA druckt dE/dx in Eh/Bohr; eine Kraft ist das Negative davon,
Faktor 51.42208 nach eV/Å. Die Modellkräfte liegen bereits in eV/Å vor.

*Definitionen.* ΔF = F_Modell − F_DFT, komponentenweise über alle 3N
Komponenten. MAE ist der Mittelwert von |ΔF|, „max Komp." das größte |ΔF|.
**|F| ist die größte Betragskomponente des jeweiligen Kraftvektors, nicht die
Norm.** Berichtet werden Mediane über die Gruppe.

```
                     MAE   max Komp.  |F| Modell  |F| DFT     (eV/Å, Mediane)
  RKS stabil    (78) 0.013     0.058      0.032     0.067
  RKS instabil  (44) 0.031     0.142      0.032     0.163

je Modell, MAE stabil → instabil
  UMA-S    0.013 → 0.036    2.7×
  UMA-M    0.011 → 0.023    2.1×
  eSEN     0.017 → 0.038    2.3×
```

**Die entscheidende Spalte ist die letzte.** Das Modell meldet in beiden
Gruppen dieselbe Restkraft, 0.032 eV/Å. Tatsächlich wirken links 0.067 und
rechts 0.163.

**Zwei Validierungen, ohne die der Vergleich nur plausibel wäre.**

*Es ist derselbe Punkt.* `orca_freq/<rxn>_<Modell>/start.xyz` ist eine Kopie von
`transition_state.xyz`. Über 44 Paare geprüft, größte Abweichung **exakt
0.00 Å**. Das ist keine Erkenntnis, sondern eine Installationskontrolle — sie
hätte gefangen, wenn die Aufgabenliste die Referenzstruktur statt der
Modellstruktur genommen oder die Atomreihenfolge getauscht hätte.

*Die gespeicherten Kräfte gehören zu den gespeicherten Koordinaten.* Geometrie
und Kräfte stammen aus demselben ASE-Snapshot, aber das stand nur zu vermuten.
Gegentest: dasselbe Modell noch einmal als reiner Einzelpunkt auf dieselbe
Geometrie.

```
UMA-S   median max|ΔF| 3.28e-06   größte 1.07e-05 eV/Å
UMA-M                  1.53e-06            1.94e-05
eSEN                   2.10e-06            1.32e-05
```

57 von 57 ohne Abweichung, vier Größenordnungen unter dem berichteten Fehler.
Erzeugt von `pipeline/model_sp_recheck.py`.

**Vier Vorbehalte gehören mitgedruckt.**

*Basissatz.* Die Modelle sind gegen ωB97M-V/**def2-TZVPD** trainiert (OMol25),
gerechnet wird def2-TZVP. Ein Teil der Differenz ist Basissatz und nicht
Modellfehler. Für die Kernaussage trägt das, weil sie von der Größenordnung
lebt und nicht von der dritten Stelle.

*n = 44 von 57.* Dreizehn Modellgeometrien haben keinen DFT-Gradienten:
rxn7949, rxn5691 und rxn4522 je alle drei Modelle, rxn1147 (UMA-M, eSEN),
rxn7957 (UMA-S, eSEN). Auch nicht in `freq_at_model` — dort liegen nur
Hesse-Matrizen. rxn7949 fehlt damit ausgerechnet in der Tabelle, obwohl es
andernorts im Kapitel diskutiert wird.

*Wie die Kontrollgruppe vervollständigt wurde.* Eine erste Fassung dieser
Tabelle hatte links nur 26 Struktur-Modell-Paare — genau ein Modell je
Reaktion, verteilt als UMA-M 18×, UMA-S 6×, eSEN 2×. Der Gruppenvergleich war
davon unberührt, die modellweise Zeile stand für eSEN aber auf zwei
Strukturen. Nachgerechnet wurden die 52 fehlenden Paare mit demselben
zweistufigen Rezept wie die vorhandenen, kopiert aus
`orca_freq/rxn0101_UMA-M`:

```
bs_sp.inp    ! UKS wB97M-V def2-TZVP def2/J RIJCOSX TightSCF
             %scf STABPerform true / STABRestartUHFifUnstable true
                  MaxIter 300 end
             → bs_sp.gbw, kopiert nach bs_start.gbw

engrad.inp   ! UKS wB97M-V def2-TZVP def2/J RIJCOSX TightSCF EnGrad MORead
             %moinp "bs_start.gbw"
             → CARTESIAN GRADIENT
```

Beide Läufe an derselben, unveränderten Modellgeometrie
(`<modeldir>/<rxn>/transition_state.xyz`, kopiert nach `start.xyz`). Die
Zweiteilung ist erzwungen: `STABPerform` erlaubt ORCA nur mit
`RunTyp SinglePoint`, kann also im selben Lauf keinen Gradienten liefern.
`MORead` stellt sicher, dass der Gradient zu der Lösung gehört, die die
Stabilitätsanalyse gefunden hat, und nicht zu einer, auf die ein frisches SCF
zufällig konvergiert.

Vorhandene Gradienten wurden nicht überschrieben; der Lauf bricht ab, wenn ein
Einzelpunkt nicht normal endet, keine Orbitale hinterlässt oder die Ausgabe
keinen `CARTESIAN GRADIENT` enthält. 52 von 52 ohne Abbruch.

*Probe an der Kontrollgruppe.* Jede der 52 Rechnungen protokolliert ihr ⟨S²⟩
aus Stufe 1a. Über die 52 Werte: Median 0.0000, Minimum 0.0000, Maximum
0.3118; zwei Werte über 0.05 (0.070 und 0.312), die übrigen exakt null. An den
Modellgeometrien der stabilen Reaktionen findet die Stabilitätsanalyse also
nichts zu brechen — die Einordnung stammte bis dahin allein aus der Analyse am
RKS-TS.

**Erzeugt von:** `pipeline/job_orca_grad_gap.sh` über `grad_gap_tasks.txt`.
Vorherige Fassung der Tabelle als `force_error_at_ts_vor_luecke.txt`.

*|F| ist die größte Komponente*, nicht der Betrag. 0.032 und 0.163 sind
Maximalkomponenten.

*Vier der 19 Modell-NEBs sind nicht konvergiert* — siehe den folgenden
Abschnitt. Für diese gilt der Satz „ein NEB hält an, wenn seine Kraft klein
ist" nicht.

> Ein NEB hält an, wenn **seine** Kraft klein ist. Ist die Kraft um 0.14 eV/Å
> falsch, hält er an einem Punkt an, an dem die echte Kraft noch wirkt — und
> meldet dabei Konvergenz.

**Zwei Fehlerarten in derselben Spalte**, und die Unterscheidung fällt hier
gratis heraus:

```
rxn7060  eSEN   glaubt 0.028, hat 1.126     Faktor 40 — Genauigkeitsproblem
rxn8837  UMA-M  glaubt 0.752, hat 0.757     Kraft richtig, trotzdem gestoppt
                                            — Konvergenzabbruch
```

Der zweite Fall ist kein Modellfehler. Er gehört ins Kapitel, weil er zeigt,
dass „Modell liegt daneben" mindestens zwei verschiedene Dinge bedeuten kann.

**Erzeugt von:** `pipeline/force_error_at_ts.py` → `force_error_at_ts.txt`.

### 2.3b Vier Modell-NEBs sind gar nicht konvergiert

Beim Nachvollziehen des Rechenwegs oben fiel auf, dass das NEB-Log von
rxn8837/UMA-M mit drei identischen Zeilen bei fmax 0.2704 endet. Über alle 57
Läufe nachgezählt, gegen das Kriterium `--cineb-fmax 0.05` bei `--steps 500`:

```
konvergiert       15 von 19   je Modell, identisch, fmax 0.043 – 0.050
nicht konvergiert  4 von 19

   rxn7949   0.071 – 0.087    nach 476 – 490 Schritten
   rxn8885   0.136 – 0.238    nach 618 – 637
   rxn8837   0.165 – 0.319    nach 352 – 479
   rxn0894   0.119 – 0.220    nach 197 – 502
```

Alle geprüften einfachen Reaktionen konvergieren, alle drei Modelle, 0.039
bis 0.050.

**Und die vier sind genau die problematischen Fälle:**

```
rxn8885, rxn8837, rxn0894   kein Modell findet einen gültigen Sattelpunkt
rxn7949                     Modelle finden einen, Stufe 3 kippt ihn
                            (−100 bis −109 cm⁻¹, Bindungsrate ~0.01)
```

Vier von vier. Ob der Modell-NEB konvergiert ist, sagt auf diesem Satz alles
voraus, was danach kommt — und es steht im Log des Modells, kostet also nichts.

**Das teilt §2 in zwei Aussagen statt einer.** Für die 15 konvergierten gilt
der Mechanismus aus §2.3: der Kraftfehler hat dieselbe Größe wie das
Konvergenzkriterium, also trägt das Kriterium keine Information mehr. Für die
vier anderen gilt etwas Einfacheres — **das Modell hat gemeldet, dass es
gescheitert ist.** Ihre Vorhersage ist nicht der Punkt, an dem die Kraft klein
wurde, sondern der Punkt, an dem der Optimierer aufgab.

Dazu passt die Zeile aus §2.3, die vorher wie eine Kuriosität aussah:
rxn8837/UMA-M glaubt 0.752 und hat 0.757. Das Modell wusste, dass dort noch
Kraft wirkt.

> **Praktisch heißt das:** vor jeder Auswertung einer Modellvorhersage gehört
> ein Blick in `neb.log`. Ein nicht konvergierter Lauf ist keine Vorhersage,
> und ihn wie eine zu behandeln erzeugt genau die Sorte Befund, die dieses
> Kapitel sonst den Modellen anlastet.

### 2.4 (d) Die Einordnung

0.031 eV/Å ist für diese Modellklasse kein schlechter Kraftfehler. Das Problem
ist die Koinzidenz zweier Skalen:

```
Kraftfehler des Modells        0.031 eV/Å
Konvergenzkriterium eines NEB  0.03 – 0.05 eV/Å
```

Ein Kriterium trägt keine Information, wenn der Fehler dieselbe Größe hat. Für
eine Molekulardynamik oder eine Minimumssuche wäre derselbe Fehler folgenlos —
dort wird nicht auf das Verschwinden der Kraft hin abgebrochen.

Das ist die faire Formulierung: eine Aussage über den **Aufgabentyp**, nicht
über die Modelle.

### 2.5 Die praktische Konsequenz: Triage

**Aufbau.** Zehn TS-Optimierungen von Modellgeometrien aus, neun davon von
UMA-M. Sortiert nach dem DFT-Gradienten an der Startgeometrie.

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

> **Korrektur an einer früheren Fassung.** Hier stand „7 von 7". Gegen Stufe 3
> geprüft sind es sechs: `tsopt_rxn7949_UMA-M` startet bei 0.051 eV/Å, liegt
> also klar im unteren Bereich, und landet auf einem Punkt mit −69 cm⁻¹ und
> einer Bindungsrate von 0.008 — einem Torsionssattel im **Eduktbecken**, beide
> reaktiven Bindungen auf Eduktwerten (C3-C5 2.408 gegen 2.538 im Edukt,
> C4-C5 1.477 gegen 1.442). Ein sauber konvergierter Stationärpunkt mit genau
> einer imaginären Mode, 0.53 eV unter dem BS-TS-Opt-Übergangszustand, und trotzdem
> kein Übergangszustand dieser Reaktion.
>
> Passend dazu: rxn7949 ist eine der vier Reaktionen aus §2.3b, deren
> Modell-NEB nicht konvergiert ist. Der Startpunkt lag bereits im Eduktbecken,
> die DFT-Optimierung hat das nicht korrigiert, sondern bestätigt.

Unterhalb von ~0.25 eV/Å ist die Modellvorhersage **nur ungenau** — sie liegt
im Einzugsbereich des richtigen Sattelpunkts, und die DFT-Optimierung korrigiert
genau den Kraftfehler aus §2.3. Oberhalb ist sie **am falschen Ort**.

Der DFT-Gradient an der Modellgeometrie ist damit ein Triage-Kriterium: ein
Einzelpunkt, Minuten, sagt vorher, ob eine Nachoptimierung sich lohnt.

**Bei zweien bleibt die Optimierung fast stehen**, rxn7949 und rxn7957, 0.049
bzw. 0.010 Å vom Start. Die Modellgeometrie war dort bereits ein
DFT-Stationärpunkt — bei rxn7957 der korrekte Übergangszustand, bei rxn7949 ein
Torsionssattel im Eduktbecken. Derselbe Befund, zwei entgegengesetzte
Bedeutungen: „das Modell war schon richtig" und „das Modell war schon falsch,
und die Verfeinerung ändert daran nichts". Ohne Stufe 3 sind die beiden Fälle
nicht zu unterscheiden.

> **Vorbehalt, der mitgedruckt gehört.** Zehn Datenpunkte, nicht zufällig
> gewählt, und die Schwelle 0.25 eV/Å ist **abgelesen und nicht bestimmt**. Für
> die neun fehlenden Reaktionen — darunter alle fünf, bei denen die BS-TS-Opt
> vom RKS-TS aus versagt hat — ist dieser Lauf nie gemacht worden.

**Erzeugt von:** `pipeline/job_bs_tsopt_umam_missing.sh` für die Läufe,
`pipeline/force_error_at_ts.py` für die Sortiergröße (der DFT-Gradient an der
Startgeometrie), Bewertung mit `pipeline/verdict_final.py`. Die Schwelle
0.25 eV/Å liegt in der Lücke zwischen der obersten Zeile der ersten Gruppe
(0.25) und der untersten der zweiten (0.33) — sie ist **abgelesen, nicht
angepasst**, und die Vorhersage „unten gelingt es, oben nicht" wurde vor dem
Start der zweiten Gruppe notiert.

**Nebenbefund, der in §4 gebraucht wird:** keiner der zehn Läufe hat einen
Sattelpunkt gefunden, der nicht schon bekannt war. Eine elfte Suche, von zehn
anderen Startpunkten aus, hat nichts Neues gefunden. Gegen den stehenden
Vorbehalt — eine lokale Suche findet nur, was unter ihrem Startpunkt liegt — ist
das der stärkste verfügbare Hinweis.

**Zweiter Nebenbefund:** rxn0894 liefert wie rxn8885 einen tieferen
Stationärpunkt, der ein **Minimum** ist — 180 bzw. 425 meV unter dem BS-TS-Opt-
Sattelpunkt, beide ohne imaginäre Mode. Die Falle aus der Dreistufenregel ist
kein Einzelfall, sondern wiederkehrend in der diradikalischen Region.

---

## §3 · Auch die Rechnungen brechen — Zahlen und Rechenweg

### 3.0 Der RKS-TS ist kein Stationärpunkt

**Aufbau.** Für jede Reaktion ein ORCA-Gradient am RKS-TS, gemessen auf der
**Grundzustandsfläche** — also RKS, wo die restringierte Lösung extern stabil
ist, und BS, wo nicht. Die Zahl beantwortet damit genau die richtige Frage: ist
der RKS-TS ein Stationärpunkt der Fläche, auf der die Reaktion abläuft?

```
Gradientennorm am RKS-TS [eV/Å]
  RKS stabil      n=26   median 0.0434   min 0.0135   max 0.1746
  RKS instabil    n=19   median 1.6974   min 0.1620   max 2.9493
```

**0 von 19 unterschreiten 0.15 eV/Å.** Selbst der beste Fall (rxn5690, 0.162)
liegt über der Schwelle, der schlechteste (rxn4518, 2.95) um Faktor 20 darüber.

Zum Vergleich in derselben Größe: die Modellvorhersagen liegen in der
MR-Gruppe bei Median 0.1375 eV/Å. **Die Modelle sind dort zwölfmal näher an
einem gültigen Punkt als der RKS-TS.**

Und weil dieselbe Messung die Größe liefert, die Teil A in §4 braucht:

```
Modellgeometrien, Abstand von der Stationarität [eV/Å]
Klasse        n   median     mean       Q3      max   >0.3
  einfach    78   0.0661   0.0822   0.0958   0.5909      1
  MR         56   0.1375   0.2224   0.2002   1.1231     10

je Modell   einfach → MR    Faktor
  UMA-S      0.0653 → 0.1633   2.5
  UMA-M      0.0563 → 0.1125   2.0
  eSEN       0.0755 → 0.1404   1.9
```

**Faktor 2, nicht 25.** Der große Faktor früherer Auswertungen entstand
dadurch, dass gegen den BS-TS-Opt-Sattelpunkt gemessen wurde. Referenzfrei gemessen
finden die Modelle weiterhin fast-stationäre Punkte; was zunimmt, ist die
Uneinigkeit darüber, welcher es ist.

**Quellen:** `saddle_matrix.txt` (Spalte RKS-TS), `model_saddle_stats.txt`
Abschnitt 1b, `bs_ts_working.md` Abschnitt „Referenzfreie Befunde".

> **Reihenfolgefehler, der hier einmal drinsteckte.** `saddle_matrix.py` prüfte
> ursprünglich die Hesse vor dem Gradienten und gab deshalb für den RKS-TS
> „nie geprüft" aus, obwohl der Gradient bei 2.39 eV/Å lag. Die Spalte liest
> jetzt `n.stat.` — Reihenfolge Gradient vor Hesse.

### 3.1 Die Endpunktprüfung — die Prämisse hält

**Warum überhaupt.** Die Stabilitätsanalyse lief bis dahin an 180 Geometrien,
und jede war ein Übergangszustand. Die Endpunkte hatte nie jemand angesehen,
obwohl sie den Pfad definieren: ein NEB interpoliert zwischen ihnen, und eine
Barriere ist E(TS) − E(Edukt).

**Aufbau.** Ein Einzelpunkt mit `STABPerform` an jedem relaxierten Edukt und
Produkt **aller 45** Reaktionen — nicht nur der 19, weil die Einteilung allein
am Übergangszustand gemacht wurde. 90 Rechnungen, 45 komplett, 0 fehlend.

```
alle 45 Edukte           ⟨S²⟩ = 0.000   ausnahmslos
 5 von 45 Produkten      spingebrochen

rxn8832   Produkt 0.419   ΔE_BS  -84.0 meV     (TS: -428.0)
rxn8837   Produkt 0.473          -69.7         (TS: -293.9)
rxn8827   Produkt 0.425          -35.4         (TS:  -27.5)
rxn7945   Produkt 0.320          -24.2         (TS: stabil)
rxn7937   Produkt 0.103           -2.0         (TS: stabil)
die übrigen 85 Endpunkte         exakt 0.0
```

**Die 85 Nullen sind die Kontrolle:** das Verfahren erzeugt keine falschen
Brechungen.

**Was betroffen ist und was nicht** — die Unterscheidung ist leicht
verkehrtherum zu bekommen:

| Größe | Status |
|---|---|
| Vorwärtsbarriere E(TS) − E(Edukt) | **unberührt.** Der Nullpunkt sitzt am Edukt, und jedes Edukt ist auf der richtigen Fläche. Das ist die Zahl, die der Benchmark bewertet. |
| Reaktionsenergie, Rückbarriere | **falsch** um die Tiefe der Brechung am Produkt, 2 bis 84 meV |
| Produktgeometrie | auf der restringierten Fläche optimiert, also kein Minimum der richtigen Fläche — der Referenzpfad endet an einem nicht-stationären Punkt |

Zur Einordnung: bis zu 84 meV gegen bis zu 4435 meV Modelluneinigkeit auf der
TS-Seite. **Zwei Größenordnungen darunter.**

> **Korrektur an mir selbst, die im Skript steht.** Die ursprüngliche
> Auswertung gab aus: *„jede Barriere, die von diesem Punkt aus gemessen wird,
> hat den falschen Nullpunkt"*. Das war vor den Zahlen formuliert und ist
> falsch, sobald nur Produkte brechen. Der Text im Skript ist korrigiert.

**Ein Fall läuft andersherum:** bei rxn8827 ist das Produkt (−35.4 meV) stärker
gebrochen als der Übergangszustand (−27.5). Der Diradikalcharakter wächst dort
entlang der Reaktion, statt am Sattelpunkt zu kulminieren. Der einzige solche
Fall im Satz — und ein Hinweis darauf, dass die Einteilung am TS nicht die
Einteilung der Reaktion ist.

**Erzeugt von:** `pipeline/endpoint_report.py` → `endpoint_report.txt`.

### 3.2 (§3a) Wo der UKS-NEB stehen bleibt

**Aufbau.** BS-NEB-TS in ORCA, `! UKS wB97M-V def2-TZVP NEB-TS`, `%scf
BrokenSym 1,1`, 8 Bilder, Preopt.

**Das Band lässt sich nicht aus dem Log auswerten.** ORCA schreibt die SCFs
seiner Bilder nicht ins Hauptlog; was dort steht, ist PREOPT — die Relaxation
der beiden Endpunkte — und die abschließende TS-Optimierung. Die Zählprobe
zeigt es: rxn4113 hat **100 Banditerationen über 8 Bilder und 36 protokollierte
SCFs**, und zwischen `NEB OPTIMIZATION` und `THE NEB OPTIMIZATION HAS
CONVERGED` steht kein einziger.

**Also nachgemessen.** Jeder Lauf hat seine konvergierten Bildorbitale als
`<base>_im{0..9}.gbw` behalten, die Geometrien liegen als `_MEP_trj.xyz` vor.
Pro Bild ein Einzelpunkt `! UKS wB97M-V def2-TZVP TightSCF MORead` auf diese
Orbitale. Von einer konvergierten Wellenfunktion aus braucht er **einen
SCF-Zyklus** und bleibt auf derselben Lösung — gemessen wird also das Band und
keine neue Rechnung.

Der Job validiert sich selbst: vor der Messung reproduziert er mit demselben
Rezept ein ⟨S²⟩, das aus einer anderen Datei bereits bekannt ist, und bricht
sonst ab.

```
Kontrollen   21 bestanden, 0 abgebrochen, je 1 SCF-Zyklus
   bekannt 0.608973 → gemessen 0.608960
   bekannt 0.973550 → gemessen 0.973550
   bekannt 0.000000 → gemessen 0.000000
```

**Ergebnis: 190 Bilder über die 19 Reaktionen, 55 davon mit ⟨S²⟩ > 0.3.**
Gemessen wurden 21 Bänder; rxn1150 und rxn7945 sind einreferenziert und stehen
unten nur zur Anschauung.

```
rxn        Profil je Bild        Gipfel  ⟨S²⟩ dort   Grad NEB
rxn0894    ..####@#..                 6      1.036      0.022
rxn7949    ..##@##...                 4      1.000        —
rxn8837    .....##@..                 7      1.019      0.011
rxn4113    ..@#####..                 2      0.983      0.007
rxn8832    ...#@####.                 4      0.987      0.018
rxn4522    ...@####..                 3      0.922      0.009
rxn7060    .....-#@..                 7      0.830        —
rxn4518    ..#####@..                 7      0.767        —
rxn7957    ...#@.....                 4      0.647      0.009
rxn1147    ...##@....                 5      0.509      0.016
rxn3107    ......@#..                 6      0.423        —
──────────────────────────────────────────────────────────────
rxn8885    .....=##..                 5      0.186        —
rxn1283    ...####o..                 7      0.000        —
rxn5691    ....#o....                 5      0.000        —
rxn8827    .....##o..                 7      0.000      1.074
rxn0346    ....o.....                 4      0.000      2.553
rxn1150    .....o.....                5      0.000      0.017   (einreferenziert)
rxn1320    ....o.....                 4      0.000      2.062
rxn5690    .....o....                 5     -0.000      0.004
rxn6196    ..o.......                 2     -0.000      0.683
rxn7945    ......-o..                 7     -0.000      0.025   (einreferenziert)

.  <0.05      -  0.05–0.3      #  >0.3      o/@ markieren den Gipfel
```

Vier Bänder brechen, **ohne dass der Gipfel bricht** — rxn8885, rxn1283,
rxn5691, rxn8827. Unter dem gröberen Maß „hält das Band die Brechung" hätten
sie wie Erfolge ausgesehen; rxn8827 hat trotz Brechung ein Ergebnis mit
Gradient 1.074, das 0.019 Å neben dem RKS-TS liegt.

> **Korrektur, und sie betrifft die zentrale Aussage dieses Abschnitts.** Bis
> zum 14.08. stand hier „22 von 22 Bandphasen durchgehend restringiert", gezählt
> aus dem Hauptlog. Das waren die Endpunkte. Vorher stand „9 von 22 Bänder
> halten die Brechung", gezählt ohne Phasentrennung. Beide Zahlen sind ersetzt
> durch die Messung oben.
>
> Zwei Rezepte für diese Messung sind zuvor gescheitert und haben dabei
> plausible Zahlen erzeugt: `NoIter` überspringt die Eigenschaftsauswertung und
> lieferte 220 Nullen bei `FINAL SINGLE POINT ENERGY 0.000000000000`;
> `MaxIter 1` konvergiert nicht und druckt gar kein ⟨S²⟩. Beim ersten Versuch
> gab es keine Kontrolle und die Nullen sahen aus wie eine Bestätigung. Beim
> zweiten gab es eine, und 18 von 22 Tasks brachen nach 19 Sekunden ab.

**Eine geprüfte und verworfene Erklärung.** Naheliegend war: gebrochene Bilder
liegen tiefer, also hat ein gemischtes Profil eine Senke über der gebrochenen
Strecke, und das Maximum wird an deren Rand gedrückt. Auf den Energien geprüft,
hält das nicht:

```
Energieschritt über eine Naht     median 584 meV
Schritt innerhalb eines Blatts    median 468 meV      Verhältnis 1.25
Absenkung gegen die Gerade durch die Nachbarn
                                  median +48 meV, negativ in 7 von 15
```

Kein Sprung, keine systematische Absenkung. Dass der Gipfel im Median ein Bild
neben der Naht sitzt, trägt ebenfalls wenig — die gebrochene Strecke überdeckt
meist die Bildmitte, und der Gipfel liegt dort ohnehin.

**Damit bleibt die Korrelation ohne Mechanismus,** und sie steht als solche im
Kapitel.

**Was daraus über die neun folgt, deren TS-Optimierung bricht.** Sie setzt auf
dem Climbing Image auf. Liegt das auf dem gebrochenen Blatt, bleibt sie dort;
liegt es auf dem restringierten, muss sie erst hinüberfinden — und bei rxn8827
tut sie es nicht.

**Wo Band und TS-Optimierung beide brechen, ist das Ergebnis belastbar.**
rxn8837 hat einen gebrochenen Gipfel (⟨S²⟩ 1.019), die TS-Optimierung danach
bleibt gebrochen (17 Werte, max 1.076), und das Ergebnis liegt 0.003 Å von der
Struktur, die die BS-TS-Optimierung von einem ganz anderen Startpunkt erreicht.
Zwei Suchen, verschiedene Startpunkte, derselbe Punkt.

Die weiteren Übereinstimmungen aus derselben Auswertung:

```
Welcher Seite der unabhängige NEB näher landet (Toleranz 0.05 Å)
  die BS-TS-Opt-Struktur   6   rxn1147 rxn1283 rxn4113 rxn4522 rxn7957 rxn8837
  Modell            3   rxn1320 rxn5691 rxn8827
  keiner            5   rxn0346 rxn0894 rxn3107 rxn6196 rxn8832
```

**Erzeugt von:** `pipeline/job_bs_uks_neb18.sh` →
`bs_uks_neb_results/`, ausgewertet in `bs_neb_check.txt`.

**Laufzeiten** (für die Planung des Nachfolgeexperiments):

```
rxn1320   7 h 26 m        rxn8827   9 h 49 m
rxn8837   9 h 54 m        rxn7949   kein TOTAL RUN TIME → Walltime-Abbruch
```

> **Korrektur.** rxn7949 stand zwischenzeitlich als „keine konvergierte
> Struktur" und damit als Methodenversagen im Text. Das Log hat kein
> `TOTAL RUN TIME` — es ist ein Walltime-Abbruch, kein inhaltliches Scheitern.

### 3.3 Zwei Blätter, nicht zwei Rechnungen

Der begriffliche Kern, auf den §3a und §3b beide zurückgehen. Schematisch in
`two_sheets.png`.

Die SCF-Gleichungen sind nichtlinear und können bei **fester Geometrie**
mehrere selbstkonsistente Lösungen haben: die restringierte (α und β in
denselben Raumorbitalen) und eine oder mehrere spingebrochene. Welche man
bekommt, hängt vom Startraten ab — 19-mal direkt vorgeführt.

Daraus folgt: **„die Potentialfläche" ist keine Funktion der Geometrie allein.**
Der Grundzustand ist die untere Hülle mehrerer Blätter, mit einer Naht dort, wo
sie sich kreuzen.

**Zwei Konsequenzen:**

*Eine Reaktion kann zwei gültige Sattelpunkte haben*, weil jedes Blatt einen
eigenen tragen kann. rxn1147 ist genau das:

```
die BS-TS-Opt-Struktur   ⟨S²⟩ = 0.456   auf dem gebrochenen Blatt
UMA-S-Struktur    ⟨S²⟩ = 0.000   dort ist die restringierte Lösung extern stabil
```

Beide sind echte Sattelpunkte der Grundzustandsfläche, in Gegenden, wo
verschiedene Blätter gewinnen. Welcher zur Reaktion gehört, entscheidet nicht
die Energie, sondern der IRC.

*Wer einen Weg abläuft und den Startraten an jedem Punkt neu herleitet, springt
zwischen den Blättern.* Das ist §3a.

**An der BS-TS-Opt-Struktur vorgeführt:** bei rxn4522 verzeichnet der
PySCF-Lauf ⟨S²⟩ = 0.000, ORCAs Stabilitätsanalyse an derselben Geometrie findet
**0.470**. Die Optimierung ist die letzten Schritte auf der restringierten
Fläche gelaufen, während die gebrochene existierte — und konnte deshalb nicht
konvergieren, weil der Sattelpunkt auf dem anderen Blatt liegt. Endgradient
1.199 eV/Å.

**Einschränkung, die dazugehört:** jenseits des Coulson-Fischer-Punkts
verschmilzt das gebrochene Blatt mit dem restringierten und existiert nicht
mehr. ⟨S²⟩ = 0 an einem relaxierten Edukt ist deshalb **korrekt** und kein
Rechenfehler — alle 45 Edukte sitzen dort (§3.1).

### 3.4 (§3b) TS-Opt hängt vom Startpunkt ab

**Aufbau.** BS-TS-Optimierung, gestartet am RKS-TS, anschließend die
dreistufige Prüfkette. 18 der 19 Reaktionen versucht.

```
Methode              versucht  gültig  niedrigster  Abdeckung
  TS-Opt                   18      13           11       72%
  UMA-M                    19      11            9       58%
  eSEN                     19       7            7       37%
  UMA-S                    19       6            6       32%
  TS-Opt ab UMA-M           9       6            6       67%
  UKS-NEB                  12       8            4       67%
```

**Das Versagen ist subtil, weil die Rechnung korrekt ist.** Sie startet an einer
Geometrie mit im Median 1.70 eV/Å Restkraft und konvergiert sauber gegen den
Sattelpunkt, der von dort aus erreichbar ist. Zwei Fälle machen das explizit:

```
rxn1320   die BS-TS-Opt-Struktur: Modenanteil 0.00 auf den reaktiven Atomen
          C2-H6 bei 3.36 Å — das Wasserstoffatom ist vollständig ab
          alle drei Modelle bestehen alle drei Stufen, bei C2-H6 = 2.60 Å
          → wir sind über den Übergangszustand hinausgelaufen

rxn4518   die BS-TS-Opt-Struktur: Modenanteil 0.03, Rate 0.206 bei 3.269 Å
          fast die gesamte Bewegung sitzt außerhalb der Reaktionskoordinate
          UMA-M besteht (UMA-S und eSEN: je 2 imaginäre Moden)
```

Beide bestehen Stufe 1 und 2 und liegen tiefer als der RKS-TS. **Stufe 3
entscheidet sie gegen uns.**

**Konkurrierende Sattelpunkte, und wie Stufe 3 die Zahl reduziert:**

```
nach Stufe 1+2 (saddle_matrix.txt)        10 von 19 mit verschiedenen Sattelpunkten
nach allen drei Stufen (lowest_saddle)     6 von 19 mit mehr als einem gültigen
                                           1.32 verschiedene Sattelpunkte im Mittel
```

Die vier Reaktionen, die Stufe 3 auflöst, sind der beste Beleg dafür, dass die
Stufe gebraucht wird: ohne sie liest der Satz „die Hälfte der Reaktionen hat
konkurrierende Übergangszustände", mit ihr „ein knappes Drittel".

**Erzeugt von:** `pipeline/saddle_matrix.py` → `saddle_matrix.txt`;
`pipeline/lowest_saddle.py` → `lowest_saddle.txt`.

> **Fehler in `lowest_saddle.py`, der eine Zeit lang die Bilanz geschönt hat.**
> Die Label-Zuordnung fand die BS-TS-Opt-Strukturen bei rxn1147 und rxn7957 nicht und
> ließ sie stillschweigend fallen — ausgerechnet die beiden Reaktionen, in
> denen sie verlieren. Aus 11 von 13 wurde so 11 von 11. Behoben; die Funktion
> `odir()` sucht jetzt in `orca_freq/`, `orca_irc/` und dem
> `<rxn>_ours`-Muster.

> **Vorbehalt zur Spalte „versucht".** Die von-Modell-Optimierungen wurden nur
> auf 10 der 19 Reaktionen gestartet, 9 davon von UMA-M allein. Diese Zeile
> steht nicht auf derselben Grundlage wie die übrigen, und ihre Quote ist nicht
> vergleichbar. Ebenso: 13 der 19 Reaktionen haben nur einen einzigen gültigen
> Sattelpunkt — die meisten „niedrigster"-Treffer sind Gleichstände, keine Siege.

### 3.5 (§3c) Die Dreistufenbewertung und ihre zwei Spiegelfälle

**Warum die Energie überhaupt schiedsrichtern kann.** Sind zwei Strukturen
beide Sattelpunkte erster Ordnung und verbinden sie dasselbe Edukt mit
demselben Produkt, dann läuft die Reaktion über den **tieferen**. Energie ist
ein physikalisches Kriterium, und sie behandelt beide Seiten gleich: sie fragt
nicht, wessen Struktur es ist. Jedes frühere Maß fragte, wie weit das Modell von
der BS-TS-Opt-Struktur entfernt liegt — damit war die BS-TS-Opt-Struktur der Maßstab.

Zwei Bedingungen müssen erfüllt sein, damit eine Energiedifferenz etwas bedeutet:
beide Punkte müssen echte Sattelpunkte sein, und beide müssen dieselben Minima
verbinden. Die zweite ist die Schwachstelle — die dafür gebaute Endpunktprüfung
erwies sich als untauglich (siehe *Verworfen* in `bs_ts_working.md`), die
Modenanalyse ist der verfügbare Ersatz.

**Die Falle.** Ein fast-stationärer Punkt unterhalb des BS-TS-Opt-Sattels hat zwei
Lesarten: ein tieferer Sattelpunkt — dann ist er der relevante — oder ein
Minimum bergab vom Übergangszustand — dann ist er gar kein Kandidat. Die
Frequenz trennt Minimum von Sattelpunkt. Sie trennt **nicht** einen Sattelpunkt
dieser Reaktion von dem einer anderen Bewegung.

**Spiegelfall 1 — rxn1147, die Modelle liegen hinter dem Übergangszustand:**

```
Struktur       Anteil  C1-C2 d/dQ  C1-C2 [Å]  C1-O5 d/dQ  C1-O5 [Å]
BS-TS-Opt       0.601       0.134      3.196       0.943      1.864
UMA-S           0.239       0.053      3.570       0.059      1.497   dE -234 meV
UMA-M           0.217       0.057      3.565       0.072      1.499   dE -231 meV
eSEN            0.230       0.055      3.565       0.068      1.499   dE -232 meV
```

Die Modelle bestanden Stufe 1 (231–234 meV tiefer, Gradienten 0.050–0.077) und
Stufe 2 (je genau eine imaginäre Frequenz). So war es auch notiert. Stufe 3
kehrt es um: die zu knüpfende C1-O5-Bindung liegt bei **1.497 Å**, einer
normalen Einfachbindung, gegen 1.864 Å bei uns — sie ist bereits fertig. Und die
Mode bewegt sie mit 0.06 gegen 0.94 bei BS-TS-Opt, dreizehnmal schwächer. Die
Modellgeometrie sitzt im Produkttal; die 231 meV sind der Abfall von einem
Übergangszustand zu seinem Produkt, kein tieferer Pass.

**Spiegelfall 2 — rxn7957, wir liegen hinter dem Übergangszustand:**

```
Struktur       Anteil  C1-H7 d/dQ  C1-H7 [Å]  C5-H7 d/dQ  C5-H7 [Å]
BS-TS-Opt       0.275       0.544      2.462       0.061      1.120
UMA-S           0.551       0.846      1.887       0.310      1.170   dE -890 meV
UMA-M           0.744       1.012      1.866       0.570      1.190   dE -890 meV
eSEN            0.570       0.871      1.884       0.339      1.173   dE -890 meV
```

Dieselbe Prüfung, umgekehrtes Ergebnis. Die BS-TS-Opt-Struktur hat das wandernde
Wasserstoffatom bei 1.120 Å von C5 — eine fertige C-H-Bindung — und bei 2.462 Å
von C1. Der Transfer ist vorbei. Die Modelle liegen mitten drin, mit
Modenanteilen bis 0.74 gegen 0.275 bei BS-TS-Opt, und 890 meV tiefer. **Die Modelle
haben recht.**

> **Diese beiden Fälle sind der Beleg, dass die Prüfung nicht zu den eigenen
> Gunsten kalibriert ist.** Sie gehören genau deshalb ins Kapitel: je ein Urteil
> in jede Richtung, aus demselben Kriterium.

**Der IRC als unabhängige Instanz.** Vier ORCA-IRC-Läufe (`%irc`, `InitHess
read`, `Hess_Filename`, `Direction both`, `Follow_CoordType cartesian`). Bei
rxn1147 entscheidet er ohne dieses Urteil — die BS-TS-Opt-Struktur liegt 0.20 Å neben
dem IRC-Pfad, die Mode bewegt C1-H7. Bei rxn7957 bleibt das Bild
widersprüchlich.

### 3.6 Der unabhängig bestätigte Sattelpunkt

**Definition** (Teil A, §3c) und ihre Anwendung:

| Reaktion | zwei Wege | NEB-Seite brach in der TS-Opt | Abstand |
|---|---|---|---|
| rxn8837 | BS-TS-Opt ab RKS-TS und NEB-TS | 19 Werte, max 1.039 | 0.003 Å |
| rxn7957 | dieselbe Paarung | 10 Werte, max 0.727 | 0.019 Å |
| rxn1147 | dieselbe Paarung | 11 Werte, max 0.509 | 0.022 Å |

Drei Reaktionen, in denen zwei Suchen von verschiedenen Startpunkten auf
denselben Punkt laufen. Die dritte Spalte ist notwendig: eine NEB-Seite, die
durchgehend ⟨S²⟩ = 0 zeigt, ist keine zweite Rechnung auf der richtigen Fläche,
sondern eine RKS-Rechnung.

> **Wie unabhängig sind die beiden Wege wirklich?** Nach der Phasentrennung
> (§3.2) weniger, als die erste Fassung dieses Abschnitts behauptet hat. Beide
> Wege enden in einer TS-Optimierung auf der gebrochenen Fläche; unabhängig sind
> sie im **Startpunkt** — RKS-TS gegen Climbing Image eines Bandes — nicht im
> Verfahren. Das ist schwächer als „zwei verschiedene Methoden", aber immer noch
> das Stärkste, was ohne gültigen Vergleichspunkt zu haben ist: eine lokale
> Suche findet nur, was in ihrem Einzugsbereich liegt, und zwei weit
> auseinanderliegende Startpunkte im selben Becken sind eine echte Aussage.

> **Ein Fall fällt hier heraus und darf nicht mitgezählt werden.** Bei rxn4113
> liegt der NEB 0.008 Å von der BS-TS-Opt-Struktur — aber die BS-TS-Opt-Struktur stammt
> ihrerseits aus diesem NEB (`ours from: fromneb` in `bs_neb_check.txt`). Die
> beiden Wege sind nicht unabhängig, die Übereinstimmung ist zirkulär. Dasselbe
> gilt für rxn6196. Wer die Tabelle nachbaut, muss die Herkunftsspalte prüfen.

**Die Abgrenzung zu §4, ohne die das zweierlei Maß wäre:**

```
drei Modelle             teilen Trainingsdaten, Zielniveau, Architekturfamilie
                         → Übereinstimmung kann geerbt sein
BS-TS-Opt und NEB-TS     teilen das Verfahren am Ende, aber nicht den
                         Startpunkt: RKS-TS gegen Climbing Image
                         → Übereinstimmung ist Evidenz über den Einzugsbereich
```

Die Unterscheidung ist nicht rhetorisch. Bei den Modellen kann die Ursache der
Übereinstimmung *vor* der Rechnung liegen — im geteilten Training. Bei den zwei
Optimierungen kann sie nur in der Fläche liegen, weil sie sonst nichts teilen,
das sie an denselben Punkt führen könnte.

---

### 3.7 (§3d) Pfad und Sattelpunkt trennen — Zahlen und Rechenweg

**Warum überhaupt getrennt.** ORCA prüft ein NEB gegen zwei Schwellen:
max|Fp| ≤ 0.020 eV/Å für das Band, max|FCI| ≤ **0.002** für das Climbing
Image. Bewegt wird das Climbing Image mit denselben Kräften erster Ordnung wie
jedes andere Bild. Praktisch jeder abgebrochene Lauf steht an der zweiten
Schwelle, nicht an der ersten.

**Das Rezept**, `pipeline/job_orca_nebci_split.sh`:

```
1  NEB-CI   ! UKS <METHODE> NEB-CI TightSCF SlowConv
            Endpunkte aus orca_neb_results/<rxn>/{reactant,product}.xyz
            Ergebnis: <name>_NEB-CI_converged.xyz

2  SP       ! UKS <METHODE> SP TightSCF SlowConv       am Climbing Image
            liefert bs.gbw -- die konvergierten gebrochenen Orbitale

3  TS-Opt   ! UKS <METHODE> OptTS <Freq|NumFreq> TightSCF SlowConv MORead
            %moinp aus Schritt 2, bei LEVEL=prod zusaetzlich NumHess true
```

`LEVEL` wählt die Methodenzeile — `wB97X 6-31G(d)` gegen
`wB97M-V def2-TZVP def2/J RIJCOSX`. Beide Niveaus laufen durch dasselbe
Skript, damit die Rezepte nicht auseinanderdriften. Auf Produktionsniveau muss
die Hesse numerisch sein: VV10 hat keine analytischen zweiten Ableitungen, und
`Calc_Hess true` scheitert dort mit
`ORCA_CPSCF: The CPSCF equations can not yet handle non-local correlation`.

**Ergebnis, billiges Niveau** (19 Reaktionen):

```
Climbing Image erreicht    16 von 19    ohne: rxn7060, rxn5691, rxn8827
TS-Optimierung konvergiert 16 von 16
genau eine imaginäre Mode  16 von 16
besteht alle drei Stufen   15 von 16    ohne: rxn7949 (Anteil 0.02, Rate 0.025)
```

rxn7949 ist ein sauber konvergierter Sattelpunkt — nur nicht der dieser
Reaktion. Der Lauf *vom höchsten gebrochenen Bild* schließt zwei der drei
Lücken: rxn5691 (Anteil 0.48, Rate 0.363) und rxn8827 (0.97, 1.391). Zusammen
**17 von 19**; Ausgangslage war 0 von 19 (T0).

**Ergebnis, Produktionsniveau** wB97M-V/def2-TZVP:

```
rxn         ⟨S²⟩ am CI   Zyklen   ν_imag /cm⁻¹   Anteil   Rate   Urteil
rxn0346        0.600        —        −1244.3      0.70    1.072   besteht
rxn6196        0.177       22         −730.6      0.97    1.296   besteht
rxn8827        0.370       23         −582.7      0.97    1.389   besteht
```

Drei von drei. rxn1320 hat sein Band nach über 24 h noch nicht abgeschlossen;
die restlichen 15 laufen. Bemerkenswert: die Brechung ist oben durchweg
stärker als unten (0.18–0.60 gegen 0.14–0.37 an denselben Reaktionen).

**Die Gegenprobe — sind es dieselben Punkte.** Kabsch-RMSD der
Produktionsstruktur (`tsopt2.xyz`) gegen die dort bekannten, beide Seiten auf
demselben Niveau:

```
rxn      BS-TS-Opt   RKS-TS   UKS-NEB    UMA-M   TSoptM
rxn0346      0.002    0.173     0.177    0.010    0.001
rxn6196      0.008    0.101     0.137    0.070        —
rxn8827      0.006    0.355     0.371    0.132    0.003
```

Tausendstel Ångström zu `BS-TS-Opt` und `TSoptM`. Und `sweep_summary` bestätigt es
von der anderen Seite: für alle drei besteht dort mindestens eine bekannte
Struktur alle drei Stufen, mit Frequenzen direkt daneben (rxn6196 −742.7 gegen
−730.6, rxn8827 −592.0 gegen −582.7, rxn0346 −1295.1 gegen −1244.3).

> **Auf dem billigen Niveau sah es anders aus.** Dort lagen drei der 16 Punkte
> weit von allem Bekannten — rxn6196 mit 1.044 Å, rxn3107 mit 0.941, rxn4518
> mit 0.516 — und bestanden trotzdem alle drei Stufen. Für rxn6196 ist das auf
> Produktionsniveau widerlegt: 0.008 Å. Der weit entfernte Punkt war ein
> Sattelpunkt der billigen Fläche, den es oben nicht gibt. rxn3107 und rxn4518
> laufen noch. Bis dahin gilt: **Strukturaussagen aus dem Prüfstand sind nicht
> übertragbar** (§5.1).

**Die Frequenzen sind nachgerechnet.** Die in Schritt 3 eingebettete
`NumFreq` nimmt **vorwärts**-Differenzen — 3N Verschiebungen gegen einen
Referenzpunkt, Fehler O(h). Jede eigenständige `NumFreq` dieses Projekts
nimmt dagegen zentrale Differenzen (6N, Fehler O(h²), und der Gegenschubser
fiele auf, wenn eine Verschiebung auf der anderen Lösung landete). Die Zeile
`Central differences ... used` / `... NOT used` steht in der Ausgabe.

Dieselbe Struktur, dieselben Orbitale, nur die Differenzenformel gewechselt
(`pipeline/job_orca_freq_central.sh`, Rezept wörtlich aus
`orca_freq/nebts_rxn0346/numfreq.inp`: `%freq CentralDiff true / Increment
0.005 end`, Geometrie und `MORead` aus `tsopt2.xyz` beziehungsweise
`tsopt2.gbw`):

```
           vorwärts (3N)   zentral (6N)   Differenz      Stufe 3 Anteil/Rate
rxn0346      −1244.26        −1287.95      43.7 cm⁻¹     0.70/1.072 → 0.68/1.039
rxn6196       −730.55         −775.28      44.7          0.97/1.296 → 0.97/1.276
rxn8827       −582.68         −588.98       6.3          0.97/1.389 → 0.97/1.390
```

**Kein Urteil ändert sich**, in Stufe 2 nicht und in Stufe 3 nicht — der
Eigenvektor ist gegen numerisches Rauschen deutlich robuster als der
Eigenwert. Die Frequenzen selbst verschieben sich um 1.1 bis 6.1 %, in allen
drei Fällen zu stärker negativ; vorwärts unterschätzt die Krümmung
systematisch. **Berichtet werden die zentralen Werte.**

> Der Test war ursprünglich anders gemeint: die Vermutung war, einzelne der
> 3N Verschiebungen könnten auf der *anderen* SCF-Lösung landen und so die
> Hesse verderben. Das ließ sich nicht prüfen — ORCA schreibt die verschobenen
> Rechnungen nach `tsopt2.lastscf`, das jede folgende überschreibt. Die
> gemessene Abweichung von 1 bis 6 % liegt im Rahmen dessen, was der Wechsel
> der Differenzenformel ohnehin erklärt; **ein Blattwechsel ist damit weder
> belegt noch ausgeschlossen.**

**Was nicht funktioniert hat, damit es niemand wiederholt:** `MORead` direkt in
der NEB-Eingabe wird abgewiesen; `NEB_Restart_GBWName` erwartet einen Basisnamen
und akzeptiert nur ORCA-eigene NEB-Orbitale (fremde Orbitale ergaben fünf
Segmentierungsfehler); `Rotate {HOMO,LUMO}` braucht numerische Indizes und
scheitert auch damit innerhalb eines NEB.

**Erzeugt von:** `pipeline/job_orca_nebci_split.sh`,
`pipeline/job_orca_tsopt_prod_resume.sh`, `pipeline/job_orca_tsopt_from_broken.sh`;
ausgewertet mit `pipeline/stage3_new.py` → `stage3_new.txt`.

---

## §4 · Thesen — Belege

### 4.1 Relabeling reicht nicht

**Die Aussage präzise:** ein UKS-Einzelpunkt an einer Geometrie, die auf der
restringierten Fläche optimiert wurde, gibt die Energie an einem Punkt, an dem
noch eine Kraft wirkt. Bei den 19 Reaktionen sind das im Median 1.697 eV/Å
(§3.0). Eine Barriere daraus ist keine Barriere; die Geometrie muss auf der
unrestringierten Fläche **optimiert** werden, nicht nur dort ausgewertet.

**Quantitativ, wie viel die Auswertung allein einbringt** — der Unterschied
zwischen der RKS- und der BS-Energie am selben RKS-TS:

```
ΔE_BS am RKS-TS, RKS-instabile Reaktionen
  am tiefsten   rxn4518   -648.5 meV
  Median-Bereich          -100 bis -200 meV
  am flachsten  rxn5690     -1.3 meV
```

Das Relabeling korrigiert diesen Betrag — und lässt den Geometriefehler
unberührt, der bei den Modellen bis zu 5.5 eV Energiedifferenz erzeugt (§2).

### 4.2 Modell-Uneinigkeit, zweiseitig

**Die Uneinigkeit ist ein verlässliches Warnsignal.** Die sechs Reaktionen ohne
einen einzigen Modell-Sattelpunkt sind rxn8837, rxn4113, rxn8885, rxn0894,
rxn7060, rxn0346 — und fünf davon sind die fünf größten Streuungen im Satz
(Ränge 1, 2, 3, 4, 6 der Tabelle in §1.3).

**Das korrigiert eine frühere Deutung von mir.** Ich hatte geschlossen, die
Modelle seien uneinig, *welcher* Sattelpunkt es ist. Falsch: in 7 von 8
Reaktionen, in denen mehrere Modelle einen Sattelpunkt treffen, ist es derselbe.

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

  7  gleicher Sattel        6  kein Modell ist ein Sattel
  5  nur ein Modell         1  verschiedene Sattelpunkte
```

**Erzeugt von:** `pipeline/mode_compare.py` — paarweiser Kabsch-RMSD über alle
Atome und, getrennt davon, über die reaktiven Atome der jeweiligen Reaktion.
Gezählt werden nur Reaktionen, in denen **mehr als ein** Modell einen Punkt
liefert, der Stufe 1 und 2 besteht; die übrigen stehen als „nur ein Modell"
beziehungsweise „kein Modell ist ein Sattel" in der Bilanz darunter.

**Die Streuung misst gemeinsames Danebenliegen, nicht konkurrierende
Übergangszustände.** Das ist die Formulierung, die ins Kapitel gehört — sie ist
korrekt und interessanter als die naheliegende.

**Einigkeit ist kein Beleg.** Bei den 26 stabilen Reaktionen stimmen die drei
auf 0.0045 Å überein; sie können ebenso gemeinsam falsch liegen, weil sie
Trainingsdaten, Zielniveau und Architekturfamilie teilen. Hierhin gehört das
Argument „es könnte einen tieferen Sattelpunkt geben, den alle drei verfehlen" —
zur Einigkeit, nicht zur Uneinigkeit.

### 4.3 Instabilität als Prädiktor

Aus §1.1: die Instabilitätsanalyse trifft in 84 von 100 Fällen die
richtige Reaktion, N_FOD in 78. Ob man sie als Ja/Nein oder als Zahl
berichtet, macht dabei keinen Unterschied (0.829 gegen 0.836) — also
genügt Ja/Nein.

**Was sie alles zugleich vorhersagt** — das ist der eigentliche Wert:

```
1. dass die Modelle danebenliegen        §1.2, §2.2
2. dass der RKS-TS ungültig ist          §3.0
3. dass auch die UKS-Rechnung schwer wird §3.2
```

Ein Deskriptor, drei Konsequenzen, eine Rechnung von Minuten.

### 4.4 Reichweite — was nicht gezeigt ist

Diese Liste gehört gedruckt, nicht in eine Fußnote:

- Alle BS-TS-Opt-Läufe starteten am RKS-TS. Eine lokale Suche findet nur,
  was unter ihrem Startpunkt liegt.
- Bei 10 von 19 Reaktionen existieren konkurrierende Sattelpunkte (6 nach
  Stufe 3). Ein weiterer, tieferer ist prinzipiell nicht ausgeschlossen.
- Die Prüfung „verbinden beide dieselben Minima?" ist nicht direkt
  durchgeführt; die Modenanalyse ist ein Ersatz.
- Die Triage-Schwelle 0.25 eV/Å ist abgelesen, nicht bestimmt, und beruht auf
  zehn nicht zufällig gewählten Punkten.
- Für 9 der 19 Reaktionen wurde nie eine TS-Optimierung von einer
  Modellgeometrie aus gestartet: rxn1283, rxn1320, rxn4113, rxn4518, rxn4522,
  rxn5690, rxn5691, rxn6196, rxn8885. Darunter alle fünf, bei denen die
  Optimierung vom RKS-TS aus versagt hat.

**Der einzige Gegenbeleg**, und er ist schwach, aber real: zehn zusätzliche
Suchen von Modellgeometrien aus haben keinen Sattelpunkt gefunden, der nicht
schon bekannt war (§2.5). Ein zweiter kommt aus §3d hinzu, ebenso schwach: die
aufgeteilte Suche über NEB-CI startet am Climbing Image eines Bandes statt am
RKS-TS und hat auf Produktionsniveau ebenfalls nur bekannte Punkte geliefert
(0.002 bis 0.010 Å).

> **Warum der Höhenvergleich nicht gerechnet werden konnte.** Der naheliegende
> Test — die Energien der konkurrierenden Sattelpunkte auf einem Niveau
> gegeneinander — ist versucht und **verworfen**. Alle 16 neuen Punkte wurden
> mit ihren Vergleichsstrukturen bei ωB97X/6-31G(d) gerechnet, die Kontrolle
> stimmte (16 von 16 Einzelpunkte reproduzieren die Energie ihrer
> TS-Optimierung). Der zugehörige Gradient zeigte dann, dass **keine einzige**
> der 65 Vergleichsgeometrien auf diesem Niveau stationär ist:
>
> ```
> neuer Sattelpunkt        0.002 – 0.015 eV/Å
> Vergleichsgeometrien     Minimum 0.37   Median 1.44   Maximum 5.97
>                          unter 0.15 eV/A:  0 von 65
> ```
>
> Eine Energie an einem Punkt, der 1.4 eV/Å vom Stationären entfernt liegt, ist
> ein Hangwert und keine Sattelhöhe. Strukturen, die auf ωB97M-V/def2-TZVP
> optimiert wurden, überstehen den Niveauwechsel nicht — Energievergleiche über
> Niveaugrenzen hinweg sind hier gegenstandslos, nicht bloß ungenau. Der Test
> ist auf Produktionsniveau nachzuholen; dort liegen die Vergleichsstrukturen
> von Haus aus auf ihrer eigenen Fläche.
>
> Innerhalb dieser Spalten war das Ergebnis erwartungstreu: der RKS-TS hatte in
> jeder Zeile den grössten Gradienten (1.84 bis 5.97), `BS-TS-Opt` und `UKS-NEB`
> die kleinsten.

---

## §5 · Ausblick — was validiert ist und was noch nicht

### 5.1 Der Prüfstand ist validiert

**Warum überhaupt ein zweites Niveau.** Ein BS-NEB bei ωB97M-V/def2-TZVP
braucht 7 bis 10 Stunden pro Reaktion, und jede Hesse-Matrix ist wegen VV10
numerisch. Methodenentwicklung in dieser Taktung ist nicht praktikabel.

**Der Test.** Ein Einzelpunkt plus Stabilitätsanalyse an jedem RKS-TS, bei
ωB97X/6-31G(d), gegen dieselbe Größe bei ωB97M-V/def2-TZVP.

```
instabil bei wB97X/6-31G(d)        26
instabil bei wB97M-V/def2-TZVP     19
beide                              19        ← echte Obermenge
Einstufung stimmt überein          38        widerspricht 7

Tiefe der Brechung, billiges Niveau:  Median -189.7 meV, tiefste -887.0
Produktionsniveau zum Vergleich:      -648.5 bis -1.3 meV
```

**Drei Eigenschaften, alle günstig:**

1. **Echte Obermenge** — alle 19 Produktionsfälle brechen auch billig. Die
   Aufgabenliste überträgt sich vollständig.
2. **Tiefere Brechung** — der Median liegt bei −190 meV gegen −100 bis −200 am
   Produktionsniveau, die tiefste bei −887 gegen −649. Das Phänomen ist
   verstärkt, nicht abgeschwächt: ein Verfahren, das hier den Zustand verliert,
   verliert ihn erst recht oben.
3. **Analytische Hesse-Matrizen** — ωB97X hat kein VV10.

Die 7 Abweichungen sind alle in derselben Richtung (billig instabil, teuer
stabil): rxn7945, rxn7937, rxn1150, rxn0896, rxn7936, rxn10005, rxn10054. Sie
vergrößern den Testsatz und verfälschen ihn nicht.

**Erzeugt von:** `pipeline/job_orca_cheap_stability.sh`,
`pipeline/cheap_stab_report.py` → `cheap_stab_report.txt`.

> **Wofür der Prüfstand nicht validiert ist: Geometrien.** Die Übertragung
> oben betrifft die *Einstufung* — welche Reaktion bricht. Sie sagt nichts
> darüber, ob ein Sattelpunkt an derselben Stelle liegt. rxn6196 zeigt, dass er
> es nicht muss: bei ωB97X/6-31G(d) liefert die aufgeteilte Suche dort einen
> sauber konvergierten Sattelpunkt, der alle drei Stufen besteht (Modenanteil
> 0.93) und 1.04 Å von jeder bekannten Struktur entfernt liegt. Auf
> ωB97M-V/def2-TZVP liegt derselbe Lauf 0.008 Å von der BS-TS-Opt-Struktur.
> Der billige Sattelpunkt existiert oben nicht. Der Prüfstand taugt zur
> Methodenentwicklung, nicht zur Strukturaussage — und Strukturvergleiche über
> die Niveaugrenze hinweg sind mit Vorsicht zu lesen, Energievergleiche gar
> nicht (§4.4).


### 5.2 Die Baseline läuft

`pipeline/job_orca_bs_neb_cheap.sh` — BS-NEB bei ωB97X/6-31G(d) auf vier
Reaktionen. Das Produktionsskript `job_bs_uks_neb18.sh` und
`bs_uks_neb_results/` bleiben unangetastet; die Ergebnisse gehen nach
`bs_uks_neb_cheap/`.

```
rxn7949   billig ΔE_BS -837 meV   Produktion: Walltime-Abbruch
rxn1320   billig -543            Produktion: Band RKS, TS-Opt danach auch —
                                 nie gebrochen
rxn8827   billig -167            Produktion: dasselbe
rxn8837   billig -507            KONTROLLE: Produktionsband ebenfalls
                                 durchgehend RKS, aber die TS-Optimierung danach
                                 brach (19 Werte, max 1.039) und landete 0.003 Å
                                 von der BS-TS-Opt-Struktur.
```

> **Die Kontrolllogik hat sich durch die Phasentrennung verschoben.** Das
> Jobskript formuliert sie noch in der alten Fassung: *„hielt die Brechung über
> 19 von 29 Bildern — scheitert diese hier auch, liegt es am billigen Aufbau"*.
> Nach §3.2 hielt auch dieses Band nicht. Ein kollabiertes Band am billigen
> Niveau ist damit **kein** Hinweis auf den Aufbau, sondern die exakte
> Reproduktion des Produktionsverhaltens.
>
> Die Frage, die die Kontrolle jetzt stellt, ist die nächste Phase: **bricht die
> TS-Optimierung nach dem Band auch hier?** Tut sie es, ist der Prüfstand
> vollständig äquivalent und für die Entwicklung brauchbar. Tut sie es nicht,
> unterscheidet sich das billige Niveau in genau dem Schritt, um den es geht.

**Stand 2026-08-13, ≈ 2 h Laufzeit — die Kontrollfrage ist beantwortet.**

```
                    Bandphase              TS-Opt danach
rxn8827  Produktion  0.000                  0.000            nie gebrochen
         billig      0.000                  1.051  (47 von 50)
rxn8837  Produktion  0.000                  1.039  (19 von 19)
         billig      0.000                  1.071  ( 8 von  8)
```

**Die Kontrolle besteht.** rxn8837 verhält sich am billigen Niveau genau wie in
der Produktion: Band restringiert, TS-Optimierung danach gebrochen, ⟨S²⟩ 1.071
gegen 1.039. Der Prüfstand bildet den Ablauf ab, um den es geht.

**Und eine Abweichung, die als Vorbehalt mitgeführt gehört.** Bei rxn8827
bricht die TS-Optimierung billig (1.051), in der Produktion nie. Passend zur
Tiefe: ΔE_BS ist dort billig −167 meV gegen −27.5 oben. Das billige Niveau ist
in genau dem Schritt **nachsichtiger** — ein reparierter NEB, der hier
funktioniert, ist damit noch kein Beleg, dass er es oben täte. Ein Fall, keine
Statistik, aber die Richtung ist zu beachten.

> **Diese beiden Zeilen sind aus dem Hauptlog gelesen und betreffen daher nur
> die TS-Optimierung**, nicht das Band — siehe §3.2. Für die billigen Läufe
> steht die Nachmessung der Bildorbitale noch aus; erst danach lässt sich
> sagen, ob die Bandphase am billigen Niveau dasselbe tut wie oben.

rxn8827 ist nach **1 h 34 min** durch, gegen 9 h 49 min in der Produktion.
rxn1320 steht kurz vor der TS-Phase (max|F_CI| 0.0036 gegen ein Kriterium von
0.002). rxn7949 pendelt seit Iteration 19 ohne klaren Trend und ist der
Kandidat für einen Lauf ohne Ergebnis.

**Das ist kein Reparaturversuch.** `BrokenSym` ist weiterhin zustandslos, die
Bilder werden weiterhin unabhängig relaxiert, `STABPerform` läuft weiterhin nicht
neben etwas anderem. Dasselbe Scheitern wird erwartet, nur schneller.

Es wird trotzdem gerechnet, weil es die **Baseline** ist: ein reparierter NEB,
der die gebrochene Lösung über das Band hält, beweist nichts, solange der
unreparierte nicht am selben Niveau, auf denselben Reaktionen, bei sonst
identischer Einstellung nachweislich verliert. Ohne das ist ein Lauf, der
funktioniert, nicht von einem Lauf zu unterscheiden, der Glück hatte.

Die Kontrollreaktion rxn8837 ist der Teil, der den Test falsifizierbar macht.

### 5.3 Die drei Vorschläge, je einer pro Fehlerart

| Fehlerart | Diagnose | Vorschlag | Aufwand |
|---|---|---|---|
| §3a Gipfel auf dem falschen Blatt | 14 von 19 Bändern halten die Brechung, aber nur 11 am höchsten Bild; wo der Gipfel restringiert ist, landet der NEB 0.055 Å vom RKS-TS | mehr Bilder oder Verfeinerung nahe dem Gipfel, damit die gebrochene Region aufgelöst wird — **ungeprüft**, weil der Mechanismus hinter der Korrelation offen ist | ein Band mit 16 statt 8 Bildern, am billigen Niveau |
| §3b Startpunktabhängigkeit | eine Suche vom RKS-TS aus; 10/19 mit konkurrierenden Sattelpunkten | Mehrfachstart aus RKS-TS + drei Modellgeometrien, Triage über den DFT-Gradienten | 4 Einzelpunkte + n Optimierungen pro Reaktion |
| §2 Kraftfehler | 0.031 eV/Å bei einem Kriterium von 0.03–0.05 | Modell als Startpunkt, DFT als Verfeinerung; unter 0.25 eV/Å 6/7, darüber 0/3 | ein Gradient pro Kandidat |

### 5.4 Offene Punkte

- TS-Opt von UMA-M für die 9 nie versuchten Reaktionen (Liste in §4.4)
- Stabilitätsprüfung an den **originalen** Transition1x-Geometrien
  (`data/Transition1x.h5`, 6.6 GB, liegt auf dem Cluster) — testet, ob der
  Datensatz selbst auf dem falschen Blatt sitzt
- `saddle_search_problem.md` trägt noch Zahlen von vor dem Sweep

---

## §6 · Reproduktion

### Erzeugende Skripte

| Datei | Zweck | Ausgabe |
|---|---|---|
| `pipeline/job_orca_freq_sweep.sh` | Array-Job, dreistufige Kette je Struktur | `orca_freq/<label>/` |
| `pipeline/make_freq_list*.py` | Aufgabenlisten | `freq_tasks.txt` |
| `pipeline/status_matrix.py` | Zellmatrix je Reaktion × Kandidat | `status_matrix.md` |
| `pipeline/saddle_matrix.py` | eine Zeile je Reaktion | `saddle_matrix.txt` |
| `pipeline/lowest_saddle.py` | wer den tiefsten gültigen Sattel fand | `lowest_saddle.txt` |
| `pipeline/model_saddle_stats.py` | 96 %/46 % und die Stufenbilanz | `model_saddle_stats.txt` |
| `pipeline/which_sheet_did_models_learn.py` | Labelhypothese | `which_sheet.txt` |
| `pipeline/force_error_at_ts.py` | Kraftfehler an der Modellgeometrie | `force_error_at_ts.txt` |
| `pipeline/model_sp_recheck.py` | Gegentest: gespeicherte gegen frisch gerechnete Modellkräfte | `model_sp_recheck/*.json` |
| `pipeline/checks.py` | Wachen: Positivkontrolle, Zählprobe, Sentinel, Abgleich | — |
| `pipeline/endpoint_report.py` | Stabilität an Edukt und Produkt | `endpoint_report.txt` |
| `pipeline/hess_compare.py` | ORCA gegen PySCF | `hess_cross_check.txt` |
| `pipeline/job_orca_irc_freq.sh` | IRC mit gelesener Hesse | `orca_irc/` |
| `pipeline/job_orca_cheap_stability.sh` + `cheap_stab_report.py` | Prüfstand | `cheap_stab_report.txt` |
| `pipeline/job_orca_bs_neb_cheap.sh` | Baseline am billigen Niveau | `bs_uks_neb_cheap/` |
| `pipeline/plot_saddle_landscape.py` | Hauptabbildung | `saddle_landscape.png` |
| `pipeline/plot_two_sheets.py` | Schema zweier Blätter | `two_sheets.png` |
| `pipeline/predictor_reffree.py` | referenzfreier Prädiktortest, §1.1 und §1.2 | Konsole |
| `pipeline/sep_analysis.py` | Vorfassung, gegen den RMSD zum RKS-TS — als Beleg für den Wechsel behalten | `stability_vs_fod_separation.txt` |
| `pipeline/verdict_final.py` | die dreistufige Regel, symmetrisch auf beide Seiten | Konsole |
| `pipeline/imag_mode.py` | Modenanalyse, Stufe 3 | Konsole |
| `pipeline/stage3_new.py` | Stufe 3 über mehrere Ergebnisverzeichnisse, `tsopt` / `tsopt2` / `numfreq` | `stage3_new.txt` |
| `pipeline/job_orca_nebci_split.sh` | **die Aufteilung**, §3d — `LEVEL=cheap\|prod`, `RXN_LIST`, `OUT_ROOT` | `bs_uks_nebci/`, `bs_uks_nebci_prod/` |
| `pipeline/job_orca_tsopt_prod_resume.sh` | Stufe 3 am Produktionsniveau mit `NumHess true` nachgezogen | `<rxn>/tsopt2.*` |
| `pipeline/job_orca_tsopt_from_broken.sh` | TS-Opt vom höchsten gebrochenen Bild | `tsopt_broken/` |
| `pipeline/job_orca_freq_central.sh` | Frequenz mit `CentralDiff true` an derselben Struktur nachgerechnet | `freq_central/` |
| `pipeline/job_orca_grad_gap.sh` | die 52 fehlenden Gradienten der Kontrollgruppe, §2.3 | `orca_freq/<rxn>_<Modell>/` |
| `pipeline/job_orca_sp_samelevel.sh` | Einzelpunkte aller Kandidaten einer Reaktion auf **einem** Niveau, §4.4 | `sp_samelevel/` |
| `pipeline/job_orca_grad_samelevel.sh` | Gradient dazu, `EnGrad MORead` — entscheidet, welche Zeile auswertbar ist | `sp_grad/` |
| `pipeline/job_orca_band_s2.sh`, `job_orca_band_s2_cheap.sh` | ⟨S²⟩ je Bandbild, mit eingebauter Positivkontrolle | `band_s2*/` |
| `pipeline/chapter_tables.py`, `chapter_tables2.py` | die Tabellen T0–T7 | `chapter_tables*.txt` |

### Cluster

DTU SLURM, `slid.fysik.dtu.dk`, Home `/home/energy/s242862/`.
ORCA parallelisiert über MPI-Ränge, nicht über Threads — also
`--ntasks=N --cpus-per-task=1`, nicht umgekehrt. Partitionslimit 48 h.

### Kreuzvalidierung der Hesse-Matrizen

Alle numerischen PySCF-Hesse-Matrizen wurden gegen ORCA geprüft:
**Modenüberlapp ≥ 0.9994**. Restunterschiede stammen aus abweichender
VV10-Implementierung und anderer Auslenkung (0.005 statt 0.01 Bohr).

> **Fehler, der dabei zuerst auftrat und keiner war.** Ohne Projektion der
> Translationen und Rotationen lasen sich die sechs Nullmoden als drei
> imaginäre. Das war mein Artefakt, kein Befund. `hess_compare.py` projiziert
> jetzt über eine massengewichtete SVD-Basis der sechs Starrkörperfreiheitsgrade.

### Bekannte Fallstricke

- `STABPerform` nur mit `RunTyp == SinglePoint` — die Kette muss dreiteilig sein
- `NumFreq` schreibt ein reduziertes Log; die SCF je Auslenkung überschreibt
  `numfreq.lastscf`
- **`NEB` schreibt die SCFs seiner Bilder überhaupt nicht ins Hauptlog.** Was
  dort steht, ist PREOPT — die Relaxation der beiden Endpunkte — und die
  abschließende TS-Optimierung. Wer die ⟨S²⟩-Werte eines NEB-Logs zählt, zählt
  die Endpunkte, wo ⟨S²⟩ = 0 richtig ist. Die Bildorbitale liegen stattdessen
  als `<base>_im{0..N}.gbw` vor und lassen sich nachträglich auslesen.
- **Zählprobe bei jeder Log-Auswertung.** Erwartete gegen gefundene
  Datensatzzahl. rxn4113 hat 100 Banditerationen über 8 Bilder und 36
  protokollierte SCFs — eine Subtraktion hätte den Punkt darüber drei Wochen
  früher aufgedeckt. Implementiert in `pipeline/checks.py`.
- **`NoIter` rechnet nichts.** Ein Einzelpunkt mit `NoIter` überspringt die
  Eigenschaftsauswertung, gibt `FINAL SINGLE POINT ENERGY 0.000000000000` aus
  und druckt kein ⟨S²⟩. Ein fehlender Wert, der als Zahl gelesen wird.
- **`MaxIter 1` konvergiert nicht** und ORCA hängt dann
  `(SCF not fully converged!)` an die Energiezeile, sodass das letzte Feld ein
  Wort ist. Auch hier wird kein ⟨S²⟩ gedruckt. Zum Auslesen gespeicherter
  Orbitale eignet sich ein gewöhnlicher `TightSCF`-Einzelpunkt mit `MORead`: von
  einer konvergierten Wellenfunktion aus braucht er einen Zyklus und bleibt auf
  derselben Lösung.
- **Modell-NEBs prüfen, ob sie konvergiert sind.** Vier von 19 sind es nicht
  (§2.3b), und ihr letzter Schritt sieht in der Ausgabedatei aus wie jeder
  andere.
- `BrokenSym` konvergiert zuerst den Hochspin-Triplett — bei ⟨S²⟩-Statistiken
  mit Schwelle 1.8 abtrennen
- SLURM-Arrays erscheinen als *eine* Zeile in `squeue` (`10737453_[4-44]` sind
  41 wartende Aufgaben, keine abgestürzten)
- Aufgabenlisten aus vorhandenen *Strukturen* bauen, nicht aus vorhandenen
  *Hesse-Matrizen* — sonst kann keine Lücke je geschlossen werden
- **Den ORCA-Pfad nie zusammenbauen.** `$EBROOTORCA/orca` existiert nicht — die
  Binärdatei liegt in `bin/`. Der Aufruf scheitert dann pro Rechnung in
  Millisekunden, das umgebende Skript läuft weiter und meldet Erfolg über einem
  leeren Verzeichnis. `ORCA=$(which orca)` plus `[ -x "$ORCA" ] || exit`, und
  nach jeder Rechnung auf `ORCA TERMINATED NORMALLY` **und** eine Energie
  ungleich null prüfen.
- **`STABPerform` verträgt nur `RunTyp SinglePoint`.** Mit `EnGrad` oder `Opt`
  bricht der Lauf nach zwei Sekunden ab mit
  `WARNING: Only RunTyp == SinglePoint possible with Stability Analysis!` und
  `Skipping actual calculation`. Wer den Gradienten *derselben* Lösung braucht,
  rechnet den Einzelpunkt mit Stabilitätsanalyse, speichert die Orbitale und
  hängt einen `EnGrad`-Lauf mit `MORead` daran.
- **Numerische Frequenzen: die Differenzenformel prüfen.** Eine eigenständige
  `NumFreq`-Rechnung nimmt hier zentrale Differenzen (6N Verschiebungen), die in
  eine `OptTS` eingebettete dagegen vorwärts (3N, Fehler O(h) statt O(h²), und
  ohne Gegenschubser, an dem eine abweichende SCF-Lösung auffiele). Die Zeile
  `Central differences ... used` / `... NOT used` steht in der Ausgabe; bei
  gebrochener Symmetrie gehört sie kontrolliert, nicht vorausgesetzt.

---

## Herkunft

Verdichtet aus `bs_ts_working.md` (Arbeitsstand, 1550 Zeilen). Dort stehen die
Einzelfallanalysen, die verworfenen Ansätze und die vollständige
Zuverlässigkeitsliste je Reaktion.
