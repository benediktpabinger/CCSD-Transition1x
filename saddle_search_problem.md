# Das Sattelpunktproblem — Briefing

Zum Weitergeben an ein anderes Modell oder einen Kollegen. Selbsterklärend
gedacht; alle Zahlen sind gerechnet, nicht geschätzt.

---

## Ausgangslage

Ein Benchmark vergleicht Übergangszustände, die maschinelle Kraftfelder (OMol25:
UMA-S, UMA-M, eSEN) vorhersagen, gegen eine DFT-Referenz. Die Referenz ist ein
NEB in ORCA bei **wB97M-V/def2-TZVP**, gerechnet mit **restringierter**
Wellenfunktion (RKS). 45 Reaktionen aus Transition1x.

## Der Befund, der alles ausgelöst hat

Bei **19 von 45 Reaktionen ist die RKS-Lösung an der Referenzgeometrie extern
instabil.** Die Stabilitätsanalyse (`mf.stability(external=True)`) liefert einen
negativen kleinsten Eigenwert: es existiert eine spingebrochene UKS-Lösung, die
tiefer liegt — bis zu 650 meV an der Referenzgeometrie, an manchen
Modellgeometrien bis 4 eV.

Die Referenz ist dort also **nicht der Grundzustand**. Der Benchmark misst gegen
einen Punkt auf der falschen Fläche.

Also braucht es eine Ersatzreferenz: Übergangszustände, optimiert auf der
Broken-Symmetry-Fläche (BS-UKS).

---

## Das eigentliche Problem

**Welchen Sattelpunkt eine Optimierung findet, hängt davon ab, wo sie startet.**

Alle unsere BS-TS-Optimierungen starteten an der RKS-Referenzgeometrie — also an
genau dem Punkt, von dem wir wissen, dass er nicht der Grundzustand ist. Die
Optimierung findet dann den Sattelpunkt, der von dort aus bergab liegt. Das ist
ein systematischer Fehler, kein Zufall.

Drei Belege, alle gerechnet:

**rxn4113.** An der Referenz ist die Symmetriebrechung minimal (ΔE_BS = −8.4 meV,
⟨S²⟩ = 0.14). Es gibt ein zweites Becken **0.93 Å entfernt**, dort voll
ausgebildet (−1940 meV, ⟨S²⟩ = 1.01). Gefunden wurde es nur, weil eine
Modellvorhersage (UMA-M) zufällig dorthin zeigte. Ohne diesen Zufall hätten wir
unseren Punkt bestätigt und UMA-M als schlechtestes Modell im Feld gezählt —
obwohl es als einziges richtig lag. Genau das hat die RMSD-Auswertung getan.

**rxn8885.** Unser bestätigter Sattelpunkt: ⟨S²⟩ = 0.153, imaginäre Frequenz
1685 cm⁻¹. Eine Optimierung von der eSEN-Geometrie aus fand eine Struktur
**1.475 Å entfernt und 425 meV tiefer**, mit ⟨S²⟩ = 1.028. Die Frequenzrechnung
sagt: einzige imaginäre Mode **−25.75 cm⁻¹**, nächste Mode +4.92 cm⁻¹. Also kein
Sattelpunkt, sondern ein **Minimum** — das Tal hinter dem Übergangszustand.

**rxn0894.** Drei Kandidaten, alle mehr als 0.9 Å voneinander entfernt: unsere
Struktur, die Modellstrukturen, und eine dritte, die ein unabhängiger BS-NEB
gefunden hat. Keine ist gegen die anderen geprüft.

---

## Warum die Energie das nicht allein entscheidet

Naheliegend wäre: der tiefere Sattelpunkt gewinnt. Das stimmt auch — aber nur
unter zwei Bedingungen, und beide sind nicht trivial.

**Ein tieferer Stationärpunkt muss kein tieferer Sattelpunkt sein.** Er kann ein
Minimum bergab vom Übergangszustand sein. Beide sind stationär, beide liegen
tiefer. rxn8885 oben ist genau dieser Fall: 425 meV tiefer und trotzdem kein
Kandidat.

**Beide müssen dieselben Minima verbinden.** Ein tieferer Sattelpunkt, der zu
einer anderen Umlagerung gehört, ist kein Konkurrent.

Die Frequenz trennt Minimum von Sattelpunkt. Sie trennt **nicht** einen
Sattelpunkt dieser Reaktion von dem einer anderen Bewegung.

### Die Regel, die daraus folgt

Ein Punkt zählt nur als Übergangszustand einer bestimmten Reaktion, wenn alle
drei Stufen erfüllt sind:

| Stufe | Frage | Prüfung |
|---|---|---|
| 1 | stationär und tiefer als der Rivale? | Gradient, Energie |
| 2 | Sattelpunkt erster Ordnung? | genau eine imaginäre Frequenz |
| 3 | gehört er zu **dieser** Reaktion? | imaginäre Mode gegen die reaktiven Bindungen |

Stufe 3 wird regelmäßig übersprungen und hat hier zwei Urteile umgedreht, je
eines gegen jede Seite:

- **rxn1147:** Die Modelle liegen 233 meV tiefer und haben je genau eine
  imaginäre Frequenz. Nach Stufe 1 und 2 hätten sie gewonnen. Aber die zu
  knüpfende C1-O5-Bindung liegt bei ihnen bei **1.497 Å** — eine fertige
  Einfachbindung, gegen 1.864 Å bei uns — und ihre Mode bewegt sie mit 0.06
  gegen unsere 0.94. Sie sitzen im Produkttal.
- **rxn7957:** Dieselbe Prüfung, umgekehrtes Ergebnis. **Unsere** Struktur hat
  das wandernde H bei 1.120 Å von C5 (fertige C-H-Bindung) und 2.462 Å von C1
  (abgelöst). Die Modelle sitzen mitten im Transfer und 890 meV tiefer.

---

## Und die Grenze, die prinzipiell bleibt

**Man kann nicht ausschließen, dass anderswo ein tieferer Sattelpunkt liegt.**
Eine Energiefläche lässt sich nicht absuchen. Jede Prüfung, die wir haben, ist
lokal: sie sagt etwas über den Punkt, an dem sie rechnet, und nichts über den
Rest der Fläche.

Daraus folgt eine Asymmetrie, die man sauber halten muss:

- **„Das Modell liegt falsch"** ist eine lokale Falsifikation. Gradient
  0.48 eV/Å an der Modellgeometrie ⇒ kein Stationärpunkt ⇒ kein
  Übergangszustand. Fertig, unabhängig davon, was sonst auf der Fläche liegt.
- **„Unserer ist richtig"** ist eine globale Behauptung. Bewiesen ist immer nur:
  unserer ist ein gültiger Sattelpunkt, und der geprüfte Konkurrent ist keiner.

---

## Was wir dagegen versucht haben

**Endpunktprüfung** — entlang der imaginären Mode auslenken, relaxieren, sehen
wo man landet. **Verworfen.** Sie liefert falsch-negative Ergebnisse, bewiesen an
rxn8832, rxn8837, rxn7949: alle drei melden „beide Seiten laufen zum selben
Minimum", obwohl ein unabhängiger NEB dieselbe Struktur auf 0.003 Å findet, also
Edukt und Produkt nachweislich verbindet. Ursache: die freie Relaxation
minimiert in allen 3N Richtungen und hat nach wenigen Schritten vergessen,
welche Mode ausgelenkt wurde.

**Enger Vertrauensradius** (0.005 statt Standard) für die drei Fälle, deren Mode
die reaktiven Bindungen nicht bewegt. **0 von 3.** Alle drei konvergieren in
dieselbe Struktur zurück, rxn1320 auf 0.0009 Å. Der falsche Sattelpunkt ist
nicht die Folge zu großer Schritte.

**Start an der Modellgeometrie mit der tiefsten BS-Lösung.** **0 von 2.** Beide
Ergebnisse sind Minima (siehe rxn8885). Der Grund ist einsichtig: die Auswahl
nach „tiefste BS-Lösung" wählt den Punkt, der **am tiefsten im Tal** liegt. Ein
TS-Optimierer, der dort startet, hat keine imaginäre Richtung zu verfolgen und
rutscht auf den Talboden. Er meldet Konvergenz, weil das Kriterium am Gradienten
hängt. Das Auswahlkriterium war verkehrt herum; richtig wäre der **kleinste
Gradient**, nicht die tiefste Energie.

**IRC** (läuft). Ersetzt bei rxn1147 und rxn7957 das Bindungslängen-Urteil durch
eine Messung: läuft der Abstieg von einem Sattelpunkt durch die Rivalenstruktur
hindurch, liegt diese hinter dem Übergangszustand. Beantwortet die **lokale**
Frage, nicht die globale.

**Unabhängige Suche: BS-NEB.** Der einzige Ansatz, der den Startpunktfehler
nicht hat — ein NEB startet an relaxiertem Edukt und Produkt und interpoliert,
er sieht die Referenzgeometrie nie. 22 Läufe vorhanden, davon haben **9 die
Symmetriebrechung über das Band gehalten** (die anderen 13 sind ⟨S²⟩ = 0 über
das ganze Band, also verkappte RKS-Rechnungen — ORCAs `BrokenSym` ist zustandslos
und leitet den gebrochenen Zustand bei jedem Bild neu her).

Bei den neun ist das Ergebnis stark. Abstand der ORCA-NEB-TS-Struktur zu unserer:

```
rxn8837   0.003 Å zu uns     1.82-2.81 zu den Modellen
rxn7957   0.019              0.20
rxn1147   0.022              0.24
rxn8832   0.107              0.15
rxn4522   0.495              0.64
rxn5691   0.598              0.20    → Modelle
rxn0894   1.142              1.15    → dritte Struktur, ungeprüft
```

---

## Absicherung der Messungen selbst

Damit niemand an der falschen Stelle zweifelt: die Zahlen sind
codeunabhängig geprüft. ORCA 5.0.4 gegen PySCF, dieselben vier strittigen
Strukturen:

```
Struktur          imag. Frequenz ORCA / PySCF   Modenüberlapp
rxn1147 unserer        -581.13 / -590.84          0.999655
rxn1147 UMA-S          -252.90 / -253.49          0.999927
rxn7957 unserer        -670.84 / -677.04          0.999355
rxn7957 UMA-M          -696.27 / -690.26          0.999709
```

Genau eine imaginäre Mode in beiden Codes, reaktive Bindungsraten auf drei
Stellen identisch, Energiedifferenzen auf unter 1 meV, ⟨S²⟩ auf fünf Stellen.
Unsere Geometrien sind auch für ORCA stationär (0.010 eV/Å gegen 0.081–0.111 an
den Modellgeometrien).

Numerische Hesse-Matrizen in beiden Fällen — VV10 hat weder in PySCF noch in
ORCA 5.0.4 analytische zweite Ableitungen.

---

## Stand des Urteils über die 19

| | n | |
|---|---|---|
| unsere Referenz nachweislich besser | 6 | davon nur 2 aus einem echten Duell mit geprüftem Gegner |
| Modelle nachweislich besser | 3 | |
| beide auf demselben Punkt, kein Unterschied | 6 | Modelle 0–20 meV über unserem |
| offen | 4 | |

---

## Die offene Frage

Wie zeigt man, dass ein gefundener Sattelpunkt der relevante ist, wenn

- die Fläche nicht absuchbar ist,
- jede Optimierung vom Startpunkt abhängt,
- der einzige startpunktfreie Ansatz (NEB) die Symmetriebrechung nur in
  9 von 22 Fällen hält,
- und ein tieferer Stationärpunkt genauso gut ein Minimum sein kann?

Konkret gesucht: Verfahren, die den Konfigurationsraum um einen
Übergangszustand herum **systematisch** absuchen, statt von einem geratenen
Startpunkt aus zu optimieren — und die mit einer spingebrochenen Wellenfunktion
umgehen können, bei der jeder SCF-Neustart die Lösung verlieren kann.
