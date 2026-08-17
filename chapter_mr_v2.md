# Übergangszustände bei Multireferenzcharakter

*Fassung 2, 16.08.2026. Neu aufgebaut gegenüber `chapter_mr_transition_states.md`:
Story voran, danach neun Abschnitte mit jeweils denselben fünf Blöcken.
Rücknahmen und korrigierte Zahlen stehen gesammelt in Anhang A, die
Reproduktion in Anhang B.*

---

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

**Es gibt einen billigen Detektor.** Eine Stabilitätsanalyse an der
Eduktgeometrie, Minuten pro Reaktion, sagt vorher, ob ein Modell überhaupt einen
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
2.55 eV/Å und zwei imaginären Moden. Vom UKS-NEB erreichten 8 von 19 überhaupt
ein Climbing Image. Die TS-Optimierung ist mit 13 von 19 die beste Methode im
Feld — und liefert trotzdem keine verlässliche Antwort, weil ihr Ergebnis eine
Funktion des Startpunkts ist. Vier unabhängige Werkzeuge, dasselbe Muster,
derselbe Reaktionssatz: **das ist eine Aussage über die Fläche, nicht über eine
Werkzeugklasse.**

**Der Fix.** Ein NEB mit Climbing Image soll zwei Dinge zugleich leisten — einen
Pfad beschreiben und einen Punkt exakt treffen. Die Genauigkeitsforderungen
unterscheiden sich um den Faktor zehn, das Werkzeug ist für beides dasselbe.
Also verlangen wir es nicht mehr in einem Lauf: das Band liefert den Pfad auf der
groben Schwelle, sein höchstes Bild geht als Startpunkt in eine eigene
TS-Optimierung mit exakter Krümmung, und der einmal gefundene elektronische
Zustand wird weitergereicht statt bei jedem Schritt neu hergeleitet. **Von 0 auf
17 von 19.** Auf Produktionsniveau bisher 3 von 3. Und sie erfindet nichts: dort
liegen die neuen Punkte 0.001 bis 0.010 Å von Strukturen, die auf ganz anderem
Weg gefunden wurden.

**Was offen bleibt, steht ausdrücklich da.** Warum Bandverfahren an dieser
Schwelle scheitern, wissen wir nicht — drei Kandidaten, keiner von den anderen
getrennt. Es wurde kein tieferer Sattelpunkt gefunden; der Höhenvergleich ist
versucht und gescheitert, weil keine der 65 Vergleichsgeometrien auf dem
gemeinsamen Niveau stationär ist. Und es gibt noch keine Barrierenhöhe auf der
gebrochenen Fläche.

**In einem Satz.** Ein billiger Test sagt vorher, wo es klemmt; dort scheitern
erst die Modelle und dann auch die etablierten Rechenverfahren, alle auf dieselbe
Weise — weil dort zwei Flächen liegen, wo jedes dieser Werkzeuge eine annimmt.
Trennt man die beiden Aufgaben, die ein NEB sonst gleichzeitig lösen muss, wird
die richtige Fläche zugänglich.

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

## Niveau der Theorie

```
Produktion   ωB97M-V/def2-TZVP  def2/J  RIJCOSX  TightSCF
Prüfstand    ωB97X/6-31G(d)     TightSCF
Modelle      OMol25-Familie: UMA-S, UMA-M, eSEN
             trainiert gegen ωB97M-V/def2-TZVPD
```

Der Prüfstand existiert, weil ein BS-NEB auf Produktionsniveau 7 bis 10 Stunden
pro Reaktion braucht und dort jede Hesse-Matrix numerisch ist — ωB97M-V trägt
einen VV10-Term ohne analytische zweite Ableitungen. ωB97X hat kein VV10 und
liefert analytische Hesse-Matrizen. Wofür der Prüfstand validiert ist und wofür
nicht, steht in §8.

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
    ! UKS <METHODE> TightSCF NumFreq MORead        (Produktion, wegen VV10)
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

Eine externe Stabilitätsanalyse an der Eduktgeometrie — ein Einzelpunkt, Minuten
pro Reaktion — sagt vorher, ob ein Modell für diese Reaktion überhaupt einen
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
  Positiven und damit die AUC.
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
  abgeleitet.
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

- **Die 96 % links sind nicht dreistufig geprüft.** Für den einreferenziellen
  Satz wurde keine Hesse am RKS-TS gerechnet; Stufe 2 ist dort *by construction*
  unbekannt. Die 96 % sind der Anteil **stationärer** Punkte, die 46 % rechts
  der Anteil vollständig geprüfter. Die beiden Zahlen sind damit nicht exakt
  dieselbe Größe — die Richtung des Unterschieds ist eindeutig, sein genauer
  Betrag nicht.
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

# §8 · Der Fix: Pfad und Sattelpunkt trennen

## Aussage

Ein NEB mit Climbing Image soll zwei Dinge zugleich leisten: ein Band, das den
Pfad beschreibt, und ein Bild, das exakt auf dem Sattelpunkt sitzt. Die
Genauigkeitsforderungen unterscheiden sich um den Faktor zehn, das Werkzeug ist
für beides dasselbe — und praktisch jeder abgebrochene Lauf steht an der
strengeren der beiden Schwellen. Verlangt man beides nicht mehr in einem Lauf,
liefern **17 von 19** Reaktionen einen gebrochen-symmetrischen Sattelpunkt, der
alle drei Stufen besteht. Ausgangslage war 0 von 19.

Dies ist der einzige Abschnitt des Kapitels, der etwas **herstellt** statt etwas
festzustellen.

## Methode

**Warum überhaupt getrennt.** ORCA prüft ein NEB gegen zwei Schwellen:

```
Band              max|Fp|  ≤ 0.020 eV/Å
Climbing Image    max|FCI| ≤ 0.002 eV/Å      zehnmal strenger
```

Bewegt wird das Climbing Image mit denselben Kräften erster Ordnung wie jedes
andere Bild, ohne jede Krümmungsinformation. Einen Sattelpunkt auf 0.002 eV/Å
zu treiben, ohne zu wissen, wie die Fläche sich krümmt, dauert praktisch ewig.

**Das Rezept**, `pipeline/job_orca_nebci_split.sh`:

```
1  NEB-CI    ! UKS <METHODE> NEB-CI TightSCF SlowConv
             Endpunkte aus orca_neb_results/<rxn>/{reactant,product}.xyz
             → <name>_NEB-CI_converged.xyz

2  SP        ! UKS <METHODE> SP TightSCF SlowConv        am Climbing Image
             → bs.gbw, die konvergierten gebrochenen Orbitale

3  TS-Opt    ! UKS <METHODE> OptTS <Freq|NumFreq> TightSCF SlowConv MORead
             %moinp aus Schritt 2
             bei LEVEL=prod zusätzlich  NumHess true
```

**Warum der Zwischenschritt.** Schritt 2 friert den elektronischen Zustand ein.
Ohne ihn müsste Schritt 3 mit `BrokenSym` arbeiten, das bei jedem SCF neu
entscheidet, welche Lösung genommen wird (§0) — der Zustand käme nie zur Ruhe.

**Ein Skript für beide Niveaus.** `LEVEL` wählt die Methodenzeile, damit die
Rezepte nicht auseinanderdriften:

```
prod    wB97M-V def2-TZVP def2/J RIJCOSX      NumFreq, NumHess true
cheap   wB97X 6-31G(d)                        Freq
```

Auf Produktionsniveau **muss** die Hesse numerisch sein. `Calc_Hess true`
scheitert dort mit

```
ORCA_CPSCF: The CPSCF equations can not yet handle non-local correlation
```

— VV10 hat keine analytischen zweiten Ableitungen.

**Ein zweiter Zugang für die Fälle ohne Climbing Image.** Wo das Band kein CI
erreicht, wird stattdessen das höchste *gebrochene* Bild als Startpunkt genommen
(`job_orca_tsopt_from_broken.sh`).

## Zahlen

**Prüfstand ωB97X/6-31G(d), alle 19:**

```
Climbing Image erreicht     16 von 19    ohne: rxn7060, rxn5691, rxn8827
TS-Optimierung konvergiert  16 von 16
genau eine imaginäre Mode   16 von 16
besteht alle drei Stufen    15 von 16
```

```
rxn      ν_imag  Anteil   Rate      rxn      ν_imag  Anteil   Rate
rxn0346   -1926   0.94   0.865      rxn4522    -540   0.40   0.338
rxn0894   -1195   0.95   1.300      rxn5690   -1773   0.95   0.943
rxn1147    -664   0.60   0.949      rxn6196    -401   0.93   1.321
rxn1283    -124   0.84   0.980      rxn7957    -182   0.54   0.585
rxn1320    -404   0.34   0.205      rxn8832    -692   0.96   1.226
rxn3107   -1827   0.41   0.648      rxn8837    -800   0.91   1.308
rxn4113    -159   0.74   0.964      rxn8885   -2834   0.36   0.590
rxn4518    -893   0.89   1.302      rxn7949    -289   0.02   0.025   ← fällt durch
```

rxn7949 ist ein sauber konvergierter Sattelpunkt — nur nicht der dieser
Reaktion; seine imaginäre Mode rührt die reaktiven Bindungen praktisch nicht an.

**Der zweite Zugang schließt zwei der drei Lücken:**

```
rxn5691   Anteil 0.48   Rate 0.363   besteht
rxn8827          0.97        1.391   besteht
rxn1283          0.83        0.788   besteht   (hat auch über NEB-CI einen Punkt)
rxn8885          0.40        0.002   fällt durch (hat über NEB-CI einen Punkt)
```

**Zusammengenommen: 17 von 19.** Offen bleiben rxn7060, das auf keinem Niveau
ein Climbing Image erreicht, und rxn7949.

**Produktionsniveau ωB97M-V/def2-TZVP:**

```
rxn        ⟨S²⟩ am CI   Zyklen   ν_imag /cm⁻¹   Anteil   Rate    Urteil
rxn0346       0.600        —        −1244.3      0.70    1.072   besteht
rxn6196       0.177       22         −730.6      0.97    1.296   besteht
rxn8827       0.370       23         −582.7      0.97    1.389   besteht
```

Drei von drei. Die Brechung ist oben durchweg stärker als unten (0.18 bis 0.60
gegen 0.14 bis 0.37 an denselben Reaktionen).

> **TODO — 19 Multireferenz und 3 Kontrollen laufen.** Stand 16.08.2026, 20 Jobs
> auf Produktionsniveau, 48 h Wandzeit je Aufgabe; die fertigen Bänder brauchten
> rund 20 Stunden. Erwartet 17.08. Die Tabelle oben ist dann auf 19 Zeilen zu
> erweitern:
>
> ```
> rxn        ⟨S²⟩ am CI   Zyklen   ν_imag   Anteil   Rate   Urteil
> rxn0894         —          —        —        —      —       —
> rxn1147         —          —        —        —      —       —
> rxn1283         —          —        —        —      —       —
> rxn1320         —          —        —        —      —       —
> rxn3107         —          —        —        —      —       —      ← 0.941 Å billig
> rxn4113         —          —        —        —      —       —
> rxn4518         —          —        —        —      —       —      ← 0.516 Å billig
> rxn4522         —          —        —        —      —       —
> rxn5690         —          —        —        —      —       —
> rxn5691         —          —        —        —      —       —
> rxn7060         —          —        —        —      —       —      ← nie ein CI
> rxn7949         —          —        —        —      —       —
> rxn7957         —          —        —        —      —       —
> rxn8832         —          —        —        —      —       —
> rxn8837         —          —        —        —      —       —
> rxn8885         —          —        —        —      —       —
> ```

**Die Gegenprobe: sind es dieselben Punkte.** Kabsch-RMSD der
Produktionsstruktur (`tsopt2.xyz`) gegen die dort bekannten, beide Seiten auf
demselben Niveau:

```
rxn      BS-TS-Opt   RKS-TS   UKS-NEB    UMA-M   TSoptM
rxn0346      0.002    0.173     0.177    0.010    0.001
rxn6196      0.008    0.101     0.137    0.070        —
rxn8827      0.006    0.355     0.371    0.132    0.003
```

Tausendstel Ångström zu BS-TS-Opt und TSoptM. Damit erfüllt jeder dieser drei
Punkte die Bedingung eines **unabhängig bestätigten Sattelpunkts**: dieselbe
Struktur, erreicht von zwei Suchen, die weder Startpunkt noch Verfahren teilen —
die eine beginnt am RKS-TS, die andere am Climbing Image eines Bandes.

`sweep_summary` bestätigt es von der anderen Seite: für alle drei besteht dort
mindestens eine bekannte Struktur alle drei Stufen, mit Frequenzen direkt
daneben (rxn6196 −742.7 gegen −730.6; rxn8827 −592.0 gegen −582.7; rxn0346
−1295.1 gegen −1244.3).

## Kontrollen

**Zwei Erklärungen sind geprüft und ausgeschieden**, bevor die dritte übrig
blieb:

```
flächiger Spinkollaps      widerlegt -- 14 von 19 Bändern halten den
                           gebrochenen Zustand (§7)
mehr Brechung hilft        widerlegt -- BrokenSym 2,2 sammelt kein
                           zusätzliches Bild ein (rxn8837 identisch,
                           rxn1320 verliert drei) und konvergiert bei
                           WENIGER Brechung schneller
```

**Die Frequenzen sind nachgerechnet.** Die in Schritt 3 eingebettete `NumFreq`
nimmt **vorwärts**-Differenzen — 3N Verschiebungen gegen einen Referenzpunkt,
Fehler O(h). Jede eigenständige `NumFreq` dieses Projekts nimmt dagegen zentrale
Differenzen (6N, Fehler O(h²)). Die Zeile `Central differences ... used` /
`... NOT used` steht in der Ausgabe. Dieselbe Struktur, dieselben Orbitale, nur
die Differenzenformel gewechselt:

```
           vorwärts (3N)   zentral (6N)   Differenz    Stufe 3 Anteil/Rate
rxn0346      −1244.26        −1287.95      43.7 cm⁻¹   0.70/1.072 → 0.68/1.039
rxn6196       −730.55         −775.28      44.7        0.97/1.296 → 0.97/1.276
rxn8827       −582.68         −588.98       6.3        0.97/1.389 → 0.97/1.390
```

**Kein Urteil ändert sich**, in Stufe 2 nicht und in Stufe 3 nicht — der
Eigenvektor ist gegen numerisches Rauschen deutlich robuster als der Eigenwert.
Die Frequenzen verschieben sich um 1.1 bis 6.1 %, in allen drei Fällen zu
stärker negativ. **Berichtet werden die zentralen Werte.**

**Eingebaute Sperre:** der Nachrechnungslauf bricht mit Exit 7 ab, wenn ORCA
entgegen der Anweisung doch vorwärts rechnet.

> **TODO — Frequenzen der 19 Produktionsläufe.** Sie werden mit der
> eingebetteten `NumFreq` erzeugt, also vorwärts. Nach diesem Ergebnis ändert
> das keine Urteile, aber die zu berichtenden Zahlen. Nachzuziehen mit
> `job_orca_freq_central.sh` aus den konvergierten Geometrien, rund 30 Minuten
> je Reaktion.

## Vorbehalte

- **Warum ein Bandverfahren an dieser Schwelle scheitert, ist nicht erklärt.**
  Drei Kandidaten, die die vorliegenden Daten nicht voneinander trennen: die
  Zwei-Blatt-Struktur der Fläche mit einer Naht, die ein Band zwangsläufig
  kreuzt; `BrokenSym` als zustandsloses Verfahren; und das Fehlen zweiter
  Ableitungen im Bandverfahren. **Belegt ist, wo es hängt und was es behebt —
  nicht, warum es hängt.**
- **Der Prüfstand ist für die Einstufung validiert, nicht für Geometrien.**
  rxn6196 liefert auf ωB97X/6-31G(d) einen sauber konvergierten Sattelpunkt, der
  alle drei Stufen besteht (Modenanteil 0.93) und **1.04 Å** von jeder bekannten
  Struktur entfernt liegt. Auf ωB97M-V/def2-TZVP liegt derselbe Lauf 0.008 Å von
  BS-TS-Opt. Der billige Sattelpunkt existiert oben nicht. Zwei weitere
  Kandidaten dieser Art — rxn3107 (0.941 Å) und rxn4518 (0.516 Å) — sind noch
  nicht auf Produktionsniveau geprüft.
- **Die Methode reproduziert, sie entdeckt nicht.** Auf Produktionsniveau landet
  sie auf Strukturen, die bereits bekannt waren. Das ist die Validierung, aber
  es ist kein Fund.
- **Die Kontrolle fehlt noch.**

> **TODO — Kontrolle mit einreferenziellen Reaktionen.** Dasselbe Rezept auf
> rxn1061, rxn0101 und rxn0896, ausgewählt nach dem Abstand zur Instabilität
> (`lmin_int` 0.919 / 0.327 / 0.224), alle mit 11 Atomen und RKS-Gradient unter
> 0.06 eV/Å. Erwartung: ⟨S²⟩ ≈ 0 und derselbe Punkt wie der RKS-TS.
>
> ```
> rxn        ⟨S²⟩ am CI   ν_imag   RMSD zum RKS-TS   Urteil
> rxn1061         —          —            —            —
> rxn0101         —          —            —            —
> rxn0896         —          —            —            —
> ```
>
> **Käme dort eine gebrochene Lösung heraus, bräche das Verfahren Symmetrie, wo
> keine zu brechen ist — und der Befund an den 19 wäre entwertet.** Ohne diese
> Zeilen ist §8 nicht abgeschlossen.

**Was nicht funktioniert hat, damit es niemand wiederholt:**

```
MORead direkt in der NEB-Eingabe          wird abgewiesen
NEB_Restart_GBWName mit fremden Orbitalen fünf Segmentierungsfehler; der
                                          Parameter erwartet einen Basisnamen
                                          und liest <base>_im{N}.gbw je Bild,
                                          akzeptiert aber nur ORCA-eigene
                                          NEB-Orbitale
Rotate {HOMO,LUMO,...}                    braucht numerische Indizes, und
                                          scheitert auch damit innerhalb eines NEB
Calc_Hess true mit wB97M-V                CPSCF kann kein VV10
```

**Erzeugt von:** `pipeline/job_orca_nebci_split.sh` (`LEVEL=cheap|prod`,
`RXN_LIST`, `OUT_ROOT`), `pipeline/job_orca_tsopt_prod_resume.sh`,
`pipeline/job_orca_tsopt_from_broken.sh`, `pipeline/job_orca_freq_central.sh`;
ausgewertet mit `pipeline/stage3_new.py` → `stage3_new.txt`.

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

- **Warum Bandverfahren an der 0.002-Schwelle scheitern, ist unbekannt.** Drei
  Kandidaten, keiner von den anderen getrennt (§8).
- **Kein tieferer Sattelpunkt gefunden.** Die Methode reproduziert, sie entdeckt
  nicht — jedenfalls nicht in den bisher auf Produktionsniveau gerechneten
  Fällen.
- **Keine Barrierenhöhe auf der BS-Fläche.** Das Kapitel liefert Sattelpunkte,
  aber noch keine Zahl, die in eine Tabelle mit Aktivierungsenergien gehörte.
  Dafür fehlen die Endpunktenergien auf derselben Fläche.
- **Alle BS-TS-Opt-Läufe starteten am RKS-TS.** Eine lokale Suche findet nur,
  was unter ihrem Startpunkt liegt.
- **Bei 10 von 19 Reaktionen existieren konkurrierende Sattelpunkte** (6 nach
  Stufe 3). Ein weiterer, tieferer ist prinzipiell nicht ausgeschlossen.
- **Die Prüfung „verbinden beide dieselben Minima?" ist nicht direkt
  durchgeführt**; die Modenanalyse ist ein Ersatz. Der IRC wurde versucht und
  verworfen (Anhang A.10).
- **Die Triage-Schwelle 0.25 eV/Å ist abgelesen, nicht bestimmt**, und beruht auf
  zehn nicht zufällig gewählten Punkten.
- **Für 9 der 19 Reaktionen wurde nie eine TS-Optimierung von einer
  Modellgeometrie aus gestartet:** rxn1283, rxn1320, rxn4113, rxn4518, rxn4522,
  rxn5690, rxn5691, rxn6196, rxn8885 — darunter alle fünf, bei denen die
  BS-TS-Opt vom RKS-TS aus versagt hat.
- **Der Prüfstand ist für Geometrien nicht validiert** (§8).

**Die zwei Gegenbelege**, beide schwach, aber real: zehn zusätzliche Suchen von
Modellgeometrien aus haben keinen Sattelpunkt gefunden, der nicht schon bekannt
war (§6); und die aufgeteilte Suche über NEB-CI startet am Climbing Image eines
Bandes statt am RKS-TS und hat auf Produktionsniveau ebenfalls nur bekannte
Punkte geliefert (§8).

**Erzeugt von:** `pipeline/job_orca_sp_samelevel.sh` (Einzelpunkte),
`pipeline/job_orca_grad_samelevel.sh` (Gradienten).

---

# Was als Nächstes zu tun ist

```
1  Die Kontrolle zu §8 auswerten                    läuft, erwartet 17.08.
2  Die 19 Produktionsläufe in §8 nachtragen         läuft, erwartet 17.08.
3  Frequenzen der 19 zentral nachrechnen            ~30 min je Reaktion
4  rxn3107 und rxn4518 auf Produktionsniveau        Teil von 2
   prüfen -- lösen sie sich auf wie rxn6196?
5  Den Höhenvergleich auf Produktionsniveau         setzt 2 voraus
   wiederholen, wo die Vergleichsstrukturen
   von Haus aus stationär sind
6  Endpunktenergien auf der BS-Fläche               noch nicht begonnen
   -- erst dann gibt es Barrierenhöhen
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

**Falsch.** Die Einträge `nebts_<rxn>` in `sweep_summary` sind unser eigenes
BS-NEB-Ergebnis, nicht der RKS-TS.

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

**Und geometrisch löst es sich ebenfalls auf.** rxn6196 lag auf dem billigen
Niveau 1.04 Å von jeder bekannten Struktur entfernt und bestand alle drei
Stufen. Auf Produktionsniveau liegt derselbe Lauf 0.008 Å von BS-TS-Opt. Der
weit entfernte Punkt war ein Sattelpunkt der billigen Fläche, den es oben nicht
gibt.

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

**Dritte und aktuelle Fassung: es ist ein Konvergenzproblem** an der
Climbing-Image-Schwelle von 0.002 eV/Å (§8). Diese Fassung hat als einzige eine
Vorhersage gemacht — die Trennung von Pfad und Sattelpunkt müsse helfen — und
sie hat sich bestätigt. **Warum** ein Bandverfahren dort scheitert, ist damit
weiterhin nicht erklärt.

---

# Anhang B · Reproduktion

## Erzeugende Skripte

| Datei | Zweck | Ausgabe |
|---|---|---|
| **Detektor und Prädiktor** | | |
| `pipeline/stability_pipeline.py` | Stabilitätsanalyse, 45 Reaktionen × 4 Geometrien | `stab_pipeline/<rxn>/result.json` |
| `pipeline/predictor_reffree.py` | referenzfreier Prädiktortest, §1 | Konsole |
| `pipeline/sep_analysis.py` | Vorfassung gegen den RMSD zum RKS-TS, als Beleg für den Wechsel behalten (A.7) | `stability_vs_fod_separation.txt` |
| `pipeline/job_orca_cheap_stability.sh` + `cheap_stab_report.py` | Prüfstand: überträgt sich die Einstufung auf ωB97X/6-31G(d)? | `cheap_stab_report.txt` |
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
| `pipeline/job_orca_bs_neb_cheap.sh` | dieselbe Baseline am billigen Niveau | `bs_uks_neb_cheap/` |
| `pipeline/job_orca_band_s2.sh`, `job_orca_band_s2_cheap.sh` | ⟨S²⟩ je Bandbild, mit eingebauter Positivkontrolle | `band_s2*/` |
| `pipeline/tsopt_null.py` | Nullmessung: Streuung der TS-Opt gegen die des NEB | Konsole |
| `pipeline/bs_freq.py`, `bs_freq2.py` | numerische BS-UKS-Hesse | `BSFREQ_OUT` |
| `pipeline/hess_compare.py` | ORCA gegen PySCF | `hess_cross_check.txt` |
| **Der Fix** | | |
| `pipeline/job_orca_nebci_split.sh` | **die Aufteilung**, §8 — `LEVEL=cheap\|prod`, `RXN_LIST`, `OUT_ROOT` | `bs_uks_nebci/`, `bs_uks_nebci_prod/` |
| `pipeline/job_orca_tsopt_prod_resume.sh` | Stufe 3 am Produktionsniveau mit `NumHess true` nachgezogen | `<rxn>/tsopt2.*` |
| `pipeline/job_orca_tsopt_from_broken.sh` | TS-Opt vom höchsten gebrochenen Bild | `tsopt_broken/` |
| `pipeline/job_orca_freq_central.sh` | Frequenz mit `CentralDiff true` an derselben Struktur | `freq_central/` |
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
  Bildes ist bandspezifisch; ihn von einem Produktionsband auf eine billige
  Messung zu übertragen liefert stillschweigend das falsche Bild.

---

*Ende. Vorfassung als `chapter_mr_transition_states.md`, Commit e24f53f.*
