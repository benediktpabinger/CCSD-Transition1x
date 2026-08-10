# Broken-Symmetry-Übergangszustände — Arbeitsstand

Arbeitsdokument, laufend fortgeschrieben. Alle Zahlen werden aus den
gespeicherten Ergebnissen erzeugt, nicht abgetippt.

## Die Frage

Der Benchmark vergleicht von MLIPs vorhergesagte Übergangszustände gegen eine
DFT-Referenz aus einem ORCA-NEB bei wB97M-V/def2-TZVP, gerechnet mit
restringierter Wellenfunktion. Bei 19 von 45 Reaktionen ist diese RKS-Lösung
an der Referenzgeometrie extern instabil — es existiert eine spingebrochene
Lösung, die tiefer liegt. Dort ist die Referenz nicht der Grundzustand.

Daraus folgen zwei Fragen, die unterschiedlich schwer zu beantworten sind:
ob die Referenz falsch ist, und ob die Modelle falsch sind.

## Das belastbarste Ergebnis

Der Gradient an einer Geometrie sagt, ob dort noch eine Kraft wirkt. Ein
Übergangszustand hat keine. Das Maß vergleicht mit nichts — es braucht weder
eine optimierte Struktur noch eine Annahme darüber, wo der richtige
Sattelpunkt liegt, und bleibt gültig, auch wenn sich alle unsere
BS-Sattelpunkte als falsch herausstellen sollten.

Gemessen wird jeweils auf der Fläche, die dort der Grundzustand ist: RKS wo
die restringierte Lösung extern stabil ist, BS wo sie es nicht ist.

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

**Die Referenzgeometrie liegt bei den Multireferenz-Reaktionen 1.70 eV/Å von
der Stationarität entfernt, die Modellvorhersagen bei 0.14** — die Modelle
sind dort zwölfmal näher an einem gültigen Punkt als die Referenz.

**Die Modelle verschlechtern sich um Faktor 2, nicht um 25.** Der große Faktor
aus den RMSD-Vergleichen entstand dadurch, dass gegen *unseren* optimierten
Sattelpunkt gemessen wurde. Nach dem referenzfreien Maß finden die Modelle
weiterhin fast-stationäre Punkte; was zunimmt, ist die Uneinigkeit darüber,
welcher der richtige ist.

## Warum die beiden Fragen unterschiedlich schwer sind

**„Unsere Referenz ist richtig" ist eine globale Aussage.** Man müsste zeigen:
es ist ein Sattelpunkt (Frequenz), es ist der richtige (Modenanalyse), er
verbindet die richtigen Minima (Endpunktprüfung) — und es gibt keinen
niedrigeren. Der letzte Punkt ist prinzipiell nicht beweisbar; eine
Energiefläche lässt sich nie vollständig absuchen. rxn4113 hat das vorgeführt:
Der zweite Sattelpunkt war da, wir hatten ihn nur nicht gesucht.

**„Das Modell liegt falsch" ist eine lokale Falsifikation.** Ist der Gradient
an der Modellgeometrie groß, ist sie kein Stationärpunkt und damit kein
Übergangszustand — unabhängig davon, was sonst auf der Fläche liegt.

## Prüfkette für einen Sattelpunkt

| Stufe | Frage | Verfahren | Kosten |
|---|---|---|---|
| 1 | Existiert eine BS-Lösung? | λ_min_ext aus der Stabilitätsanalyse | mittel |
| 2 | Ist der Punkt stationär? | analytischer Gradient | gering |
| 3 | Ist es ein Sattelpunkt? | numerische Hesse, 6N Gradienten | hoch |
| 4 | Gehört er zu dieser Reaktion? | Projektion der imaginären Mode auf die reaktiven Bindungen | keine |
| 5 | Verbindet er Edukt und Produkt? | Auslenkung entlang der Mode plus Relaxation | hoch |

Stufe 4 kostet nichts, weil die Hesse-Matrix ohnehin gespeichert wird, und hat
sich als der schärfste Filter erwiesen — sie hat drei Strukturen aussortiert,
die alle vorherigen Stufen bestanden hatten.

Stufe 5 ersetzt einen IRC, den PySCF nicht bietet. Sie ist einseitig: ein
Durchfallen widerlegt den Sattelpunkt, ein Bestehen beweist wenig, weil die
freie Relaxation die Information über die ausgelenkte Mode verliert.

## Modenanalyse

Aus der Hesse-Matrix erhält man den Eigenvektor der imaginären Mode — die
Bergab-Richtung des Sattelpunkts. Zwei Kennzahlen: welcher Anteil der Bewegung
auf den vier Atomen der beiden reaktiven Bindungen liegt, und wie stark sich
diese Bindungen entlang der Mode dehnen.

Eine gehinderte Methylrotation ist ebenfalls ein Sattelpunkt erster Ordnung.
Ohne diese Prüfung lässt sich der Fall nicht ausschließen.

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
rxn5691 (Bindungsrate 0.01). Alle drei galten zuvor als frequenzbestätigt.
rxn1320 ist der klarste Fall: die brechende C2-H6-Bindung ist dort von 1.981
auf 3.359 Å gelaufen, das Wasserstoffatom also vollständig abgelöst — die
Optimierung ist über den Übergangszustand hinausgeschossen.

## Reaktion für Reaktion

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

## Modellbewertung

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

geschrieben: /home/energy/s242862/final_scoring.json  (195 Zeilen)
```

## Korrekturen

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

**Die BS-NEB-Route ist der TS-Optimierung unterlegen.** Die Nullmessung zeigt
0.021 Å Streuung für die Optimierung gegen 0.669 Å für den NEB an derselben
Reaktion. `BrokenSym` ist zustandslos und hält die Symmetriebrechung nicht über
das Band — nur 5 von 11 ⟨S²⟩-Profilen sind zusammenhängend.

## Laufende Rechnungen

```
=== Endpunktpruefung (IRC-artig) (7) ===
  rxn0346            verdict=verbindet Edukt und Produkt  elapsed_s=9605.9
  rxn0894            verdict=verbindet Edukt und Produkt  elapsed_s=9653.4
  rxn1147            verdict=verbindet Edukt und Produkt  elapsed_s=6801.2
  rxn3107            verdict=verbindet Edukt und Produkt  elapsed_s=10398.4
  rxn7957            verdict=verbindet Edukt und Produkt  elapsed_s=11606.5
  rxn8827            verdict=verbindet Edukt und Produkt  elapsed_s=10957.5
  rxn8832            verdict=beide Seiten laufen zum selben Minimum  elapsed_s=12372.0

=== Reparaturen (5) ===
  rxn1283            mode=frommodel  n_geom_steps=76
  rxn1320            mode=tight  n_geom_steps=88
  rxn4518            mode=tight  n_geom_steps=89
  rxn5691            mode=tight  n_geom_steps=64
  rxn8885            mode=frommodel  n_geom_steps=73

=== Frequenzen ab NEB (2) ===
  rxn4113            n_imag=1  imag_freq=[155.9]  verdict=echter TS
  rxn6196            n_imag=1  imag_freq=[783.93]  verdict=echter TS

=== Frequenzen an Modellgeometrie (3) ===
  rxn1147/UMA-M      surface=RKS  grad_max_evang=0.049592  n_imag=1  mode_fraction=0.2166  e_vs_our_ts_meV=-230.94  lower=Modellgeometrie  verdict=Uebergangszustand
  rxn1147/UMA-S      surface=RKS  grad_max_evang=0.077368  n_imag=1  mode_fraction=0.2395  e_vs_our_ts_meV=-233.76  lower=Modellgeometrie  verdict=Uebergangszustand
  rxn1147/eSEN       surface=RKS  grad_max_evang=0.068172  n_imag=1  mode_fraction=0.2301  e_vs_our_ts_meV=-231.86  lower=Modellgeometrie  verdict=Uebergangszustand

=== TS-Opt ab Modellgeometrie (8) ===
  rxn0346/UMA-M      surface=BS  n_geom_steps=53
  rxn0894/UMA-S      surface=BS  n_geom_steps=54
  rxn3107/UMA-M      surface=BS  n_geom_steps=54
  rxn7949/UMA-M      surface=BS  n_geom_steps=39
  rxn7957/UMA-M      surface=BS  n_geom_steps=39
  rxn8827/UMA-M      surface=BS  n_geom_steps=37
  rxn8832/UMA-M      surface=BS  n_geom_steps=38
  rxn8837/UMA-M      surface=BS  n_geom_steps=40
```

| Job | Frage |
|---|---|
| 10720263 | Verbinden unsere Sattelpunkte Edukt und Produkt? (10 Reaktionen) |
| 10720520 | Negativkontrolle dazu an den drei falschen Sattelpunkten |
| 10720278 | Reparatur der drei falschen und zwei fraglichen Fälle |
| 10720438 | Sind die Modellgeometrien selbst Übergangszustände? (3 × 3) |
| 10720452 | Taugt die Modellgeometrie als Startpunkt einer TS-Suche? (10) |

## Offene Punkte

- rxn4522 in die Wandzeit gelaufen, 332 Schritte ohne Konvergenz
- rxn5690 mit ΔE_BS = −1.3 meV der Grenzfall; an der NEB-Geometrie existiert
  keine BS-Lösung mehr
- Keine Spinprojektion (Yamaguchi) — für Geometrien unkritisch, für absolute
  Barrieren nötig
- Nur ein Funktional (ωB97M-V); der Hybrid-Anteil steuert die Neigung zur
  Symmetriebrechung stark
- Energien der Modelle nicht untersucht, alles bisher ist Geometrie
