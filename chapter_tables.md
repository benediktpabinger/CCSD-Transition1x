# Die Tabellen zum Kapitel

Fünf Tabellen, die `chapter_mr_transition_states.md` braucht und nicht hatte.
Erzeugt von `pipeline/chapter_tables.py`, Rohausgabe in `chapter_tables.txt`.

Jede steht hier, weil eine bestimmte Frage aus dem Kapitel heraus nicht zu
beantworten war. Die Reihenfolge folgt dem Kapitel.

| | Tabelle | gehört nach | beantwortet |
|---|---|---|---|
| T0 | Der RKS-TS auf beiden Flächen | §3 Eröffnung | Ist die Referenz falsch gerechnet oder auf der falschen Fläche? |
| T1 | Wer hat einen Sattelpunkt gefunden | §3c | Welche Methode bei welcher Reaktion — mit Frequenz |
| T2 | Barrieren in eV | §3c | Wie groß sind die Barrieren eigentlich? |
| T3 | Welche Reaktionen das sind | §0 | Ist die MR-Gruppe chemisch homogen? |
| T4 | Die konditionierende Größe | §1 | Wie multireferenziell ist jede einzelne? |

Alles bei ωB97M-V/def2-TZVP, ORCA 5.0.4. Gradienten als größte Komponente in
eV/Å, stationär heißt < 0.15.

---

## T0 · Der RKS-TS auf beiden Flächen

**Die Frage, die diese Tabelle klärt:** ist der RKS-TS eine schlechte Rechnung
oder eine gute Rechnung auf der falschen Fläche? Dieselbe Geometrie, zwei
Gradienten — links die Fläche, auf der sie optimiert wurde, rechts die, auf der
die Reaktion abläuft.

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

**Das ist die knappste Fassung der These des Kapitels.** Der RKS-TS ist kein
Fehler. Er ist ein sauberer Sattelpunkt der restringierten Fläche — 18 von 19
unterschreiten dort 0.15 eV/Å, mit einem Median von 0.059. Auf der Fläche, auf
der die Reaktion abläuft, ist keiner von ihnen ein Stationärpunkt.

Der Faktor zwischen den beiden Spalten läuft von 4 bis 63.

**Der Ausreißer gehört dazu.** rxn6196 ist mit 0.179 eV/Å auch restringiert
nicht ganz konvergiert. Es ist der einzige Fall, in dem die Referenz auf ihrer
eigenen Fläche unsauber ist, und die Tabelle sagt es, statt es zu glätten.

> **Warum die beiden Zahlen leicht zu verwechseln sind.** Im Datensatz heißen
> sie `rks_grad.max_evang` und `bs.bs_grad.max_evang` und stehen im selben
> Eintrag. Wer die erste liest, wo die zweite gemeint ist, bekommt eine
> Referenz, die überall gültig aussieht — die genaue Umkehrung des Befunds. Beim
> Bauen dieser Tabelle ist mir das zuerst passiert.

---

## T1 · Wer hat einen Sattelpunkt gefunden

Die Tabelle, auf die das Kapitel siebenmal verweist. Gegenüber
`saddle_matrix.txt` um die **Imaginärfrequenz** erweitert, damit konkurrierende
Sattelpunkte nicht nur nach Lage, sondern nach Charakter unterscheidbar sind.

```
rxn            RKS-TS      UKS-NEB       unsere        UMA-S        UMA-M         eSEN      TSopt/M
-----------------------------------------------------------------------------------------------------
rxn7949   n.stat 1.69           --      JA -732           --           --           --       JA -69
rxn8832   n.stat 2.73      JA -656      JA -651      JA -634      JA -644  n.stat 0.23      JA -652
rxn1320   n.stat 2.07  n.stat 2.06      JA -313      JA -416      JA -405      JA -408           --
rxn4113   n.stat 0.39      JA -148      JA -152  n.stat 0.17  n.stat 0.19  n.stat 0.18           --
rxn8885   n.stat 2.64           --     JA -1673  n.stat 0.48  n.stat 0.18  n.stat 0.38           --
rxn6196   n.stat 0.64  n.stat 0.68      JA -760      JA -781      JA -743      JA -744           --
rxn0346   n.stat 2.61  n.stat 2.55     JA -1289  n.stat 0.24  n.stat 0.18  n.stat 0.48     JA -1295
rxn4518   n.stat 2.95           --       JA -90   2 imag -86      JA -105  2 imag -133           --
rxn3107   n.stat 1.65           --     JA -1486  n.stat 0.16     JA -1438  n.stat 0.16     JA -1484
rxn8837   n.stat 1.70      JA -817      JA -819  n.stat 0.75  n.stat 0.76   2 imag -81       JA -59
rxn7060   n.stat 1.77           --     JA -2024  n.stat 0.88  n.stat 0.33  n.stat 1.13  n.stat 1.71
rxn5691   n.stat 1.42           --       JA -95           --           --           --           --
rxn1283   n.stat 2.39           --  2 imag -123  n.stat 0.16       JA -94       JA -73           --
rxn8827   n.stat 1.13  n.stat 1.07      JA -590  n.stat 0.18      JA -590  n.stat 0.23      JA -592
rxn4522   n.stat 1.88      JA -495  n.stat 1.20           --           --           --           --
rxn1147   n.stat 1.84      JA -589      JA -581      JA -253           ·           ·      JA -230
rxn0894   n.stat 1.35     JA -1075      JA -617  n.stat 0.78  n.stat 1.32  n.stat 0.79           --
rxn7957   n.stat 0.90      JA -670      JA -671           ·      JA -696           ·      JA -624
rxn5690   n.stat 0.16      JA -448           --  n.stat 0.18      JA -869  n.stat 0.15           --

JA ν        stationär und genau eine imaginäre Mode; ν in cm-1
n.stat x    Gradient x eV/Å, kein Stationärpunkt und damit kein Sattelpunkt
N imag ν    stationär, aber Sattelpunkt N-ter Ordnung
--          keine Struktur vorhanden
·           Struktur existiert und ist laut saddle_matrix.txt ein Sattelpunkt,
            liegt aber in einem Verzeichnis, das diese Tabelle nicht durchsucht
```

> **Abdeckung, ehrlich.** T1 liest `orca_freq/<label>` und `orca_irc/<label>`.
> Vier Zellen bei rxn1147 und rxn7957 tragen deshalb `·` statt einer Frequenz,
> obwohl `saddle_matrix.txt` dort ein Urteil hat. **Für die Abdeckung bleibt
> `saddle_matrix.txt` maßgeblich; T1 fügt die Frequenz hinzu, wo sie auffindbar
> ist.**
>
> Die erste Fassung dieser Tabelle war schlimmer: sie zeigte für rxn1147 und
> rxn7957 in der Spalte *unsere* ein `--`, weil unsere Strukturen dort im
> IRC-Lauf gerechnet wurden und in `orca_irc/<rxn>_ours` liegen. Das sind genau
> die beiden Reaktionen, in denen unsere Struktur verliert — dieselbe
> Auslassung, die `lowest_saddle.py` einmal hatte und die dort aus 11 von 13
> ein 11 von 11 machte. Behoben, die Zeilen oben sind die korrigierten.

**Was die Frequenzspalte zusätzlich zeigt.** Zwei Fälle, in denen zwei Methoden
denselben Sattelpunkt gefunden haben, sind an der Frequenz sofort erkennbar:
rxn8837 mit −817 gegen −819 cm⁻¹ und rxn8832 mit −656 gegen −651. Und zwei, in
denen sie es nicht haben: rxn8837/TSopt-ab-UMA-M steht bei **−59 cm⁻¹** — eine
fast flache Mode, ein völlig anderer Punkt, obwohl die Zelle „JA" trägt. Ohne
die Frequenz sieht diese Zelle aus wie ein Erfolg.

Dasselbe bei rxn7949/TSopt mit −69 gegen −732 cm⁻¹ bei unserer Struktur.

> Die Stufe-3-Prüfung des Kapitels fängt genau diese Fälle ab. Die Frequenz
> allein tut es nicht, aber sie macht sichtbar, wo man hinsehen muss.

---

## T2 · Die Barrieren, in eV, vom Edukt aus

Nullpunkt ist das relaxierte Edukt der Referenz — geschlossenschalig in allen 45
Reaktionen und damit der einzige Punkt, über den alle Methoden einig sind.
Energien vom Stufe-1a-Einzelpunkt, also auf der Grundzustandsfläche.

`bs_ts_energies.txt` misst dieselben Größen relativ zu *unserem* Sattelpunkt und
ist damit die zirkuläre Bezugsgröße, die der Rest der Analyse vermeidet.

```
rxn          RKS-TS     UKS-NEB      unsere       UMA-S       UMA-M        eSEN     TSopt/M   Spanne
-----------------------------------------------------------------------------------------------------
rxn7949      3.402*          --       3.289          --          --          --       2.756    0.533
rxn8832      2.785*       2.460       2.473       2.501       2.485      2.491*       2.473    0.041
rxn1320      3.076*      3.034*       2.520       2.771       2.770       2.770          --    0.252
rxn4113      5.437*       4.402       4.402      5.426*      4.361*      5.426*          --    0.000
rxn8885      3.569*          --       3.593      3.251*      3.597*      3.289*          --
rxn6196      4.250*      4.242*       4.211       4.219       4.220       4.221          --    0.010
rxn0346      3.411*      3.413*       3.321      3.326*      3.322*      3.329*       3.321    0.000
rxn4518      4.799*          --       3.830       3.925       3.928       3.925          --    0.098
rxn3107      4.108*          --       4.123      4.127*       4.124      4.125*       4.123    0.001
rxn8837      3.938*       3.553       3.553      9.024*      6.908*       4.587       6.460    2.907
rxn7060      6.147*          --       6.161      6.183*      6.161*      6.207*      6.143*
rxn5691      2.976*          --       2.537          --          --          --          --
rxn1283      5.327*          --       4.772      4.820*       4.848       4.822          --    0.076
rxn8827      3.841*      3.835*       3.422      3.455*       3.443      3.441*       3.422    0.021
rxn4522      5.518*       5.061      5.610*          --          --          --          --
rxn1147      4.097*       4.118          --          --          --          --       3.880    0.239
rxn0894      4.751*       4.788       4.102      4.171*      4.824*      4.424*          --    0.686
rxn7957      3.858*       3.808          --          --          --          --       2.916    0.892
rxn5690      3.506*       3.466          --      3.487*       3.486      3.491*          --    0.020

*  nicht stationär — die Zahl ist eine Energie, keine Barriere
   Spanne = über die ungesternten Zellen
```

**Die RKS-TS-Spalte trägt in allen 19 Zeilen einen Stern.** Das ist T0 in einer
anderen Darstellung: die Zahl, gegen die der Benchmark bewertet, ist an keiner
Stelle eine Barriere im eigentlichen Sinn.

**Der Bereich.** Die Barrieren liegen zwischen 2.5 und 6.2 eV — hoch, weil es
Umlagerungen kleiner Moleküle sind. Der Größenordnungsvergleich, den das Kapitel
braucht: die Modelluneinigkeit von bis zu 4.4 eV ist damit nicht ein Fehler
*auf* einer Barriere, sondern in derselben Größe wie die Barriere selbst.

**Wo gültige Sattelpunkte auseinanderliegen:**

```
rxn8837   2.907 eV     rxn1320   0.252 eV
rxn7957   0.892        rxn1147   0.239
rxn0894   0.686        rxn4518   0.098
rxn7949   0.533        rxn1283   0.076
```

rxn8837 ist der Extremfall und erklärt sich aus T1: die TS-Optimierung ab UMA-M
landet auf einem Punkt mit −59 cm⁻¹ und 2.9 eV höher. Ein Sattelpunkt, aber
nicht dieser.

**Die restlichen elf liegen unter 0.1 eV oder haben nur einen Kandidaten.** Wo
mehrere Methoden einen gültigen Sattelpunkt finden, sind sie sich meist einig —
dieselbe Aussage wie in §4.2 des Kapitels, hier in Energie statt in RMSD.

---

## T3 · Welche Reaktionen das sind

RDKit ist auf dem Cluster nicht installiert, also keine SMILES. Stattdessen
Summenformel und die Atompaare, deren Abstand sich zwischen Edukt und Produkt am
stärksten ändert — beschränkt auf Paare, die auf mindestens einer Seite eine
Bindung sind.

```
rxn      Formel      N   Änderung Edukt → Produkt [Å]
------------------------------------------------------------------------------------------
rxn7949  C5H5NO     12   C3-C5 knüpft 2.54→1.47   C4-C5 bricht 1.44→2.46   C2-C3 bricht 1.50→2.50
rxn8832  C5H5NO     12   C1-C6 knüpft 2.56→1.48   C1-C2 bricht 1.47→2.47   C4-C6 bricht 1.51→2.34
rxn1320  C3H5NO2    11   C2-H6 knüpft 2.89→1.09   O0-H6 bricht 0.96→2.56   C1-C2 bricht 1.54→2.95
rxn4113  C3H5NO2    11   O0-C3 knüpft 3.48→1.43   N2-C3 bricht 1.45→3.48
rxn8885  C5H5NO     12   C1-O2 bricht 1.42→2.67   C1-N6 knüpft 2.64→1.57   O2-C5 knüpft 2.37→1.32
rxn6196  C5H5NO     12   C2-C5 bricht 1.47→4.12   C2-H10 bricht 1.09→3.42  C5-H10 knüpft 2.08→1.08
rxn0346  C3H5NO2    11   C5-H10 bricht 1.09→2.78  C2-C5 bricht 1.50→2.68   C2-H10 knüpft 2.21→1.09
rxn4518  C3H5NO2    11   N0-O5 knüpft 3.33→1.43   N0-C1 bricht 1.44→2.92   C2-C4 knüpft 2.04→1.93
rxn3107  C3H5NO2    11   C2-O3 bricht 1.41→2.65   C2-N5 knüpft 2.62→1.56   O3-C4 knüpft 2.35→1.32
rxn8837  C5H5NO     12   N0-C6 knüpft 3.58→1.46   C4-C6 bricht 1.51→2.34
rxn7060  C5H5NO     12   O0-C1 bricht 1.19→5.08   O0-C5 knüpft 4.48→1.17   C3-N6 knüpft 2.60→1.44
rxn5691  C5H5NO     12   O0-N6 knüpft 2.83→1.44   C4-N6 bricht 1.46→2.33
rxn1283  C3H5NO2    11   C4-O5 bricht 1.42→3.85   O2-O5 knüpft 3.61→1.43   C3-O5 bricht 1.42→2.63
rxn8827  C5H5NO     12   N0-C5 knüpft 3.89→1.45   C4-C5 bricht 1.48→2.57   C2-C4 knüpft 2.04→1.97
rxn4522  C3H5NO2    11   O3-C4 bricht 1.37→3.34   N0-O3 knüpft 3.29→1.43   N0-C1 bricht 1.44→2.89
rxn1147  C3H5NO2    11   C1-C2 bricht 1.54→3.32   C1-O5 knüpft 2.54→1.43
rxn0894  C3H5NO2    11   C4-H8 knüpft 4.01→1.10   C0-H8 bricht 1.09→3.99   N2-O5 knüpft 3.84→1.44
rxn7957  C5H5NO     12   C1-H7 bricht 1.11→4.35   C5-H7 knüpft 2.70→1.08   C4-N6 knüpft 2.59→1.53
rxn5690  C5H5NO     12   C3-H8 knüpft 3.47→1.09   C1-C4 bricht 1.49→3.03   C4-H8 bricht 1.08→2.60
```

**Das ist die Tabelle mit der unangenehmsten Botschaft, und sie gehört genau
deshalb gedruckt.**

```
C5H5NO     10 Reaktionen    12 Atome
C3H5NO2     9 Reaktionen    11 Atome
```

Die neunzehn Multireferenz-Reaktionen sind **zwei Isomerienfamilien**. Es gibt
keine dritte Summenformel im Satz, keine Reaktion mit mehr als zwölf Atomen und
keine ohne Stickstoff.

Alles, was das Kapitel zeigt, ist damit an diesen beiden Gerüsten gezeigt. Der
Mechanismus — Instabilität als Prädiktor, Kraftfehler in der Größe des
Konvergenzkriteriums, Zustandsverlust im Band — ist keiner, der von der
Molekülgröße abhinge, und die Argumente übertragen sich der Sache nach. Aber
*demonstriert* ist er an zwei Familien, und das ist eine Einschränkung, die in
§4.4 gehört.

**Chemisch** zerfallen sie in zwei Typen: Ringumlagerungen mit C-C-, C-N- oder
C-O-Bindungsbruch und -knüpfung (rxn7949, rxn8832, rxn8885, rxn3107, rxn4518,
rxn8837, rxn8827, rxn5691, rxn1283, rxn4522, rxn1147, rxn7060, rxn4113) und
Wasserstofftransfers (rxn1320, rxn0346, rxn6196, rxn0894, rxn7957, rxn5690).
Dass die diradikalische Zwischenregion bei beiden auftritt, ist konsistent damit,
dass der Bruch einer σ-Bindung ohne gleichzeitige Neubildung sie erzeugt.

> **Kontrolle.** Die hier abgeleiteten reaktiven Paare stimmen mit den in der
> Pipeline gespeicherten (`reactive_bonds` in `stab_pipeline/<rxn>/result.json`)
> überein — für rxn7949 etwa C3-C5 und C4-C5 in beiden. Die Ableitung ist also
> dieselbe wie die, auf der Stufe 3 beruht.

---

## T4 · Die Größe, auf die alles bedingt ist

Alle Werte am RKS-TS, außer den beiden Produktspalten. ΔE_BS ist, wie viel
tiefer die gebrochene Lösung dort liegt.

```
rxn        N_FOD  lmin_ext   S² TS  ΔE_BS TS  Grad TS   S² Prod  ΔE_BS Prod
----------------------------------------------------------------------------
rxn7949    1.146   -0.0631   0.893    -559.6    1.686     0.000         0.0
rxn8832    1.000   -0.0493   0.870    -428.0    2.733     0.419       -84.0
rxn1320    0.968   -0.0480   0.785    -339.2    2.073    -0.000         0.0
rxn4113    0.960   -0.0085   0.140      -8.4    0.386     0.000         0.0
rxn8885    0.923   -0.0111   0.507     -42.8    2.637    -0.000        -0.0
rxn6196    0.869   -0.0069   0.216     -10.7    0.638     0.000        -0.0
rxn0346    0.847   -0.0272   0.628    -147.6    2.613    -0.000         0.0
rxn4518    0.833   -0.0778   0.842    -648.5    2.949     0.000         0.0
rxn3107    0.801   -0.0125   0.409     -38.8    1.646    -0.000        -0.0
rxn8837    0.798   -0.0440   0.741    -293.9    1.697     0.473       -69.7
rxn7060    0.788   -0.0079   0.374     -22.1    1.766     0.000        -0.0
rxn5691    0.778   -0.0290   0.629    -155.9    1.419    -0.000         0.0
rxn1283    0.769   -0.0139   0.419     -44.5    2.386    -0.000         0.0
rxn8827    0.760   -0.0110   0.338     -27.5    1.128     0.425       -35.4
rxn4522    0.731   -0.0325   0.662    -184.0    1.875    -0.000        -0.0
rxn1147    0.725   -0.0245   0.534    -105.2    1.840    -0.000        -0.0
rxn0894    0.716   -0.0401   0.580    -190.2    1.350    -0.000        -0.0
rxn7957    0.684   -0.0240   0.513     -99.8    0.901    -0.000         0.0
rxn5690    0.433         —   0.068      -1.3    0.162     0.000        -0.0
```

`lmin_ext` ist der kleinste externe Stabilitätseigenwert. Negativ heißt: die
restringierte Lösung ist kein Minimum der unrestringierten Gleichungen.

**Der Spannbereich ist groß, und das erklärt einiges.** ΔE_BS läuft von
−648.5 meV (rxn4518) bis −1.3 meV (rxn5690), also über fast drei
Größenordnungen. „Multireferenz" ist keine Eigenschaft, die man hat oder nicht
hat — die Gruppe reicht von einer tief gebrochenen Diradikalregion bis zu einem
Fall, der die Schwelle gerade eben reißt.

**rxn5690 ist der Grenzfall in jeder Spalte:** kleinstes N_FOD (0.433),
schwächste Brechung (−1.3 meV), ⟨S²⟩ von 0.068, und mit 0.162 eV/Å der einzige
RKS-TS, der nur knapp über der Stationaritätsschwelle liegt. Dass die Methoden
dort weitgehend übereinstimmen, ist kein Zufall, sondern der weiche Rand der
Gruppe.

**Die drei Spalten sind nicht redundant.** N_FOD ordnet die Reaktionen anders
als ΔE_BS: rxn4113 hat N_FOD 0.960 und bricht mit −8.4 meV fast nicht,
rxn0894 hat N_FOD 0.716 und bricht mit −190.2 meV deutlich. Das ist §1.2 des
Kapitels an einzelnen Zeilen — die beiden Deskriptoren messen verwandte, aber
verschiedene Dinge.

**Die drei gebrochenen Produkte** (rxn8832, rxn8837, rxn8827) sind dieselben,
die in `endpoint_report.txt` stehen; rxn7945 und rxn7937 fehlen hier, weil ihr
Übergangszustand stabil ist und sie deshalb nicht zu den 19 zählen.

---

## Was diese Tabellen dem Kapitel neu geben

**T0 ist der beste Einzelbefund und war nirgends aufgeschrieben.** Zwei
Gradienten an derselben Geometrie, Faktor 4 bis 63. Damit ist „die Referenz ist
ungültig" keine Anklage an eine Rechnung mehr, sondern eine präzise Aussage
darüber, welche Annahme wo nicht mehr gilt. Der Satz für das Kapitel:

> Der RKS-TS ist ein sauberer Sattelpunkt — der restringierten Fläche.
> Median 0.059 eV/Å dort, 1.697 eV/Å auf der Fläche, auf der die Reaktion
> abläuft.

**T3 zwingt zu einer Einschränkung, die bisher fehlte.** Zwei Summenformeln,
elf bis zwölf Atome. Das gehört in §4.4 neben die anderen Reichweitengrenzen.

**T1 zeigt über die Frequenz zwei Zellen, die wie Erfolge aussehen und keine
sind** — rxn8837 und rxn7949, beide TS-Opt ab UMA-M, mit −59 und −69 cm⁻¹ gegen
−819 und −732 der Vergleichsstruktur.

**T4 macht sichtbar, dass die MR-Gruppe kein Block ist**, sondern ein Verlauf
über drei Größenordnungen in der Tiefe der Brechung.

---

## T1 vervollständigt · die beiden IRC-Zellen

Vollständige Daten für die zwei Zellen, die in der ersten Fassung leer waren.
Beide bestehen alle drei Stufen.

```
Zelle                  Grad   n_imag    v_imag   Anteil    Rate   Barriere   Stufe
-----------------------------------------------------------------------------------
rxn1147 / unsere      0.010        1    -581.1     0.60   0.946      4.118   c
rxn7957 / unsere      0.011        1    -670.8     0.27   0.542      3.808   c
```

Nach Einbeziehen von `orca_irc/` **und** `freq_at_model/` fehlen noch **18 von
114** Zellen. Es sind keine Suchfehler mehr, sondern echte Abwesenheiten:

```
UKS-NEB fehlt  (7)   rxn7949 rxn8885 rxn4518 rxn3107 rxn7060 rxn5691 rxn1283
                     -- kein konvergiertes Band
TSopt/M fehlt (10)   rxn1320 rxn4113 rxn8885 rxn6196 rxn4518 rxn5691 rxn1283
                     rxn4522 rxn0894 rxn5690  -- nie gestartet
unsere fehlt   (1)   rxn5690 -- keine gebrochene Loesung vorhanden
```

---

## T5 · Kumulative Dreistufenbilanz

Pro Methode drei kumulative Zählungen statt einer. **Keine bestehende Zahl
ändert sich** — die Spalte *Stufe c* reproduziert die im Kapitel geführten
gültigen Sattelpunkte exakt.

```
Methode     geprüft   Stufe a   Stufe b   Stufe c     verloren a→b   b→c
--------------------------------------------------------------------------
UKS-NEB          12         8         8         8                0     0
unsere           18        17        16        13                1     3
UMA-S            19         7         6         6                1     0
UMA-M            19        13        12        11                1     1
eSEN             19        10         8         7                2     1
TSopt/M           9         8         8         6                0     2

a  stationär, Gradient < 0.15 eV/Å
b  zusätzlich genau eine imaginäre Mode
c  zusätzlich Modenanteil ≥ 0.10 und Bindungsrate ≥ 0.05
```

**Wo die Verluste sitzen, ist je Methode verschieden.** Bei den Modellen
zwischen Start und Stufe a — sie sind nicht stationär, alles Weitere folgt
daraus. Bei unseren Strukturen und bei TSopt/M dagegen zwischen b und c: die
Punkte sind saubere Sattelpunkte, nur nicht die dieser Reaktion.

Die sieben Zeilen, die an Stufe c scheitern, vollständig:

```
rxn1320   unsere     ν  -312.7   Anteil 0.00   Rate 0.001
rxn4518   unsere     ν   -90.2   Anteil 0.07   Rate 0.327
rxn5691   unsere     ν   -94.6   Anteil 0.60   Rate 0.015
rxn7949   UMA-M      ν  -109.3   Anteil 0.13   Rate 0.011
rxn7949   eSEN       ν  -100.1   Anteil 0.16   Rate 0.009
rxn7949   TSopt/M    ν   -69.4   Anteil 0.23   Rate 0.008
rxn8837   TSopt/M    ν   -59.4   Anteil 0.09   Rate 0.054
```

**Sechs der sieben liegen zwischen −59 und −113 cm⁻¹.** Stufe c sortiert fast
ausschließlich weiche Moden aus — die Ausnahme ist rxn1320 mit −313 cm⁻¹ und
einem Modenanteil von exakt null auf den reaktiven Atomen.

**rxn5691 fällt aus einem anderen Grund und ist bisher nirgends vermerkt:**
Anteil 0.60, also viel Bewegung auf den richtigen Atomen, aber Rate 0.015 — die
Atome bewegen sich *gemeinsam*, ohne die Bindung zu dehnen. Eine Verkippung,
keine Dissoziation.

> **Warum diese Tabelle nicht neu gerechnet ist.** Eine erste Fassung las ORCAs
> `$normal_modes` (kartesische Auslenkungen) und wich in 6 von rund 30 Urteilen
> von `sweep_summary.txt` ab. `sweep_summary.py` massengewichtet die Hesse,
> projiziert Translationen und Rotationen heraus und normiert erst dann — bei
> Wasserstoff gegen Kohlenstoff ist das Faktor 3.5 in der Amplitude. Die
> Schwellen 0.10 und 0.05 sind auf die massengewichtete Größe kalibriert.
> `chapter_tables2.py` führt deshalb den Definitionsteil von `sweep_summary.py`
> aus und ruft dessen Funktionen auf. **Abgleich: 0 Abweichungen.**

---

## T6 · Triage, gegen Stufe c neu ausgezählt

Die Behauptung des Kapitels lautet „unter 0.25 eV/Å → 7 von 7". Gegen Stufe c
geprüft stimmt sie nicht.

```
Lauf                     Start   Ergebnis    v_imag   Anteil    Rate   Stufe
------------------------------------------------------------------------------
tsopt_rxn0346_UMA-M      0.182      0.009   -1295.1     0.67   1.031   c
tsopt_rxn1147_UMA-M      0.050      0.013    -229.7     0.18   0.073   c
tsopt_rxn3107_UMA-M      0.106      0.016   -1483.6     0.40   0.598   c
tsopt_rxn7949_UMA-M      0.051      0.012     -69.4     0.23   0.008   b  ←
tsopt_rxn7957_UMA-M      0.111      0.013    -624.3     0.70   0.987   c
tsopt_rxn8827_UMA-M      0.131      0.008    -592.0     0.97   1.390   c
tsopt_rxn8832_UMA-M      0.077      0.018    -652.1     0.96   1.217   c
------------------------------------------------------------------------------
tsopt_rxn0894_UMA-S      0.776      0.010      -0.0     0.14   0.163   a
tsopt_rxn7060_UMA-M      0.335      1.707         —        —       —   none
tsopt_rxn8837_UMA-M      0.757      0.014     -59.4     0.09   0.054   b
```

```
unter 0.25 eV/Å    6 von 7      (bisher im Kapitel: 7 von 7)
über  0.25 eV/Å    0 von 3      unverändert
```

**Der Zähler ist 6, nicht 7.** `tsopt_rxn7949_UMA-M` startet bei 0.051 eV/Å,
liegt also klar im unteren Eimer, und konvergiert auf einen Punkt mit −69 cm⁻¹
und Bindungsrate 0.008 — ein Torsionssattel im Eduktbecken (beide reaktiven
Bindungen stehen dort auf Eduktwerten).

Der Nenner bleibt 7 und 3. Die Aussage der Triage überlebt, mit kleinerer Marge:
unterhalb der Schwelle lohnt die Nachoptimierung in sechs von sieben Fällen,
oberhalb in keinem.

---

## T7 · Wie weit der UKS-NEB vom RKS-TS landet

All-Atom-Kabsch-RMSD, alle 18 Reaktionen mit einer konvergierten
BS-NEB-Struktur.

```
rxn        NEB vs RKS-TS   NEB vs unsere   Grad NEB
-----------------------------------------------------
rxn0346           0.0116          0.1763      2.553
rxn8827           0.0191          0.3656      1.074
rxn1320           0.0268          1.0721      2.062
rxn6196           0.0552          0.1336      0.683
rxn7936           0.0054               —          —
rxn1150           0.0950               —      0.017
rxn7957           0.1076          0.0194      0.009
rxn5691           0.1419          0.5979          —
rxn1147           0.1571          0.0218      0.016
rxn5690           0.2881               —      0.004
rxn8832           0.2908          0.1067      0.018
rxn4522           0.5193          0.4952      0.009
rxn8837           0.5388          0.0030      0.011
rxn7945           0.6686               —      0.025
rxn1283           0.8680          0.3477          —
rxn0894           0.8876          1.1425      0.022
rxn3107           0.9057          0.8725          —
rxn4113           0.9319          0.0080      0.007

n = 18   Median 0.2226 Å   unter 0.10 Å: 6
```

**Das korrigiert eine Verallgemeinerung, die ich zu früh gezogen hatte.** Aus
rxn0346, rxn6196 und rxn8827 hatte ich geschlossen, ein restringiert laufendes
Band konvergiere gegen den restringierten Sattelpunkt. Über alle 18 stimmt das
nicht — der Median liegt bei 0.22 Å und vier Fälle liegen über 0.85 Å.

Sortiert man stattdessen nach dem **Gradienten des NEB-Ergebnisses**, wird das
Muster scharf:

```
Band nicht konvergiert (Grad > 0.15)    rxn0346 rxn8827 rxn1320 rxn6196
   Abstand zum RKS-TS   0.012 – 0.055 Å      4 von 4 unter 0.06

Band konvergiert (Grad < 0.15)          10 Reaktionen
   Abstand zum RKS-TS   0.005 – 0.932 Å      Median 0.29
```

> **Ein Band, das scheitert, bleibt auf dem RKS-TS liegen** — alle vier, unter
> 0.06 Å. Wo es konvergiert, geht es woanders hin, und dann meist dorthin, wo
> auch unsere Struktur liegt (rxn8837 0.003, rxn4113 0.008, rxn7957 0.019,
> rxn1147 0.022).

Das ist die genauere Fassung: nicht „der kollabierte NEB reproduziert den
RKS-TS", sondern „der **nicht konvergierte** NEB bleibt am Startpunkt stehen,
und dieser Startpunkt ist der RKS-TS". Der Kollaps zeigt sich nicht im
Endpunkt, sondern darin, dass es keinen gibt.

---

## Erzeugung

```
pipeline/chapter_tables.py    →   chapter_tables.txt     (T0 – T4)
pipeline/chapter_tables2.py   →   chapter_tables2.txt    (T1-Ergänzung, T5 – T7)
```

Quellen je Spalte:

| Größe | Quelle |
|---|---|
| Gradienten auf beiden Flächen | `stab_pipeline/<rxn>/result.json`, Eintrag `RKS-ref`, Felder `rks_grad.max_evang` und `bs.bs_grad.max_evang` |
| Energien der Kandidaten | `orca_freq/<label>/bs_sp.out`, Stufe 1a mit Stabilitätsanalyse |
| Gradienten der Kandidaten | `orca_freq/<label>/engrad.engrad`, Stufe 1b |
| Frequenzen | `orca_freq/<label>/numfreq.out`, Stufe 2 |
| Eduktenergie | `orca_endpoint/<rxn>_reactant/sp.out` |
| ⟨S²⟩ und ΔE_BS | `endpoint_report.txt` |
| λ_min_ext | `_collected_stability.json`, `eigs/<rxn>/ext` |
| N_FOD | `cheap_stab_report.txt` |
| Geometrien für T3 | `orca_neb_results/<rxn>/{reactant,product}.xyz` |

**Labelkonvention, die eine Falle enthält:** `nebts_<rxn>` in `orca_freq` ist
**nicht** der Benchmark-Übergangszustand, sondern unser BS-NEB-Ergebnis
(`make_freq_list.py` baut es aus `bs_uks_neb_results/<rxn>/*NEB-TS_converged.xyz`).
Der RKS-TS hat nie eine Hesse bekommen und liegt gar nicht in `orca_freq`. Wer
die Spalte verwechselt, bekommt einen RKS-TS, der in acht Reaktionen ein
gültiger Sattelpunkt zu sein scheint.
