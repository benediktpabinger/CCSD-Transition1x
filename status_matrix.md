# Statusmatrix — Stufe 1, 2 und 3 für jede Struktur jeder Reaktion

Erzeugt von `pipeline/status_matrix.py` aus den gespeicherten Ergebnissen.

## Wie die Felder zu lesen sind

| Eintrag | Bedeutung |
|---|---|
| `bestanden` | die Prüfung lief und die Struktur besteht |
| `DURCHGEFALLEN` | die Prüfung lief und die Struktur besteht nicht |
| `NICHT GEPRUEFT` | niemand hat sie je gerechnet |
| `laeuft` | eine Rechnung dafür steht gerade in der Warteschlange |
| `—` | die Struktur existiert nicht |

**Der Unterschied zwischen `DURCHGEFALLEN` und `NICHT GEPRUEFT` ist der
Grund für dieses Dokument.** Frühere Tabellen haben beides gleich
dargestellt, wodurch ungeprüfte Kandidaten wie widerlegte aussahen.

## Schwellen

| Stufe | Kriterium | Schwelle |
|---|---|---|
| 1 | ist der Punkt stationär? | Gradient < 0.05 eV/Å `stationaer`, < 0.15 `nahe`, darüber `NICHT STATIONAER`. Unsere bestätigten Sattelpunkte liegen in ORCA bei 0.006–0.011 |
| 2 | Sattelpunkt erster Ordnung? | genau eine Mode unter −20 cm⁻¹, nach Herausprojizieren von Translation und Rotation |
| 3 | gehört er zu dieser Reaktion? | Modenanteil ≥ 0.10 auf den vier reaktiven Atomen **und** Bindungsrate ≥ 0.05 |

Die **Bindungslängen** werden ausgegeben und nicht mit einer Schwelle
versehen: eine reaktive Bindung, die bereits ihren normalen Wert hat, zeigt
an, dass die Reaktion dort abgeschlossen ist — aber eine Schwelle dafür an
den zwei Fällen zu kalibrieren, die sie entscheiden soll, wäre zirkulär.

---

# Multireferenz — die 19

## rxn7949   N_FOD 1.146   ΔE_BS an der Referenz -559.6 meV

reaktive Bindungen: C3-C5, C4-C5

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, batch, converged / 0.836 | 0.001 stationaer | +535 | **bestanden** 1 @ -735 | **bestanden** 0.98 / 1.157 0.315 | 1.84 1.50 | PySCF |
| UMA-S | BS / 0.997 | 0.248 NICHT STATIONAER | +15 | **bestanden** 1 @ -96 | **DURCHGEFALLEN** 0.19 / 0.008 0.007 | 2.36 1.48 | PySCF |
| UMA-M | BS / 1.000 | 0.051 nahe | +0 | **bestanden** 1 @ -109 | **DURCHGEFALLEN** 0.13 / 0.003 0.011 | 2.40 1.48 | PySCF |
| eSEN | BS / 1.001 | 0.074 nahe | +0 | **bestanden** 1 @ -100 | **DURCHGEFALLEN** 0.16 / 0.008 0.009 | 2.41 1.48 | PySCF |
| tsopt UMA-M | TS-Opt ab Modellgeometrie / 0.993 | NICHT GEPRUEFT | — | NICHT GEPRUEFT | NICHT GEPRUEFT | 2.41 1.48 | TS-Opt ab Modellgeometrie |

## rxn8832   N_FOD 1.000   ΔE_BS an der Referenz -428.0 meV

reaktive Bindungen: C1-C6, C1-C2

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, v2, converged / 1.001 | 0.006 stationaer | +0 | **bestanden** 1 @ -653 | **bestanden** 0.96 / 1.216 0.109 | 1.74 1.48 | PySCF |
| UMA-S | BS / 0.982 | 0.096 nahe | +28 | laeuft | laeuft | 1.73 1.48 | BS |
| UMA-M | BS / 0.990 | 0.075 nahe | +12 | laeuft | laeuft | 1.73 1.48 | BS |
| eSEN | BS / 0.988 | 0.232 NICHT STATIONAER | +18 | laeuft | laeuft | 1.75 1.48 | BS |
| NEB-TS | ORCA BS-NEB / — | NICHT GEPRUEFT | — | NICHT GEPRUEFT | NICHT GEPRUEFT | 1.74 1.48 | ORCA BS-NEB |
| tsopt UMA-M | TS-Opt ab Modellgeometrie / 1.001 | NICHT GEPRUEFT | — | NICHT GEPRUEFT | NICHT GEPRUEFT | 1.74 1.48 | TS-Opt ab Modellgeometrie |

## rxn1320   N_FOD 0.968   ΔE_BS an der Referenz -339.2 meV

reaktive Bindungen: C2-H6, O0-H6

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, batch, converged / 1.019 | 0.009 stationaer | +0 | **bestanden** 1 @ -313 | **DURCHGEFALLEN** 0.00 / 0.000 0.001 | 3.36 0.97 | ORCA |
| UMA-S | BS / 0.690 | 0.067 nahe | +252 | **bestanden** 1 @ -416 | **bestanden** 0.37 / 0.213 0.010 | 2.61 0.97 | ORCA |
| UMA-M | BS / 0.697 | 0.043 stationaer | +251 | **bestanden** 1 @ -405 | **bestanden** 0.36 / 0.211 0.010 | 2.60 0.97 | ORCA |
| eSEN | BS / 0.688 | 0.115 nahe | +251 | **bestanden** 1 @ -408 | **bestanden** 0.36 / 0.215 0.010 | 2.61 0.97 | ORCA |
| NEB-TS | ORCA BS-NEB / — | NICHT GEPRUEFT | — | NICHT GEPRUEFT | NICHT GEPRUEFT | 1.98 0.99 | ORCA BS-NEB |

## rxn4113   N_FOD 0.960   ΔE_BS an der Referenz -8.4 meV

reaktive Bindungen: O0-C3, N2-C3

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, fromneb, converged / 0.969 | 0.010 stationaer | +40 | **bestanden** 1 @ -156 | **bestanden** 0.75 / 0.084 0.980 | 3.70 3.14 | PySCF |
| UMA-S | BS / 0.149 | 0.169 NICHT STATIONAER | +1065 | **bestanden** 1 @ -876 | **bestanden** 0.53 / 0.704 0.393 | 2.27 2.50 | ORCA |
| UMA-M | BS / 1.007 | 0.186 NICHT STATIONAER | +0 | **bestanden** 1 @ -49 | **bestanden** 0.75 / 0.621 0.500 | 3.47 3.56 | ORCA |
| eSEN | BS / 0.187 | 0.182 NICHT STATIONAER | +1065 | laeuft | laeuft | 2.28 2.52 | BS |
| NEB-TS | ORCA BS-NEB / — | NICHT GEPRUEFT | — | NICHT GEPRUEFT | NICHT GEPRUEFT | 3.71 3.15 | ORCA BS-NEB |

## rxn8885   N_FOD 0.923   ΔE_BS an der Referenz -42.8 meV

reaktive Bindungen: C1-O2, C1-N6

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, v2, BS_LOST / 0.153 | 0.001 stationaer | +342 | **bestanden** 1 @ -1685 | **bestanden** 0.40 / 0.595 0.024 | 2.09 2.48 | PySCF |
| UMA-S | BS / 1.024 | 0.484 NICHT STATIONAER | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | 3.33 2.57 | BS |
| UMA-M | BS / 0.175 | 0.190 NICHT STATIONAER | +346 | NICHT GEPRUEFT | NICHT GEPRUEFT | 2.09 2.48 | BS |
| eSEN | BS / 1.028 | 0.375 NICHT STATIONAER | +38 | NICHT GEPRUEFT | NICHT GEPRUEFT | 5.57 2.56 | BS |

## rxn6196   N_FOD 0.869   ΔE_BS an der Referenz -10.7 meV

reaktive Bindungen: C2-C5, C2-H10

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, fromneb, converged / 0.493 | 0.006 stationaer | +0 | **bestanden** 1 @ -784 | **bestanden** 0.96 / 1.263 0.341 | 2.01 2.19 | PySCF |
| UMA-S | BS / 0.519 | 0.092 nahe | +8 | laeuft | laeuft | 2.00 2.20 | BS |
| UMA-M | BS / 0.498 | 0.143 nahe | +9 | laeuft | laeuft | 2.02 2.23 | BS |
| eSEN | BS / 0.513 | 0.140 nahe | +10 | laeuft | laeuft | 2.02 2.23 | BS |
| NEB-TS | ORCA BS-NEB / — | NICHT GEPRUEFT | — | NICHT GEPRUEFT | NICHT GEPRUEFT | 1.89 1.93 | ORCA BS-NEB |

## rxn0346   N_FOD 0.847   ΔE_BS an der Referenz -147.6 meV

reaktive Bindungen: C5-H10, C2-C5

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, batch, BS_LOST / 0.594 | 0.018 stationaer | +0 | **bestanden** 1 @ -1289 | **bestanden** 0.68 / 0.147 1.038 | 2.24 1.88 | ORCA |
| UMA-S | BS / 0.607 | 0.241 NICHT STATIONAER | +5 | **bestanden** 1 @ -1253 | **bestanden** 0.71 / 0.123 1.054 | 2.20 1.87 | ORCA |
| UMA-M | BS / 0.608 | 0.182 NICHT STATIONAER | +2 | **bestanden** 1 @ -1334 | **bestanden** 0.68 / 0.145 1.029 | 2.23 1.87 | ORCA |
| eSEN | BS / 0.652 | 0.482 NICHT STATIONAER | +9 | **bestanden** 1 @ -1214 | **bestanden** 0.74 / 0.125 1.080 | 2.17 1.85 | ORCA |
| NEB-TS | ORCA BS-NEB / — | NICHT GEPRUEFT | — | NICHT GEPRUEFT | NICHT GEPRUEFT | 1.85 1.65 | ORCA BS-NEB |
| tsopt UMA-M | TS-Opt ab Modellgeometrie / 0.598 | NICHT GEPRUEFT | — | NICHT GEPRUEFT | NICHT GEPRUEFT | 2.25 1.88 | TS-Opt ab Modellgeometrie |

## rxn4518   N_FOD 0.833   ΔE_BS an der Referenz -648.5 meV

reaktive Bindungen: N0-O5, N0-C1

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, batch, converged / 1.009 | 0.010 stationaer | +0 | **bestanden** 1 @ -89 | **DURCHGEFALLEN** 0.04 / 0.039 0.207 | 3.33 3.27 | PySCF |
| UMA-S | BS / 1.009 | 0.059 nahe | +96 | **DURCHGEFALLEN** 2 @ -86 | **DURCHGEFALLEN** 0.07 / 0.314 0.054 | 2.99 4.48 | ORCA |
| UMA-M | BS / 1.008 | 0.057 nahe | +98 | **bestanden** 1 @ -105 | **bestanden** 0.18 / 0.134 0.438 | 2.94 4.46 | ORCA |
| eSEN | BS / 1.009 | 0.050 nahe | +96 | **DURCHGEFALLEN** 2 @ -133 | **DURCHGEFALLEN** 0.03 / 0.170 0.006 | 3.11 4.63 | ORCA |

## rxn3107   N_FOD 0.801   ΔE_BS an der Referenz -38.8 meV

reaktive Bindungen: C2-O3, C2-N5

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, v2, BS_LOST / 0.174 | 0.018 stationaer | +0 | **bestanden** 1 @ -1486 | **bestanden** 0.40 / 0.595 0.071 | 2.09 2.46 | ORCA |
| UMA-S | BS / 0.201 | 0.157 NICHT STATIONAER | +4 | **bestanden** 1 @ -1440 | **bestanden** 0.43 / 0.605 0.054 | 2.09 2.46 | ORCA |
| UMA-M | BS / 0.171 | 0.106 nahe | +1 | **bestanden** 1 @ -1438 | **bestanden** 0.41 / 0.599 0.078 | 2.09 2.46 | ORCA |
| eSEN | BS / 0.142 | 0.163 NICHT STATIONAER | +2 | **bestanden** 1 @ -1491 | **bestanden** 0.41 / 0.606 0.081 | 2.08 2.46 | ORCA |
| NEB-CI | nur Bandpunkt, kein Sattelpunkt / — | NICHT GEPRUEFT | — | NICHT GEPRUEFT | NICHT GEPRUEFT | 2.06 2.48 | nur Bandpunkt, kein Sattelpunkt |
| tsopt UMA-M | TS-Opt ab Modellgeometrie / 0.178 | NICHT GEPRUEFT | — | NICHT GEPRUEFT | NICHT GEPRUEFT | 2.09 2.46 | TS-Opt ab Modellgeometrie |

## rxn8837   N_FOD 0.798   ΔE_BS an der Referenz -293.9 meV

reaktive Bindungen: N0-C6, C4-C6

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, batch, converged / 1.039 | 0.008 stationaer | +0 | **bestanden** 1 @ -823 | **bestanden** 0.91 / 1.310 0.056 | 2.05 2.33 | PySCF |
| UMA-S | BS / 1.010 | 0.757 NICHT STATIONAER | +5469 | NICHT GEPRUEFT | NICHT GEPRUEFT | 6.17 1.48 | BS |
| UMA-M | BS / 1.007 | 0.764 NICHT STATIONAER | +3352 | NICHT GEPRUEFT | NICHT GEPRUEFT | 4.15 1.28 | BS |
| eSEN | RKS / 0.000 | 0.101 nahe | +1034 | NICHT GEPRUEFT | NICHT GEPRUEFT | 3.98 1.28 | RKS |
| NEB-TS | ORCA BS-NEB / — | NICHT GEPRUEFT | — | NICHT GEPRUEFT | NICHT GEPRUEFT | 2.05 2.33 | ORCA BS-NEB |
| tsopt UMA-M | TS-Opt ab Modellgeometrie / 1.006 | NICHT GEPRUEFT | — | NICHT GEPRUEFT | NICHT GEPRUEFT | 6.10 1.27 | TS-Opt ab Modellgeometrie |

## rxn7060   N_FOD 0.788   ΔE_BS an der Referenz -22.1 meV

reaktive Bindungen: O0-C1, O0-C5

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, v2, BS_LOST / 0.047 | 0.007 stationaer | +0 | **bestanden** 1 @ -2498 | **bestanden** 0.58 / 0.457 0.582 | 1.62 1.29 | PySCF |
| UMA-S | RKS / -0.000 | 0.884 NICHT STATIONAER | +22 | laeuft | laeuft | 1.59 1.28 | RKS |
| UMA-M | RKS / -0.000 | 0.335 NICHT STATIONAER | +0 | laeuft | laeuft | 1.62 1.29 | RKS |
| eSEN | RKS / -0.000 | 1.126 NICHT STATIONAER | +46 | laeuft | laeuft | 1.59 1.28 | RKS |
| tsopt UMA-M | TS-Opt ab Modellgeometrie / — | NICHT GEPRUEFT | — | NICHT GEPRUEFT | NICHT GEPRUEFT | 1.60 1.29 | TS-Opt ab Modellgeometrie |

## rxn5691   N_FOD 0.778   ΔE_BS an der Referenz -155.9 meV

reaktive Bindungen: O0-N6, C4-N6

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, batch, converged / 0.973 | 0.009 stationaer | +164 | **bestanden** 1 @ -102 | **DURCHGEFALLEN** 0.58 / 0.014 0.009 | 2.91 2.51 | PySCF |
| UMA-S | BS / 1.011 | 0.154 NICHT STATIONAER | +0 | **bestanden** 1 @ -223 | **bestanden** 0.45 / 0.031 0.527 | 2.97 2.46 | PySCF |
| UMA-M | BS / 1.017 | 0.085 nahe | +63 | **DURCHGEFALLEN** 2 @ -237 | **bestanden** 0.35 / 0.483 0.379 | 2.81 2.49 | PySCF |
| eSEN | BS / 1.011 | 0.068 nahe | +1 | **bestanden** 1 @ -231 | **bestanden** 0.44 / 0.057 0.530 | 2.97 2.46 | PySCF |
| NEB-CI | nur Bandpunkt, kein Sattelpunkt / — | NICHT GEPRUEFT | — | NICHT GEPRUEFT | NICHT GEPRUEFT | 2.21 2.17 | nur Bandpunkt, kein Sattelpunkt |

## rxn1283   N_FOD 0.769   ΔE_BS an der Referenz -44.5 meV

reaktive Bindungen: C4-O5, O2-O5

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, v2, BS_LOST / 0.985 | NICHT GEPRUEFT | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | 2.80 3.32 | PySCF, v2, BS_LOST |
| UMA-S | BS / 0.978 | 0.162 NICHT STATIONAER | +48 | **bestanden** 1 @ -80 | **bestanden** 0.82 / 0.603 0.975 | 3.21 2.33 | ORCA |
| UMA-M | BS / 0.992 | 0.141 nahe | +82 | **bestanden** 1 @ -94 | **bestanden** 0.84 / 0.980 0.671 | 3.00 2.29 | ORCA |
| eSEN | BS / 0.987 | 0.105 nahe | +50 | **bestanden** 1 @ -73 | **bestanden** 0.83 / 0.624 0.958 | 3.20 2.39 | ORCA |
| NEB-CI | nur Bandpunkt, kein Sattelpunkt / — | NICHT GEPRUEFT | — | NICHT GEPRUEFT | NICHT GEPRUEFT | 3.94 3.58 | nur Bandpunkt, kein Sattelpunkt |

## rxn8827   N_FOD 0.760   ΔE_BS an der Referenz -27.5 meV

reaktive Bindungen: N0-C5, C4-C5

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, batch, converged / 1.024 | 0.003 stationaer | +0 | **bestanden** 1 @ -596 | **bestanden** 0.97 / 1.389 0.094 | 2.03 2.58 | PySCF |
| UMA-S | BS / 1.001 | 0.176 NICHT STATIONAER | +34 | laeuft | laeuft | 2.02 2.56 | BS |
| UMA-M | BS / 1.007 | 0.131 nahe | +21 | laeuft | laeuft | 2.02 2.57 | BS |
| eSEN | BS / 1.009 | 0.228 NICHT STATIONAER | +20 | laeuft | laeuft | 2.04 2.57 | BS |
| NEB-TS | ORCA BS-NEB / — | NICHT GEPRUEFT | — | NICHT GEPRUEFT | NICHT GEPRUEFT | 1.96 2.32 | ORCA BS-NEB |
| tsopt UMA-M | TS-Opt ab Modellgeometrie / 1.024 | NICHT GEPRUEFT | — | NICHT GEPRUEFT | NICHT GEPRUEFT | 2.03 2.58 | TS-Opt ab Modellgeometrie |

## rxn4522   N_FOD 0.731   ΔE_BS an der Referenz -184.0 meV

reaktive Bindungen: O3-C4, N0-O3

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, batch, BS_LOST / 0.000 | NICHT GEPRUEFT | +1845 | NICHT GEPRUEFT | NICHT GEPRUEFT | 1.85 2.07 | PySCF, batch, BS_LOST |
| UMA-S | BS / 1.005 | 0.075 nahe | +0 | **bestanden** 1 @ -83 | **bestanden** 0.34 / 0.017 0.357 | 1.39 2.88 | PySCF |
| UMA-M | BS / 1.005 | 0.083 nahe | +4 | **bestanden** 1 @ -81 | **bestanden** 0.36 / 0.017 0.366 | 1.39 2.88 | PySCF |
| eSEN | BS / 1.005 | 0.073 nahe | +2 | **bestanden** 1 @ -80 | **bestanden** 0.36 / 0.017 0.357 | 1.39 2.86 | PySCF |
| NEB-TS | ORCA BS-NEB / — | NICHT GEPRUEFT | — | NICHT GEPRUEFT | NICHT GEPRUEFT | 2.93 2.92 | ORCA BS-NEB |

## rxn1147   N_FOD 0.725   ΔE_BS an der Referenz -105.2 meV

reaktive Bindungen: C1-C2, C1-O5

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, batch, converged / 0.456 | 0.002 stationaer | +234 | **bestanden** 1 @ -591 | **bestanden** 0.60 / 0.134 0.943 | 3.20 1.86 | PySCF |
| UMA-S | RKS / -0.000 | 0.081 nahe | +0 | **bestanden** 1 @ -253 | **bestanden** 0.24 / 0.056 0.059 | 3.57 1.50 | ORCA |
| UMA-M | RKS / 0.000 | 0.050 stationaer | +3 | **bestanden** 1 @ -248 | **bestanden** 0.22 / 0.057 0.071 | 3.57 1.50 | PySCF |
| eSEN | RKS / 0.000 | 0.068 nahe | +2 | **bestanden** 1 @ -250 | **bestanden** 0.23 / 0.055 0.067 | 3.56 1.50 | PySCF |
| NEB-TS | ORCA BS-NEB / — | NICHT GEPRUEFT | — | NICHT GEPRUEFT | NICHT GEPRUEFT | 3.21 1.87 | ORCA BS-NEB |
| tsopt UMA-M | TS-Opt ab Modellgeometrie / — | NICHT GEPRUEFT | — | NICHT GEPRUEFT | NICHT GEPRUEFT | 3.59 1.50 | TS-Opt ab Modellgeometrie |

## rxn0894   N_FOD 0.716   ΔE_BS an der Referenz -190.2 meV

reaktive Bindungen: C4-H8, C0-H8

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, batch, converged / 0.816 | 0.016 stationaer | +0 | **bestanden** 1 @ -617 | **bestanden** 0.58 / 0.303 1.036 | 1.17 1.91 | ORCA |
| UMA-S | BS / 0.986 | 0.776 NICHT STATIONAER | +68 | **bestanden** 1 @ -241 | **bestanden** 0.44 / 0.127 0.580 | 1.15 2.12 | ORCA |
| UMA-M | ? / 1.038 | 1.320 NICHT STATIONAER | +715 | **bestanden** 1 @ -52 | **bestanden** 0.80 / 0.120 0.214 | 6.61 10.05 | ORCA |
| eSEN | BS / 1.028 | 0.794 NICHT STATIONAER | +319 | **bestanden** 1 @ -39 | **bestanden** 0.87 / 0.629 0.680 | 6.85 10.37 | ORCA |
| NEB-TS | ORCA BS-NEB / — | NICHT GEPRUEFT | — | NICHT GEPRUEFT | NICHT GEPRUEFT | 1.68 4.23 | ORCA BS-NEB |
| tsopt UMA-S | TS-Opt ab Modellgeometrie / 1.024 | NICHT GEPRUEFT | — | NICHT GEPRUEFT | NICHT GEPRUEFT | 1.11 2.70 | TS-Opt ab Modellgeometrie |

## rxn7957   N_FOD 0.684   ΔE_BS an der Referenz -99.8 meV

reaktive Bindungen: C1-H7, C5-H7

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, v2, converged / 0.709 | 0.001 stationaer | +890 | **bestanden** 1 @ -677 | **bestanden** 0.27 / 0.544 0.061 | 2.46 1.12 | PySCF |
| UMA-S | BS / 0.734 | 0.137 nahe | +0 | **bestanden** 1 @ -436 | **bestanden** 0.55 / 0.846 0.309 | 1.89 1.17 | PySCF |
| UMA-M | BS / 0.688 | 0.111 nahe | +0 | **bestanden** 1 @ -696 | **bestanden** 0.75 / 1.015 0.569 | 1.87 1.19 | ORCA |
| eSEN | BS / 0.731 | 0.109 nahe | +0 | **bestanden** 1 @ -459 | **bestanden** 0.57 / 0.870 0.339 | 1.88 1.17 | PySCF |
| NEB-TS | ORCA BS-NEB / — | NICHT GEPRUEFT | — | NICHT GEPRUEFT | NICHT GEPRUEFT | 2.48 1.12 | ORCA BS-NEB |
| tsopt UMA-M | TS-Opt ab Modellgeometrie / 0.696 | NICHT GEPRUEFT | — | NICHT GEPRUEFT | NICHT GEPRUEFT | 1.86 1.18 | TS-Opt ab Modellgeometrie |

## rxn5690   N_FOD 0.433   ΔE_BS an der Referenz -1.3 meV

reaktive Bindungen: C3-H8, C1-C4

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | BS / 0.302 | 0.184 NICHT STATIONAER | +1 | laeuft | laeuft | 1.11 1.83 | BS |
| UMA-M | BS / 0.315 | 0.110 nahe | +0 | laeuft | laeuft | 1.11 1.83 | BS |
| eSEN | BS / 0.287 | 0.154 NICHT STATIONAER | +6 | laeuft | laeuft | 1.11 1.83 | BS |
| NEB-TS | ORCA BS-NEB / — | NICHT GEPRUEFT | — | NICHT GEPRUEFT | NICHT GEPRUEFT | 1.18 1.78 | ORCA BS-NEB |

---

# Einfach — die 26 (Kontrollgruppe)

## rxn7945   N_FOD 0.903

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.059 nahe | +514 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.081 nahe | +513 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.135 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| NEB-TS | ORCA BS-NEB / — | NICHT GEPRUEFT | — | NICHT GEPRUEFT | NICHT GEPRUEFT | — | ORCA BS-NEB |

## rxn7937   N_FOD 0.877

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.093 nahe | +2 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.033 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.225 NICHT STATIONAER | +5 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn1150   N_FOD 0.847

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.083 nahe | +1 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.071 nahe | +1 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.097 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| NEB-TS | ORCA BS-NEB / — | NICHT GEPRUEFT | — | NICHT GEPRUEFT | NICHT GEPRUEFT | — | ORCA BS-NEB |

## rxn0896   N_FOD 0.840

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.064 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.082 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.119 nahe | +1 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn7936   N_FOD 0.727

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.111 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.065 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.166 NICHT STATIONAER | +2 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| NEB-CI | nur Bandpunkt, kein Sattelpunkt / — | NICHT GEPRUEFT | — | NICHT GEPRUEFT | NICHT GEPRUEFT | — | nur Bandpunkt, kein Sattelpunkt |

## rxn0101   N_FOD 0.713

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.138 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.097 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.099 nahe | +1 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn10005   N_FOD 0.695

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.051 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.030 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.069 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn10054   N_FOD 0.695

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | BS / 0.306 | 0.591 NICHT STATIONAER | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | BS |
| UMA-M | BS / 0.080 | 0.066 nahe | +3 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | BS |
| eSEN | BS / 0.066 | 0.125 nahe | +3 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | BS |

## rxn1154   N_FOD 0.566

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.126 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.044 stationaer | +1 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.107 nahe | +12 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn4513   N_FOD 0.307

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.158 NICHT STATIONAER | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.153 NICHT STATIONAER | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.150 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn7955   N_FOD 0.219

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.056 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.041 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.052 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn4519   N_FOD 0.154

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.050 stationaer | +1 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.035 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.068 nahe | +1 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn4500   N_FOD 0.106

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.108 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.102 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.113 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn2553   N_FOD 0.076

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.049 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.075 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.067 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn8829   N_FOD 0.048

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.048 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.042 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.057 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn1155   N_FOD 0.017

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.066 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.038 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.064 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn9246   N_FOD 0.014

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.078 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.070 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.104 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn4498   N_FOD 0.012

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.085 nahe | +1 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.075 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.082 nahe | +1 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn1061   N_FOD 0.012

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.064 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.046 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.057 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn4003   N_FOD 0.010

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.040 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.042 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.050 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn4004   N_FOD 0.009

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.070 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.058 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.059 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn4063   N_FOD 0.008

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.082 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.090 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.088 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn4114   N_FOD 0.008

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.036 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.055 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.054 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn4060   N_FOD 0.007

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.049 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.038 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.059 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn1961   N_FOD 0.004

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.049 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.047 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.048 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn1962   N_FOD 0.003

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.049 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.038 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.032 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

