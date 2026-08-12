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
| ours | PySCF, batch, converged / 0.838 | 0.010 stationaer | +535 | **bestanden** 1 @ -732 | **bestanden** 0.98 / 1.159 0.310 | 1.84 1.50 | ORCA |
| UMA-S | BS / 0.997 | 0.248 NICHT STATIONAER | +15 | **bestanden** 1 @ -96 | **DURCHGEFALLEN** 0.19 / 0.008 0.007 | 2.36 1.48 | PySCF |
| UMA-M | BS / 1.000 | 0.051 nahe | +0 | **bestanden** 1 @ -109 | **DURCHGEFALLEN** 0.13 / 0.003 0.011 | 2.40 1.48 | PySCF |
| eSEN | BS / 1.001 | 0.074 nahe | +0 | **bestanden** 1 @ -100 | **DURCHGEFALLEN** 0.16 / 0.008 0.009 | 2.41 1.48 | PySCF |
| tsopt UMA-M | TS-Opt ab Modellgeometrie / 0.993 | 0.012 stationaer | -3 | **bestanden** 1 @ -69 | **DURCHGEFALLEN** 0.23 / 0.003 0.008 | 2.41 1.48 | ORCA |

## rxn8832   N_FOD 1.000   ΔE_BS an der Referenz -428.0 meV

reaktive Bindungen: C1-C6, C1-C2

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, v2, converged / 1.001 | 0.014 stationaer | +0 | **bestanden** 1 @ -651 | **bestanden** 0.96 / 1.217 0.107 | 1.74 1.48 | ORCA |
| UMA-S | BS / 0.982 | 0.097 nahe | +28 | **bestanden** 1 @ -634 | **bestanden** 0.96 / 1.218 0.109 | 1.73 1.48 | ORCA |
| UMA-M | BS / 0.990 | 0.077 nahe | +12 | **bestanden** 1 @ -644 | **bestanden** 0.96 / 1.216 0.110 | 1.73 1.48 | ORCA |
| eSEN | BS / 0.988 | 0.227 NICHT STATIONAER | +18 | **bestanden** 1 @ -650 | **bestanden** 0.96 / 1.224 0.095 | 1.75 1.48 | ORCA |
| NEB-TS | ORCA BS-NEB / 1.008 | 0.018 stationaer | -18 | **bestanden** 1 @ -656 | **bestanden** 0.96 / 1.217 0.108 | 1.74 1.48 | ORCA |
| tsopt UMA-M | TS-Opt ab Modellgeometrie / 1.001 | 0.018 stationaer | -5 | **bestanden** 1 @ -652 | **bestanden** 0.96 / 1.217 0.109 | 1.74 1.48 | ORCA |

## rxn1320   N_FOD 0.968   ΔE_BS an der Referenz -339.2 meV

reaktive Bindungen: C2-H6, O0-H6

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, batch, converged / 1.019 | 0.009 stationaer | +0 | **bestanden** 1 @ -313 | **DURCHGEFALLEN** 0.00 / 0.000 0.001 | 3.36 0.97 | ORCA |
| UMA-S | BS / 0.690 | 0.067 nahe | +252 | **bestanden** 1 @ -416 | **bestanden** 0.37 / 0.213 0.010 | 2.61 0.97 | ORCA |
| UMA-M | BS / 0.697 | 0.043 stationaer | +251 | **bestanden** 1 @ -405 | **bestanden** 0.36 / 0.211 0.010 | 2.60 0.97 | ORCA |
| eSEN | BS / 0.688 | 0.115 nahe | +251 | **bestanden** 1 @ -408 | **bestanden** 0.36 / 0.215 0.010 | 2.61 0.97 | ORCA |
| NEB-TS | ORCA BS-NEB / 0.810 | 2.062 NICHT STATIONAER | +507 | **bestanden** 1 @ -233 | **bestanden** 0.36 / 0.169 0.123 | 1.98 0.99 | ORCA |

## rxn4113   N_FOD 0.960   ΔE_BS an der Referenz -8.4 meV

reaktive Bindungen: O0-C3, N2-C3

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, fromneb, converged / 0.969 | 0.010 stationaer | +40 | **bestanden** 1 @ -152 | **bestanden** 0.75 / 0.107 0.970 | 3.70 3.14 | ORCA |
| UMA-S | BS / 0.149 | 0.169 NICHT STATIONAER | +1065 | **bestanden** 1 @ -876 | **bestanden** 0.53 / 0.704 0.393 | 2.27 2.50 | ORCA |
| UMA-M | BS / 1.007 | 0.186 NICHT STATIONAER | +0 | **bestanden** 1 @ -49 | **bestanden** 0.75 / 0.621 0.500 | 3.47 3.56 | ORCA |
| eSEN | BS / 0.187 | 0.182 NICHT STATIONAER | +1065 | **bestanden** 1 @ -900 | **bestanden** 0.51 / 0.699 0.354 | 2.28 2.52 | ORCA |
| NEB-TS | ORCA BS-NEB / 0.970 | 0.007 stationaer | +34 | **bestanden** 1 @ -148 | **bestanden** 0.75 / 0.103 0.970 | 3.71 3.15 | ORCA |

## rxn8885   N_FOD 0.923   ΔE_BS an der Referenz -42.8 meV

reaktive Bindungen: C1-O2, C1-N6

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, v2, BS_LOST / 0.149 | 0.016 stationaer | +342 | **bestanden** 1 @ -1673 | **bestanden** 0.42 / 0.602 0.032 | 2.09 2.48 | ORCA |
| UMA-S | BS / 1.024 | 0.483 NICHT STATIONAER | +0 | **DURCHGEFALLEN** 2 @ -44 | **bestanden** 0.72 / 0.865 0.013 | 3.33 2.57 | ORCA |
| UMA-M | BS / 0.171 | 0.178 NICHT STATIONAER | +346 | **bestanden** 1 @ -1683 | **bestanden** 0.40 / 0.581 0.015 | 2.09 2.48 | ORCA |
| eSEN | BS / 1.028 | 0.380 NICHT STATIONAER | +38 | **bestanden** 1 @ -29 | **DURCHGEFALLEN** 0.63 / 0.042 0.005 | 5.57 2.56 | ORCA |

## rxn6196   N_FOD 0.869   ΔE_BS an der Referenz -10.7 meV

reaktive Bindungen: C2-C5, C2-H10

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, fromneb, converged / 0.494 | 0.010 stationaer | +0 | **bestanden** 1 @ -760 | **bestanden** 0.97 / 1.283 0.385 | 2.01 2.19 | ORCA |
| UMA-S | BS / 0.519 | 0.092 nahe | +8 | **bestanden** 1 @ -781 | **bestanden** 0.97 / 1.301 0.432 | 2.00 2.20 | ORCA |
| UMA-M | BS / 0.498 | 0.143 nahe | +9 | **bestanden** 1 @ -743 | **bestanden** 0.97 / 1.304 0.470 | 2.02 2.23 | ORCA |
| eSEN | BS / 0.513 | 0.140 nahe | +10 | **bestanden** 1 @ -744 | **bestanden** 0.97 / 1.304 0.465 | 2.02 2.23 | ORCA |
| NEB-TS | ORCA BS-NEB / 0.224 | 0.683 NICHT STATIONAER | +26 | **DURCHGEFALLEN** 2 @ -845 | **bestanden** 0.95 / 1.158 0.089 | 1.89 1.93 | ORCA |

## rxn0346   N_FOD 0.847   ΔE_BS an der Referenz -147.6 meV

reaktive Bindungen: C5-H10, C2-C5

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, batch, BS_LOST / 0.594 | 0.018 stationaer | +0 | **bestanden** 1 @ -1289 | **bestanden** 0.68 / 0.147 1.038 | 2.24 1.88 | ORCA |
| UMA-S | BS / 0.607 | 0.241 NICHT STATIONAER | +5 | **bestanden** 1 @ -1253 | **bestanden** 0.71 / 0.123 1.054 | 2.20 1.87 | ORCA |
| UMA-M | BS / 0.608 | 0.182 NICHT STATIONAER | +2 | **bestanden** 1 @ -1334 | **bestanden** 0.68 / 0.145 1.029 | 2.23 1.87 | ORCA |
| eSEN | BS / 0.652 | 0.482 NICHT STATIONAER | +9 | **bestanden** 1 @ -1214 | **bestanden** 0.74 / 0.125 1.080 | 2.17 1.85 | ORCA |
| NEB-TS | ORCA BS-NEB / 0.609 | 2.553 NICHT STATIONAER | +84 | **DURCHGEFALLEN** 2 @ -790 | **bestanden** 0.80 / 0.384 0.696 | 1.85 1.65 | ORCA |
| tsopt UMA-M | TS-Opt ab Modellgeometrie / 0.595 | 0.009 stationaer | -8 | **bestanden** 1 @ -1295 | **bestanden** 0.67 / 0.142 1.031 | 2.25 1.88 | ORCA |

## rxn4518   N_FOD 0.833   ΔE_BS an der Referenz -648.5 meV

reaktive Bindungen: N0-O5, N0-C1

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, batch, converged / 1.009 | 0.010 stationaer | +0 | **bestanden** 1 @ -90 | **DURCHGEFALLEN** 0.07 / 0.036 0.327 | 3.33 3.27 | ORCA |
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
| tsopt UMA-M | TS-Opt ab Modellgeometrie / 0.173 | 0.016 stationaer | -7 | **bestanden** 1 @ -1484 | **bestanden** 0.40 / 0.598 0.071 | 2.09 2.46 | ORCA |

## rxn8837   N_FOD 0.798   ΔE_BS an der Referenz -293.9 meV

reaktive Bindungen: N0-C6, C4-C6

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, batch, converged / 1.039 | 0.017 stationaer | +0 | **bestanden** 1 @ -819 | **bestanden** 0.92 / 1.313 0.055 | 2.05 2.33 | ORCA |
| UMA-S | BS / 1.010 | 0.749 NICHT STATIONAER | +5469 | **DURCHGEFALLEN** 5 @ -151 | **bestanden** 0.79 / 0.229 0.122 | 6.17 1.48 | ORCA |
| UMA-M | BS / 1.007 | 0.757 NICHT STATIONAER | +3352 | **DURCHGEFALLEN** 3 @ -60 | **bestanden** 0.68 / 0.800 0.003 | 4.15 1.28 | ORCA |
| eSEN | RKS / 0.000 | 0.096 nahe | +1034 | **DURCHGEFALLEN** 2 @ -81 | **DURCHGEFALLEN** 0.02 / 0.020 0.006 | 3.98 1.28 | ORCA |
| NEB-TS | ORCA BS-NEB / 1.039 | 0.011 stationaer | -6 | **bestanden** 1 @ -817 | **bestanden** 0.92 / 1.313 0.057 | 2.05 2.33 | ORCA |
| tsopt UMA-M | TS-Opt ab Modellgeometrie / 1.006 | 0.014 stationaer | +2900 | **bestanden** 1 @ -59 | **DURCHGEFALLEN** 0.09 / 0.054 0.000 | 6.10 1.27 | ORCA |

## rxn7060   N_FOD 0.788   ΔE_BS an der Referenz -22.1 meV

reaktive Bindungen: O0-C1, O0-C5

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, v2, BS_LOST / 0.034 | 0.053 nahe | +0 | **bestanden** 1 @ -2024 | **bestanden** 0.51 / 0.459 0.520 | 1.62 1.29 | ORCA |
| UMA-S | RKS / -0.000 | 0.884 NICHT STATIONAER | +22 | **bestanden** 1 @ -689 | **bestanden** 0.63 / 0.624 0.447 | 1.59 1.28 | ORCA |
| UMA-M | RKS / -0.000 | 0.335 NICHT STATIONAER | +0 | **bestanden** 1 @ -688 | **bestanden** 0.65 / 0.680 0.461 | 1.62 1.29 | ORCA |
| eSEN | RKS / -0.000 | 1.126 NICHT STATIONAER | +46 | **bestanden** 1 @ -732 | **bestanden** 0.62 / 0.601 0.455 | 1.59 1.28 | ORCA |
| tsopt UMA-M | TS-Opt ab Modellgeometrie / 0.364 | 1.707 NICHT STATIONAER | -24 | **bestanden** 1 @ -2312 | **bestanden** 0.59 / 0.428 0.496 | 1.60 1.29 | ORCA |

## rxn5691   N_FOD 0.778   ΔE_BS an der Referenz -155.9 meV

reaktive Bindungen: O0-N6, C4-N6

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, batch, converged / 0.974 | 0.010 stationaer | +164 | **bestanden** 1 @ -95 | **DURCHGEFALLEN** 0.60 / 0.015 0.012 | 2.91 2.51 | ORCA |
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
| ours | PySCF, batch, converged / 1.024 | 0.010 stationaer | +0 | **bestanden** 1 @ -590 | **bestanden** 0.97 / 1.390 0.093 | 2.03 2.58 | ORCA |
| UMA-S | BS / 1.001 | 0.176 NICHT STATIONAER | +34 | **bestanden** 1 @ -590 | **bestanden** 0.96 / 1.383 0.138 | 2.02 2.56 | ORCA |
| UMA-M | BS / 1.007 | 0.131 nahe | +21 | **bestanden** 1 @ -590 | **bestanden** 0.97 / 1.386 0.129 | 2.02 2.57 | ORCA |
| eSEN | BS / 1.010 | 0.231 NICHT STATIONAER | +20 | **bestanden** 1 @ -575 | **bestanden** 0.96 / 1.380 0.117 | 2.04 2.57 | ORCA |
| NEB-TS | ORCA BS-NEB / 0.328 | 1.074 NICHT STATIONAER | +407 | **DURCHGEFALLEN** 2 @ -605 | **bestanden** 0.36 / 0.396 0.542 | 1.96 2.32 | ORCA |
| tsopt UMA-M | TS-Opt ab Modellgeometrie / 1.024 | 0.008 stationaer | -6 | **bestanden** 1 @ -592 | **bestanden** 0.97 / 1.390 0.090 | 2.03 2.58 | ORCA |

## rxn4522   N_FOD 0.731   ΔE_BS an der Referenz -184.0 meV

reaktive Bindungen: O3-C4, N0-O3

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, batch, BS_LOST / 0.000 | NICHT GEPRUEFT | +1845 | NICHT GEPRUEFT | NICHT GEPRUEFT | 1.85 2.07 | PySCF, batch, BS_LOST |
| UMA-S | BS / 1.005 | 0.075 nahe | +0 | **bestanden** 1 @ -83 | **bestanden** 0.34 / 0.017 0.357 | 1.39 2.88 | PySCF |
| UMA-M | BS / 1.005 | 0.083 nahe | +4 | **bestanden** 1 @ -81 | **bestanden** 0.36 / 0.017 0.366 | 1.39 2.88 | PySCF |
| eSEN | BS / 1.005 | 0.073 nahe | +2 | **bestanden** 1 @ -80 | **bestanden** 0.36 / 0.017 0.357 | 1.39 2.86 | PySCF |
| NEB-TS | ORCA BS-NEB / 0.951 | 0.009 stationaer | +1214 | **bestanden** 1 @ -495 | **bestanden** 0.43 / 0.038 0.416 | 2.93 2.92 | ORCA |

## rxn1147   N_FOD 0.725   ΔE_BS an der Referenz -105.2 meV

reaktive Bindungen: C1-C2, C1-O5

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, batch, converged / 0.456 | 0.002 stationaer | +234 | **bestanden** 1 @ -591 | **bestanden** 0.60 / 0.134 0.943 | 3.20 1.86 | PySCF |
| UMA-S | RKS / -0.000 | 0.081 nahe | +0 | **bestanden** 1 @ -253 | **bestanden** 0.24 / 0.056 0.059 | 3.57 1.50 | ORCA |
| UMA-M | RKS / 0.000 | 0.050 stationaer | +3 | **bestanden** 1 @ -248 | **bestanden** 0.22 / 0.057 0.071 | 3.57 1.50 | PySCF |
| eSEN | RKS / 0.000 | 0.068 nahe | +2 | **bestanden** 1 @ -250 | **bestanden** 0.23 / 0.055 0.067 | 3.56 1.50 | PySCF |
| NEB-TS | ORCA BS-NEB / 0.460 | 0.016 stationaer | +225 | **bestanden** 1 @ -589 | **bestanden** 0.60 / 0.137 0.944 | 3.21 1.87 | ORCA |
| tsopt UMA-M | TS-Opt ab Modellgeometrie / 0.000 | 0.013 stationaer | -13 | **bestanden** 1 @ -230 | **bestanden** 0.18 / 0.073 0.035 | 3.59 1.50 | ORCA |

## rxn0894   N_FOD 0.716   ΔE_BS an der Referenz -190.2 meV

reaktive Bindungen: C4-H8, C0-H8

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, batch, converged / 0.816 | 0.016 stationaer | +0 | **bestanden** 1 @ -617 | **bestanden** 0.58 / 0.303 1.036 | 1.17 1.91 | ORCA |
| UMA-S | BS / 0.986 | 0.776 NICHT STATIONAER | +68 | **bestanden** 1 @ -241 | **bestanden** 0.44 / 0.127 0.580 | 1.15 2.12 | ORCA |
| UMA-M | ? / 1.038 | 1.320 NICHT STATIONAER | +715 | **bestanden** 1 @ -52 | **bestanden** 0.80 / 0.120 0.214 | 6.61 10.05 | ORCA |
| eSEN | BS / 1.028 | 0.794 NICHT STATIONAER | +319 | **bestanden** 1 @ -39 | **bestanden** 0.87 / 0.629 0.680 | 6.85 10.37 | ORCA |
| NEB-TS | ORCA BS-NEB / 1.034 | 0.022 stationaer | +679 | **bestanden** 1 @ -1075 | **bestanden** 0.95 / 1.306 0.423 | 1.68 4.23 | ORCA |
| tsopt UMA-S | TS-Opt ab Modellgeometrie / 1.024 | 0.010 stationaer | -186 | **DURCHGEFALLEN** 0 | **bestanden** 0.14 / 0.054 0.163 | 1.11 2.70 | ORCA |

## rxn7957   N_FOD 0.684   ΔE_BS an der Referenz -99.8 meV

reaktive Bindungen: C1-H7, C5-H7

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | PySCF, v2, converged / 0.709 | 0.001 stationaer | +890 | **bestanden** 1 @ -677 | **bestanden** 0.27 / 0.544 0.061 | 2.46 1.12 | PySCF |
| UMA-S | BS / 0.734 | 0.137 nahe | +0 | **bestanden** 1 @ -436 | **bestanden** 0.55 / 0.846 0.309 | 1.89 1.17 | PySCF |
| UMA-M | BS / 0.688 | 0.111 nahe | +0 | **bestanden** 1 @ -696 | **bestanden** 0.75 / 1.015 0.569 | 1.87 1.19 | ORCA |
| eSEN | BS / 0.731 | 0.109 nahe | +0 | **bestanden** 1 @ -459 | **bestanden** 0.57 / 0.870 0.339 | 1.88 1.17 | PySCF |
| NEB-TS | ORCA BS-NEB / 0.729 | 0.009 stationaer | +886 | **bestanden** 1 @ -670 | **bestanden** 0.27 / 0.545 0.053 | 2.48 1.12 | ORCA |
| tsopt UMA-M | TS-Opt ab Modellgeometrie / 0.699 | 0.013 stationaer | -6 | **bestanden** 1 @ -624 | **bestanden** 0.70 / 0.987 0.506 | 1.86 1.18 | ORCA |

## rxn5690   N_FOD 0.433   ΔE_BS an der Referenz -1.3 meV

reaktive Bindungen: C3-H8, C1-C4

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | BS / 0.302 | 0.184 NICHT STATIONAER | +1 | **bestanden** 1 @ -870 | **bestanden** 0.83 / 0.045 1.203 | 1.11 1.83 | ORCA |
| UMA-M | BS / 0.315 | 0.110 nahe | +0 | **bestanden** 1 @ -869 | **bestanden** 0.84 / 0.046 1.209 | 1.11 1.83 | ORCA |
| eSEN | BS / 0.287 | 0.154 NICHT STATIONAER | +6 | **bestanden** 1 @ -886 | **bestanden** 0.84 / 0.043 1.207 | 1.11 1.83 | ORCA |
| NEB-TS | ORCA BS-NEB / 0.000 | 0.004 stationaer | -26 | **bestanden** 1 @ -448 | **bestanden** 0.62 / 0.088 0.741 | 1.18 1.78 | ORCA |

---

# Einfach — die 26 (Kontrollgruppe)

## rxn7945   N_FOD 0.903

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.059 nahe | +514 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | ORCA |
| UMA-M | RKS / 0.000 | 0.081 nahe | +513 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.135 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| NEB-TS | ORCA BS-NEB / 0.000 | 0.025 stationaer | +469 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | ORCA |

## rxn7937   N_FOD 0.877

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.093 nahe | +2 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.037 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | ORCA |
| eSEN | RKS / 0.000 | 0.225 NICHT STATIONAER | +5 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn1150   N_FOD 0.847

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.083 nahe | +1 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / -0.000 | 0.069 nahe | +1 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | ORCA |
| eSEN | RKS / 0.000 | 0.097 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| NEB-TS | ORCA BS-NEB / -0.000 | 0.017 stationaer | -13 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | ORCA |

## rxn0896   N_FOD 0.840

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / -0.000 | 0.057 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | ORCA |
| UMA-M | RKS / 0.000 | 0.082 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.119 nahe | +1 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn7936   N_FOD 0.727

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.111 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.066 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | ORCA |
| eSEN | RKS / 0.000 | 0.166 NICHT STATIONAER | +2 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| NEB-CI | nur Bandpunkt, kein Sattelpunkt / — | NICHT GEPRUEFT | — | NICHT GEPRUEFT | NICHT GEPRUEFT | — | nur Bandpunkt, kein Sattelpunkt |

## rxn0101   N_FOD 0.713

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.138 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / -0.000 | 0.099 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | ORCA |
| eSEN | RKS / 0.000 | 0.099 nahe | +1 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn10005   N_FOD 0.695

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.051 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / -0.000 | 0.028 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | ORCA |
| eSEN | RKS / 0.000 | 0.069 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn10054   N_FOD 0.695

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | BS / 0.306 | 0.591 NICHT STATIONAER | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | BS |
| UMA-M | BS / 0.084 | 0.068 nahe | +3 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | ORCA |
| eSEN | BS / 0.066 | 0.125 nahe | +3 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | BS |

## rxn1154   N_FOD 0.566

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.126 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / -0.000 | 0.044 stationaer | +1 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | ORCA |
| eSEN | RKS / 0.000 | 0.107 nahe | +12 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn4513   N_FOD 0.307

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.158 NICHT STATIONAER | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.153 NICHT STATIONAER | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.146 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | ORCA |

## rxn7955   N_FOD 0.219

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.056 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / -0.000 | 0.037 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | ORCA |
| eSEN | RKS / 0.000 | 0.052 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn4519   N_FOD 0.154

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.050 stationaer | +1 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / -0.000 | 0.030 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | ORCA |
| eSEN | RKS / 0.000 | 0.068 nahe | +1 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn4500   N_FOD 0.106

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.108 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / -0.000 | 0.100 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | ORCA |
| eSEN | RKS / 0.000 | 0.113 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn2553   N_FOD 0.076

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.051 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | ORCA |
| UMA-M | RKS / 0.000 | 0.075 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.067 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn8829   N_FOD 0.048

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.048 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / -0.000 | 0.045 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | ORCA |
| eSEN | RKS / 0.000 | 0.057 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn1155   N_FOD 0.017

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.066 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / -0.000 | 0.037 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | ORCA |
| eSEN | RKS / 0.000 | 0.064 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn9246   N_FOD 0.014

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.078 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.068 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | ORCA |
| eSEN | RKS / 0.000 | 0.104 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn4498   N_FOD 0.012

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.085 nahe | +1 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.072 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | ORCA |
| eSEN | RKS / 0.000 | 0.082 nahe | +1 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn1061   N_FOD 0.012

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.064 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / -0.000 | 0.049 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | ORCA |
| eSEN | RKS / 0.000 | 0.057 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn4003   N_FOD 0.010

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.039 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | ORCA |
| UMA-M | RKS / 0.000 | 0.042 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.050 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn4004   N_FOD 0.009

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.070 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / -0.000 | 0.063 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | ORCA |
| eSEN | RKS / 0.000 | 0.059 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn4063   N_FOD 0.008

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / -0.000 | 0.085 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | ORCA |
| UMA-M | RKS / 0.000 | 0.090 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.088 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn4114   N_FOD 0.008

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / -0.000 | 0.035 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | ORCA |
| UMA-M | RKS / 0.000 | 0.055 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.054 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn4060   N_FOD 0.007

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.049 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.040 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | ORCA |
| eSEN | RKS / 0.000 | 0.059 nahe | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn1961   N_FOD 0.004

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.049 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.047 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | ORCA |
| eSEN | RKS / 0.000 | 0.048 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |

## rxn1962   N_FOD 0.003

| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |
|---|---|---|---|---|---|---|---|
| ours | kein konvergierter Sattelpunkt / — | NICHT GEPRUEFT | — | — | — | — | kein konvergierter Sattelpunkt |
| UMA-S | RKS / 0.000 | 0.049 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| UMA-M | RKS / 0.000 | 0.038 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | RKS |
| eSEN | RKS / 0.000 | 0.030 stationaer | +0 | NICHT GEPRUEFT | NICHT GEPRUEFT | — | ORCA |

