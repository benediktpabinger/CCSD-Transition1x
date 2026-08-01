# Full stability pipeline: 45 reactions x 4 geometry sources

Job 10691631, `pipeline/stability_pipeline.py`. wB97M-V/def2-TZVP (PySCF, grids 3,
conv 1e-10). 180 calculations, 179 complete, 0 crashes.

Reaction set: top-26 by N_FOD, mid-9, low-10 (rxn0896 is rank 11 and in both
top-26 and mid, hence 45 not 46). Geometry sources: the RKS reference NEB TS and
the TS predicted by UMA-S, UMA-M and eSEN. No optimisation is performed -- every
quantity is evaluated at the given geometry.

## Columns

| column | meaning |
|---|---|
| `RKS` | did the restricted SCF converge |
| `max\|g\|_RKS` | max gradient component on the RKS surface, eV/A |
| `ext` | external (spin-symmetry-breaking) stability of the RKS solution |
| `lmin_ext` | lowest eigenvalue of the external orbital-rotation Hessian, Ha |
| `route` | 1 = followed the external instability eigenvector; 2 = triplet-seeded beta-HOMO flip |
| `dE_BS` | E(BS) - E(RKS); negative throughout, i.e. BS is always the lower solution |
| `S2` | <S^2> of the broken-symmetry solution |
| `max\|g\|_BS` | max gradient component on the BS surface, eV/A |
| `ratio` | `max\|g\|_RKS / max\|g\|_BS`. **< 1 = the geometry sits on the RKS surface, > 1 = on the BS surface.** |
| `UKS int` | internal stability of the BS solution |

Empty cells mean the row is externally stable, so no broken-symmetry solution
exists and the BS columns do not apply.

## Which surface do the geometries sit on?

Only the 71 externally unstable rows pose the question -- where the RKS solution
is externally stable, RKS and UKS are the same solution.

| geometries | on RKS surface | on BS surface | ratio |
|---|---|---|---|
| RKS reference (n = 19) | **19** | 0 | 0.02 - 0.28, median 0.03 |
| model (n = 52) | 1 | **51** | 0.83 - 43.55, median **11.94** |

Per model: UMA-S 0/18, UMA-M 0/17, eSEN 1/17 on the RKS surface. The single
exception, rxn0894/eSEN at ratio 0.83, has near-equal gradients (0.66 vs 0.80)
and is not a clear assignment.

Where the RKS solution is externally unstable it is not the ground state -- the
BS solution lies below it in all 71 rows. The reference geometry was optimised on
the RKS surface, i.e. on the wrong one, and carries BS gradients of 1.7-2.9 eV/A;
the model geometries carry 0.05-0.25 eV/A. So in these reactions an RMSD measured
against the RKS reference is partly a property of the reference, not of the model.

Caveat: the gradient ratio shows which surface a geometry is *closer* to. It does
not show that the model geometries are stationary on the BS surface -- 0.05-0.25
eV/A is small but not zero.

## Full table

| rxn | grp | N_FOD | geom | RKS | max\|g\|_RKS | ext | lmin_ext | route | dE_BS [meV] | S2 | max\|g\|_BS | ratio | UKS int | lmin_int(UKS) | lmin_ext(UKS) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rxn7949 | high | 1.1465 | RKS-ref | ja | 0.1049 | **instabil** | -0.06315 | 1 | -559.6 | 0.8930 | 1.6860 | 0.06 | stabil | 0.10546 | -0.04955 |
| rxn7949 | high | 1.1465 | UMA-S | ja | 1.3939 | **instabil** | -0.10803 | 1 | -1276.7 | 0.9971 | 0.2476 | 5.63 | stabil | 0.10724 | -0.06246 |
| rxn7949 | high | 1.1465 | UMA-M | ja | 1.5896 | **instabil** | -0.11018 | 1 | -1319.0 | 0.9998 | 0.0514 | 30.92 | stabil | 0.10681 | -0.06162 |
| rxn7949 | high | 1.1465 | eSEN | ja | 1.6405 | **instabil** | -0.11127 | 1 | -1338.8 | 1.0011 | 0.0737 | 22.25 | stabil | 0.10688 | -0.06233 |
| rxn8832 | high | 0.9997 | RKS-ref | ja | 0.1420 | **instabil** | -0.04925 | 1 | -428.0 | 0.8695 | 2.7328 | 0.05 | stabil | 0.09880 | -0.04231 |
| rxn8832 | high | 0.9997 | UMA-S | ja | 2.6137 | **instabil** | -0.08726 | 1 | -1000.1 | 0.9819 | 0.0963 | 27.14 | stabil | 0.17180 | -0.06139 |
| rxn8832 | high | 0.9997 | UMA-M | ja | 2.7662 | **instabil** | -0.08784 | 1 | -1033.3 | 0.9899 | 0.0754 | 36.71 | stabil | 0.16612 | -0.05961 |
| rxn8832 | high | 0.9997 | eSEN | ja | 2.7699 | **instabil** | -0.08854 | 1 | -1032.6 | 0.9882 | 0.2320 | 11.94 | stabil | 0.17044 | -0.06129 |
| rxn1320 | high | 0.9678 | RKS-ref | ja | 0.0588 | **instabil** | -0.04801 | 1 | -339.2 | 0.7846 | 2.0728 | 0.03 | stabil | 0.09891 | -0.05209 |
| rxn1320 | high | 0.9678 | UMA-S | ja | 1.8300 | **instabil** | -0.04999 | 1 | -301.9 | 0.6881 | 0.0687 | 26.63 | stabil | 0.15855 | -0.05673 |
| rxn1320 | high | 0.9678 | UMA-M | ja | 1.7957 | **instabil** | -0.05071 | 1 | -310.3 | 0.6947 | 0.0441 | 40.74 | stabil | 0.16019 | -0.05643 |
| rxn1320 | high | 0.9678 | eSEN | ja | 1.7797 | **instabil** | -0.04977 | 1 | -299.3 | 0.6861 | 0.1174 | 15.16 | stabil | 0.15799 | -0.05640 |
| rxn4113 | high | 0.9596 | RKS-ref | ja | 0.0786 | **instabil** | -0.00846 | 1 | -8.4 | 0.1399 | 0.3859 | 0.20 | stabil | 0.03181 | -0.02270 |
| rxn4113 | high | 0.9596 | UMA-S | ja | 0.3258 | **instabil** | -0.00922 | 1 | -9.8 | 0.1501 | 0.1728 | 1.88 | stabil | 0.03461 | -0.02216 |
| rxn4113 | high | 0.9596 | UMA-M | ja | 4.0293 | **instabil** | -0.14655 | 1 | -1940.0 | 1.0072 | 0.1848 | 21.80 | stabil | 0.10750 | -0.06988 |
| rxn4113 | high | 0.9596 | eSEN | ja | 0.5853 | **instabil** | -0.01158 | 1 | -15.6 | 0.1886 | 0.1860 | 3.15 | stabil | 0.04191 | -0.02233 |
| rxn8885 | high | 0.9225 | RKS-ref | ja | 0.0423 | **instabil** | -0.01109 | 1 | -42.8 | 0.5074 | 2.6373 | 0.02 | stabil | 0.02089 | -0.02871 |
| rxn8885 | high | 0.9225 | UMA-S | ja | 0.8377 | **instabil** | -0.15646 | 1 | -2033.7 | 1.0244 | 0.4845 | 1.73 | stabil | 0.00070 | -0.09491 |
| rxn8885 | high | 0.9225 | UMA-M | ja | 0.6401 | **instabil** | -0.00421 | 1 | -5.0 | 0.1753 | 0.1899 | 3.37 | stabil | 0.01122 | -0.01933 |
| rxn8885 | high | 0.9225 | eSEN | ja | 1.6953 | **instabil** | -0.21622 | 1 | -3014.8 | 1.0280 | 0.3753 | 4.52 | stabil | 0.00540 | -0.09630 |
| rxn7945 | high | 0.9033 | RKS-ref | ja | 0.0422 | stabil | 0.00434 |  |  |  |  |  |  |  |  |
| rxn7945 | high | 0.9033 | UMA-S | ja | 0.0594 | stabil | 0.00848 |  |  |  |  |  |  |  |  |
| rxn7945 | high | 0.9033 | UMA-M | ja | 0.0805 | stabil | 0.00660 |  |  |  |  |  |  |  |  |
| rxn7945 | high | 0.9033 | eSEN | ja | 0.1347 | stabil | 0.06167 |  |  |  |  |  |  |  |  |
| rxn7937 | high | 0.8775 | RKS-ref | ja | 0.0394 | stabil | 0.00733 |  |  |  |  |  |  |  |  |
| rxn7937 | high | 0.8775 | UMA-S | ja | 0.0930 | stabil | 0.01016 |  |  |  |  |  |  |  |  |
| rxn7937 | high | 0.8775 | UMA-M | ja | 0.0327 | stabil | 0.01002 |  |  |  |  |  |  |  |  |
| rxn7937 | high | 0.8775 | eSEN | ja | 0.2251 | stabil | 0.00809 |  |  |  |  |  |  |  |  |
| rxn6196 | high | 0.8689 | RKS-ref | ja | 0.1793 | **instabil** | -0.00687 | 1 | -10.7 | 0.2161 | 0.6384 | 0.28 | stabil | 0.02589 | -0.03455 |
| rxn6196 | high | 0.8689 | UMA-S | ja | 1.2876 | **instabil** | -0.01728 | 1 | -72.2 | 0.5204 | 0.0900 | 14.30 | stabil | 0.05627 | -0.03872 |
| rxn6196 | high | 0.8689 | UMA-M | ja | 1.2174 | **instabil** | -0.01617 | 1 | -64.2 | 0.4997 | 0.1376 | 8.84 | stabil | 0.05335 | -0.03835 |
| rxn6196 | high | 0.8689 | eSEN | ja | 1.2424 | **instabil** | -0.01671 | 1 | -68.8 | 0.5146 | 0.1404 | 8.85 | stabil | 0.05459 | -0.03861 |
| rxn0346 | high | 0.8474 | RKS-ref | ja | 0.0519 | **instabil** | -0.02719 | 1 | -147.6 | 0.6278 | 2.6125 | 0.02 | stabil | 0.06859 | -0.04215 |
| rxn0346 | high | 0.8474 | UMA-S | ja | 2.4732 | **instabil** | -0.02270 | 1 | -118.1 | 0.6091 | 0.2437 | 10.15 | stabil | 0.06217 | -0.04513 |
| rxn0346 | high | 0.8474 | UMA-M | ja | 2.5684 | **instabil** | -0.02126 | 1 | -110.4 | 0.6087 | 0.1735 | 14.80 | stabil | 0.05586 | -0.04597 |
| rxn0346 | high | 0.8474 | eSEN | ja | 2.7482 | **instabil** | -0.02536 | 1 | -145.8 | 0.6540 | 0.4849 | 5.67 | stabil | 0.06633 | -0.04725 |
| rxn1150 | high | 0.8466 | RKS-ref | ja | 0.1746 | stabil | 0.00498 |  |  |  |  |  |  |  |  |
| rxn1150 | high | 0.8466 | UMA-S | ja | 0.0834 | stabil | 0.00333 |  |  |  |  |  |  |  |  |
| rxn1150 | high | 0.8466 | UMA-M | ja | 0.0708 | stabil | 0.00347 |  |  |  |  |  |  |  |  |
| rxn1150 | high | 0.8466 | eSEN | ja | 0.0971 | stabil | 0.00361 |  |  |  |  |  |  |  |  |
| rxn0896 | high | 0.8404 | RKS-ref | ja | 0.0490 | stabil | 0.00347 |  |  |  |  |  |  |  |  |
| rxn0896 | high | 0.8404 | UMA-S | ja | 0.0643 | stabil | 0.00374 |  |  |  |  |  |  |  |  |
| rxn0896 | high | 0.8404 | UMA-M | ja | 0.0820 | stabil | 0.00464 |  |  |  |  |  |  |  |  |
| rxn0896 | high | 0.8404 | eSEN | ja | 0.1193 | stabil | 0.00296 |  |  |  |  |  |  |  |  |
| rxn4518 | high | 0.8332 | RKS-ref | ja | 0.0681 | **instabil** | -0.07780 | 1 | -648.5 | 0.8418 | 2.9493 | 0.02 | stabil | 0.15427 | -0.06665 |
| rxn4518 | high | 0.8332 | UMA-S | ja | 2.2357 | **instabil** | -0.20707 | 1 | -2852.2 | 1.0086 | 0.0582 | 38.44 | stabil | 0.15644 | -0.08273 |
| rxn4518 | high | 0.8332 | UMA-M | ja | 2.3986 | **instabil** | -0.20360 | 1 | -2789.6 | 1.0082 | 0.0554 | 43.32 | stabil | 0.15804 | -0.08249 |
| rxn4518 | high | 0.8332 | eSEN | ja | 1.9617 | **instabil** | -0.21483 | 1 | -2974.5 | 1.0089 | 0.0514 | 38.13 | stabil | 0.15488 | -0.08318 |
| rxn3107 | high | 0.8006 | RKS-ref | ja | 0.0630 | **instabil** | -0.01255 | 1 | -38.8 | 0.4086 | 1.6457 | 0.04 | stabil | 0.02879 | -0.02941 |
| rxn3107 | high | 0.8006 | UMA-S | ja | 0.6384 | **instabil** | -0.00595 | 1 | -8.5 | 0.2049 | 0.1633 | 3.91 | stabil | 0.01653 | -0.02253 |
| rxn3107 | high | 0.8006 | UMA-M | ja | 0.6827 | **instabil** | -0.00525 | 1 | -6.4 | 0.1757 | 0.0922 | 7.41 | stabil | 0.01569 | -0.02225 |
| rxn3107 | high | 0.8006 | eSEN | ja | 0.7127 | **instabil** | -0.00433 | 1 | -4.4 | 0.1476 | 0.1426 | 5.00 | stabil | 0.01353 | -0.02173 |
| rxn8837 | high | 0.7983 | RKS-ref | ja | 0.0568 | **instabil** | -0.04401 | 1 | -293.9 | 0.7406 | 1.6974 | 0.03 | stabil | 0.11590 | -0.03950 |
| rxn8837 | high | 0.7983 | UMA-S | ja | 0.9658 | **instabil** | -0.24438 | 1 | -3474.5 | 1.0099 | 0.7568 | 1.28 | stabil | 0.05833 |  |
| rxn8837 | high | 0.7983 | UMA-M | ja | 1.5087 | **instabil** | -0.22515 | 1 | -3230.9 | 1.0070 | 0.7640 | 1.97 | stabil | 0.12492 |  |
| rxn8837 | high | 0.7983 | eSEN | ja | 0.1011 | stabil | 0.04728 |  |  |  |  |  |  |  |  |
| rxn7060 | high | 0.7881 | RKS-ref | ja | 0.0335 | **instabil** | -0.00790 | 1 | -22.1 | 0.3740 | 1.7658 | 0.02 | stabil | 0.01939 | -0.02917 |
| rxn7060 | high | 0.7881 | UMA-S | ja | 0.8820 | stabil | 0.00277 |  |  |  |  |  |  |  |  |
| rxn7060 | high | 0.7881 | UMA-M | ja | 0.3342 | stabil | 0.00134 |  |  |  |  |  |  |  |  |
| rxn7060 | high | 0.7881 | eSEN | ja | 1.1231 | stabil | 0.00288 |  |  |  |  |  |  |  |  |
| rxn5691 | high | 0.7777 | RKS-ref | ja | 0.0407 | **instabil** | -0.02902 | 1 | -155.9 | 0.6290 | 1.4192 | 0.03 | stabil | 0.06520 | -0.04680 |
| rxn5691 | high | 0.7777 | UMA-S | ja | 2.9360 | **instabil** | -0.12242 | 1 | -1592.6 | 1.0112 | 0.1537 | 19.10 | stabil | 0.13512 | -0.08063 |
| rxn5691 | high | 0.7777 | UMA-M | ja | 2.5080 | **instabil** | -0.12451 | 1 | -1725.8 | 1.0171 | 0.0853 | 29.41 | stabil | 0.12794 | -0.08166 |
| rxn5691 | high | 0.7777 | eSEN | ja | 2.9647 | **instabil** | -0.12294 | 1 | -1601.7 | 1.0114 | 0.0681 | 43.55 | stabil | 0.13471 | -0.08071 |
| rxn1283 | high | 0.7688 | RKS-ref | ja | 0.0380 | **instabil** | -0.01389 | 1 | -44.5 | 0.4195 | 2.3861 | 0.02 | stabil | 0.03597 | -0.03086 |
| rxn1283 | high | 0.7688 | UMA-S | ja | 1.3199 | **instabil** | -0.12721 | 1 | -2030.2 | 0.9784 | 0.1597 | 8.27 | stabil | 0.00267 | -0.14550 |
| rxn1283 | high | 0.7688 | UMA-M | ja | 1.5483 | **instabil** | -0.11962 | 1 | -1867.9 | 0.9991 | 0.1251 | 12.38 | **instabil** | -0.00087 | -0.14424 |
| rxn1283 | high | 0.7688 | eSEN | ja | 1.1161 | **instabil** | -0.12516 | 1 | -1944.1 | 0.9872 | 0.1054 | 10.59 | stabil | 0.00153 | -0.14917 |
| rxn8827 | high | 0.7599 | RKS-ref | ja | 0.0263 | **instabil** | -0.01096 | 1 | -27.5 | 0.3376 | 1.1278 | 0.02 | stabil | 0.03241 | -0.02703 |
| rxn8827 | high | 0.7599 | UMA-S | ja | 2.1917 | **instabil** | -0.08559 | 1 | -1026.0 | 1.0009 | 0.1733 | 12.65 | stabil | 0.15554 | -0.06418 |
| rxn8827 | high | 0.7599 | UMA-M | ja | 2.1912 | **instabil** | -0.08554 | 1 | -1051.1 | 1.0070 | 0.1336 | 16.40 | stabil | 0.15046 | -0.06292 |
| rxn8827 | high | 0.7599 | eSEN | ja | 2.3331 | **instabil** | -0.08726 | 1 | -1083.7 | 1.0092 | 0.2284 | 10.21 | stabil | 0.15303 | -0.06543 |
| rxn4522 | high | 0.7308 | RKS-ref | ja | 0.0982 | **instabil** | -0.03247 | 1 | -184.0 | 0.6621 | 1.8750 | 0.05 | stabil | 0.07554 | -0.03990 |
| rxn4522 | high | 0.7308 | UMA-S | ja | 1.3031 | **instabil** | -0.16682 | 1 | -2209.6 | 1.0055 | 0.0748 | 17.43 | stabil | 0.15810 | -0.08266 |
| rxn4522 | high | 0.7308 | UMA-M | ja | 1.2875 | **instabil** | -0.16629 | 1 | -2193.4 | 1.0050 | 0.0831 | 15.50 | stabil | 0.15734 | -0.08297 |
| rxn4522 | high | 0.7308 | eSEN | ja | 1.2982 | **instabil** | -0.16624 | 1 | -2197.3 | 1.0053 | 0.0730 | 17.77 | stabil | 0.15861 | -0.08283 |
| rxn7936 | high | 0.7271 | RKS-ref | ja | 0.0273 | stabil | 0.00814 |  |  |  |  |  |  |  |  |
| rxn7936 | high | 0.7271 | UMA-S | ja | 0.1114 | stabil | 0.00741 |  |  |  |  |  |  |  |  |
| rxn7936 | high | 0.7271 | UMA-M | ja | 0.0655 | stabil | 0.00774 |  |  |  |  |  |  |  |  |
| rxn7936 | high | 0.7271 | eSEN | ja | 0.1660 | stabil | 0.00675 |  |  |  |  |  |  |  |  |
| rxn1147 | high | 0.7254 | RKS-ref | ja | 0.0651 | **instabil** | -0.02450 | 1 | -105.2 | 0.5340 | 1.8398 | 0.04 | stabil | 0.05685 | -0.02824 |
| rxn1147 | high | 0.7254 | UMA-S | ja | 0.0774 | stabil | 0.00698 |  |  |  |  |  |  |  |  |
| rxn1147 | high | 0.7254 | UMA-M | ja | 0.0496 | stabil | 0.00688 |  |  |  |  |  |  |  |  |
| rxn1147 | high | 0.7254 | eSEN | ja | 0.0682 | stabil | 0.00660 |  |  |  |  |  |  |  |  |
| rxn0894 | high | 0.7160 | RKS-ref | ja | 0.0624 | **instabil** | -0.04014 | 1 | -190.2 | 0.5804 | 1.3498 | 0.05 | stabil | 0.08660 | -0.04011 |
| rxn0894 | high | 0.7160 | UMA-S | ja | 3.2894 | **instabil** | -0.11221 | 1 | -1278.2 | 0.9857 | 0.7812 | 4.21 | stabil | 0.06494 | -0.06145 |
| rxn0894 | high | 0.7160 | UMA-M | **NEIN** |  |  |  |  |  |  |  |  |  |  |  |
| rxn0894 | high | 0.7160 | eSEN | ja | 0.6602 | **instabil** | -0.27735 | 1 | -3992.0 | 1.0276 | 0.7991 | 0.83 | stabil | 0.20992 |  |
| rxn0101 | high | 0.7132 | RKS-ref | ja | 0.0584 | stabil | 0.07143 |  |  |  |  |  |  |  |  |
| rxn0101 | high | 0.7132 | UMA-S | ja | 0.1376 | stabil | 0.07094 |  |  |  |  |  |  |  |  |
| rxn0101 | high | 0.7132 | UMA-M | ja | 0.0968 | stabil | 0.07119 |  |  |  |  |  |  |  |  |
| rxn0101 | high | 0.7132 | eSEN | ja | 0.0985 | stabil | 0.07068 |  |  |  |  |  |  |  |  |
| rxn10005 | high | 0.6951 | RKS-ref | ja | 0.0182 | stabil | 0.00335 |  |  |  |  |  |  |  |  |
| rxn10005 | high | 0.6951 | UMA-S | ja | 0.0508 | stabil | 0.00329 |  |  |  |  |  |  |  |  |
| rxn10005 | high | 0.6951 | UMA-M | ja | 0.0296 | stabil | 0.00319 |  |  |  |  |  |  |  |  |
| rxn10005 | high | 0.6951 | eSEN | ja | 0.0693 | stabil | 0.00350 |  |  |  |  |  |  |  |  |
| rxn10054 | high | 0.6949 | RKS-ref | ja | 0.0135 | stabil | 0.00131 |  |  |  |  |  |  |  |  |
| rxn10054 | high | 0.6949 | UMA-S | ja | 1.0142 | **instabil** | -0.01040 | 1 | -23.4 | 0.3062 | 0.5909 | 1.72 | stabil | 0.02944 | -0.02973 |
| rxn10054 | high | 0.6949 | UMA-M | ja | 0.2810 | **instabil** | -0.00254 | 1 | -1.4 | 0.0800 | 0.0659 | 4.26 | stabil | 0.00932 | -0.02407 |
| rxn10054 | high | 0.6949 | eSEN | ja | 0.3485 | **instabil** | -0.00207 | 1 | -0.9 | 0.0657 | 0.1249 | 2.79 | stabil | 0.00769 | -0.02374 |
| rxn7957 | high | 0.6845 | RKS-ref | ja | 0.0265 | **instabil** | -0.02396 | 1 | -99.8 | 0.5132 | 0.9010 | 0.03 | stabil | 0.06303 | -0.03346 |
| rxn7957 | high | 0.6845 | UMA-S | ja | 3.9200 | **instabil** | -0.06190 | 1 | -416.8 | 0.7345 | 0.1374 | 28.54 | stabil | 0.16763 | -0.04822 |
| rxn7957 | high | 0.6845 | UMA-M | ja | 3.4918 | **instabil** | -0.05466 | 1 | -332.5 | 0.6850 | 0.1130 | 30.90 | stabil | 0.15990 | -0.04637 |
| rxn7957 | high | 0.6845 | eSEN | ja | 3.8347 | **instabil** | -0.06137 | 1 | -410.4 | 0.7312 | 0.1094 | 35.04 | stabil | 0.16632 | -0.04805 |
| rxn1154 | mid | 0.5664 | RKS-ref | ja | 0.0457 | stabil | 0.03271 |  |  |  |  |  |  |  |  |
| rxn1154 | mid | 0.5664 | UMA-S | ja | 0.1261 | stabil | 0.02067 |  |  |  |  |  |  |  |  |
| rxn1154 | mid | 0.5664 | UMA-M | ja | 0.0439 | stabil | 0.02375 |  |  |  |  |  |  |  |  |
| rxn1154 | mid | 0.5664 | eSEN | ja | 0.1069 | stabil | 0.02658 |  |  |  |  |  |  |  |  |
| rxn5690 | mid | 0.4334 | RKS-ref | ja | 0.0367 | **instabil** | -0.00268 | 1 | -1.3 | 0.0682 | 0.1620 | 0.23 | stabil | 0.01040 | -0.03441 |
| rxn5690 | mid | 0.4334 | UMA-S | ja | 0.8888 | **instabil** | -0.01260 | 1 | -28.3 | 0.3024 | 0.1907 | 4.66 | stabil | 0.04343 | -0.03841 |
| rxn5690 | mid | 0.4334 | UMA-M | ja | 0.9365 | **instabil** | -0.01328 | 1 | -31.3 | 0.3157 | 0.1119 | 8.37 | stabil | 0.04550 | -0.03907 |
| rxn5690 | mid | 0.4334 | eSEN | ja | 0.9547 | **instabil** | -0.01186 | 1 | -25.2 | 0.2873 | 0.1585 | 6.02 | stabil | 0.04121 | -0.03772 |
| rxn4513 | mid | 0.3073 | RKS-ref | ja | 0.0881 | stabil | 0.05820 |  |  |  |  |  |  |  |  |
| rxn4513 | mid | 0.3073 | UMA-S | ja | 0.1577 | stabil | 0.05667 |  |  |  |  |  |  |  |  |
| rxn4513 | mid | 0.3073 | UMA-M | ja | 0.1531 | stabil | 0.05681 |  |  |  |  |  |  |  |  |
| rxn4513 | mid | 0.3073 | eSEN | ja | 0.1496 | stabil | 0.05700 |  |  |  |  |  |  |  |  |
| rxn7955 | mid | 0.2191 | RKS-ref | ja | 0.0401 | stabil | 0.07485 |  |  |  |  |  |  |  |  |
| rxn7955 | mid | 0.2191 | UMA-S | ja | 0.0561 | stabil | 0.07452 |  |  |  |  |  |  |  |  |
| rxn7955 | mid | 0.2191 | UMA-M | ja | 0.0407 | stabil | 0.07462 |  |  |  |  |  |  |  |  |
| rxn7955 | mid | 0.2191 | eSEN | ja | 0.0517 | stabil | 0.07469 |  |  |  |  |  |  |  |  |
| rxn4519 | mid | 0.1545 | RKS-ref | ja | 0.0278 | stabil | 0.09009 |  |  |  |  |  |  |  |  |
| rxn4519 | mid | 0.1545 | UMA-S | ja | 0.0499 | stabil | 0.09263 |  |  |  |  |  |  |  |  |
| rxn4519 | mid | 0.1545 | UMA-M | ja | 0.0350 | stabil | 0.09220 |  |  |  |  |  |  |  |  |
| rxn4519 | mid | 0.1545 | eSEN | ja | 0.0677 | stabil | 0.09183 |  |  |  |  |  |  |  |  |
| rxn4500 | mid | 0.1061 | RKS-ref | ja | 0.0927 | stabil | 0.07882 |  |  |  |  |  |  |  |  |
| rxn4500 | mid | 0.1061 | UMA-S | ja | 0.1081 | stabil | 0.07882 |  |  |  |  |  |  |  |  |
| rxn4500 | mid | 0.1061 | UMA-M | ja | 0.1016 | stabil | 0.07880 |  |  |  |  |  |  |  |  |
| rxn4500 | mid | 0.1061 | eSEN | ja | 0.1130 | stabil | 0.07881 |  |  |  |  |  |  |  |  |
| rxn2553 | mid | 0.0760 | RKS-ref | ja | 0.0536 | stabil | 0.12428 |  |  |  |  |  |  |  |  |
| rxn2553 | mid | 0.0760 | UMA-S | ja | 0.0485 | stabil | 0.12427 |  |  |  |  |  |  |  |  |
| rxn2553 | mid | 0.0760 | UMA-M | ja | 0.0746 | stabil | 0.12399 |  |  |  |  |  |  |  |  |
| rxn2553 | mid | 0.0760 | eSEN | ja | 0.0667 | stabil | 0.12396 |  |  |  |  |  |  |  |  |
| rxn8829 | mid | 0.0478 | RKS-ref | ja | 0.0607 | stabil | 0.11405 |  |  |  |  |  |  |  |  |
| rxn8829 | mid | 0.0478 | UMA-S | ja | 0.0482 | stabil | 0.11445 |  |  |  |  |  |  |  |  |
| rxn8829 | mid | 0.0478 | UMA-M | ja | 0.0421 | stabil | 0.11448 |  |  |  |  |  |  |  |  |
| rxn8829 | mid | 0.0478 | eSEN | ja | 0.0573 | stabil | 0.11393 |  |  |  |  |  |  |  |  |
| rxn1155 | mid | 0.0167 | RKS-ref | ja | 0.0343 | stabil | 0.13380 |  |  |  |  |  |  |  |  |
| rxn1155 | mid | 0.0167 | UMA-S | ja | 0.0662 | stabil | 0.13398 |  |  |  |  |  |  |  |  |
| rxn1155 | mid | 0.0167 | UMA-M | ja | 0.0378 | stabil | 0.13416 |  |  |  |  |  |  |  |  |
| rxn1155 | mid | 0.0167 | eSEN | ja | 0.0636 | stabil | 0.13402 |  |  |  |  |  |  |  |  |
| rxn9246 | low | 0.0142 | RKS-ref | ja | 0.0187 | stabil | 0.15454 |  |  |  |  |  |  |  |  |
| rxn9246 | low | 0.0142 | UMA-S | ja | 0.0777 | stabil | 0.15384 |  |  |  |  |  |  |  |  |
| rxn9246 | low | 0.0142 | UMA-M | ja | 0.0704 | stabil | 0.15362 |  |  |  |  |  |  |  |  |
| rxn9246 | low | 0.0142 | eSEN | ja | 0.1042 | stabil | 0.15350 |  |  |  |  |  |  |  |  |
| rxn4498 | low | 0.0124 | RKS-ref | ja | 0.0494 | stabil | 0.17434 |  |  |  |  |  |  |  |  |
| rxn4498 | low | 0.0124 | UMA-S | ja | 0.0849 | stabil | 0.17436 |  |  |  |  |  |  |  |  |
| rxn4498 | low | 0.0124 | UMA-M | ja | 0.0747 | stabil | 0.17426 |  |  |  |  |  |  |  |  |
| rxn4498 | low | 0.0124 | eSEN | ja | 0.0817 | stabil | 0.17343 |  |  |  |  |  |  |  |  |
| rxn1061 | low | 0.0120 | RKS-ref | ja | 0.0575 | stabil | 0.14174 |  |  |  |  |  |  |  |  |
| rxn1061 | low | 0.0120 | UMA-S | ja | 0.0643 | stabil | 0.14210 |  |  |  |  |  |  |  |  |
| rxn1061 | low | 0.0120 | UMA-M | ja | 0.0463 | stabil | 0.14218 |  |  |  |  |  |  |  |  |
| rxn1061 | low | 0.0120 | eSEN | ja | 0.0568 | stabil | 0.14217 |  |  |  |  |  |  |  |  |
| rxn4003 | low | 0.0100 | RKS-ref | ja | 0.0163 | stabil | 0.17509 |  |  |  |  |  |  |  |  |
| rxn4003 | low | 0.0100 | UMA-S | ja | 0.0396 | stabil | 0.17586 |  |  |  |  |  |  |  |  |
| rxn4003 | low | 0.0100 | UMA-M | ja | 0.0424 | stabil | 0.17578 |  |  |  |  |  |  |  |  |
| rxn4003 | low | 0.0100 | eSEN | ja | 0.0502 | stabil | 0.17591 |  |  |  |  |  |  |  |  |
| rxn4004 | low | 0.0095 | RKS-ref | ja | 0.0447 | stabil | 0.15292 |  |  |  |  |  |  |  |  |
| rxn4004 | low | 0.0095 | UMA-S | ja | 0.0702 | stabil | 0.15278 |  |  |  |  |  |  |  |  |
| rxn4004 | low | 0.0095 | UMA-M | ja | 0.0578 | stabil | 0.15280 |  |  |  |  |  |  |  |  |
| rxn4004 | low | 0.0095 | eSEN | ja | 0.0589 | stabil | 0.15280 |  |  |  |  |  |  |  |  |
| rxn4063 | low | 0.0080 | RKS-ref | ja | 0.0396 | stabil | 0.18308 |  |  |  |  |  |  |  |  |
| rxn4063 | low | 0.0080 | UMA-S | ja | 0.0823 | stabil | 0.18299 |  |  |  |  |  |  |  |  |
| rxn4063 | low | 0.0080 | UMA-M | ja | 0.0903 | stabil | 0.18298 |  |  |  |  |  |  |  |  |
| rxn4063 | low | 0.0080 | eSEN | ja | 0.0881 | stabil | 0.18307 |  |  |  |  |  |  |  |  |
| rxn4114 | low | 0.0079 | RKS-ref | ja | 0.0330 | stabil | 0.17696 |  |  |  |  |  |  |  |  |
| rxn4114 | low | 0.0079 | UMA-S | ja | 0.0360 | stabil | 0.17689 |  |  |  |  |  |  |  |  |
| rxn4114 | low | 0.0079 | UMA-M | ja | 0.0548 | stabil | 0.17684 |  |  |  |  |  |  |  |  |
| rxn4114 | low | 0.0079 | eSEN | ja | 0.0543 | stabil | 0.17703 |  |  |  |  |  |  |  |  |
| rxn4060 | low | 0.0067 | RKS-ref | ja | 0.0570 | stabil | 0.16458 |  |  |  |  |  |  |  |  |
| rxn4060 | low | 0.0067 | UMA-S | ja | 0.0494 | stabil | 0.16456 |  |  |  |  |  |  |  |  |
| rxn4060 | low | 0.0067 | UMA-M | ja | 0.0382 | stabil | 0.16458 |  |  |  |  |  |  |  |  |
| rxn4060 | low | 0.0067 | eSEN | ja | 0.0591 | stabil | 0.16461 |  |  |  |  |  |  |  |  |
| rxn1961 | low | 0.0043 | RKS-ref | ja | 0.0249 | stabil | 0.17749 |  |  |  |  |  |  |  |  |
| rxn1961 | low | 0.0043 | UMA-S | ja | 0.0489 | stabil | 0.17738 |  |  |  |  |  |  |  |  |
| rxn1961 | low | 0.0043 | UMA-M | ja | 0.0467 | stabil | 0.17734 |  |  |  |  |  |  |  |  |
| rxn1961 | low | 0.0043 | eSEN | ja | 0.0476 | stabil | 0.17735 |  |  |  |  |  |  |  |  |
| rxn1962 | low | 0.0026 | RKS-ref | ja | 0.0450 | stabil | 0.18280 |  |  |  |  |  |  |  |  |
| rxn1962 | low | 0.0026 | UMA-S | ja | 0.0486 | stabil | 0.18264 |  |  |  |  |  |  |  |  |
| rxn1962 | low | 0.0026 | UMA-M | ja | 0.0378 | stabil | 0.18265 |  |  |  |  |  |  |  |  |
| rxn1962 | low | 0.0026 | eSEN | ja | 0.0320 | stabil | 0.18266 |  |  |  |  |  |  |  |  |


