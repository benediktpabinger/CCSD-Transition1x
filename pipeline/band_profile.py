"""Where along the band does the symmetry breaking sit -- and is it at the top?

The retroactive measurement showed that 15 of 21 bands did hold a broken
solution, which overturns the diagnosis that BrokenSym loses it everywhere.
But "the band broke somewhere" is not the question that matters.  The image
that becomes the transition state is the highest one, so the number to look at
is <S^2> at the energy maximum of the band.

A band that breaks at images 2 and 3 while image 5 carries the barrier top and
stays restricted is on the wrong sheet exactly where it counts, and would look
like a success under the cruder measure.  rxn8827 is the case that raised the
suspicion: it broke on two images, and its NEB-TS still landed 0.019 A from the
RKS-TS.

Nothing here is recomputed.  band_s2_v2/<rxn>/band_s2.txt already carries
<S^2> and the total energy per image, and the NEB outcomes are in files that
exist.  This only puts them side by side.
"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checks

H = '/home/energy/s242862'
HA_EV = 27.211386245988
BREAK = 0.3          # same threshold the rest of the analysis uses

# bs_uks_neb_results holds 22 runs, and three of them -- rxn1150, rxn7936,
# rxn7945 -- are single-reference reactions.  Iterating over the directory
# instead of over this list put two of them in the "top restricted" group,
# where a restricted top is the correct answer and proves nothing.  It moved
# the group median from 1.074 to 0.683 and the stationary count from 1 of 5 to
# 3 of 7.
MR = ['rxn7949', 'rxn8832', 'rxn1320', 'rxn4113', 'rxn8885', 'rxn6196',
      'rxn0346', 'rxn4518', 'rxn3107', 'rxn8837', 'rxn7060', 'rxn5691',
      'rxn1283', 'rxn8827', 'rxn4522', 'rxn1147', 'rxn0894', 'rxn7957',
      'rxn5690']


def band(rx):
    """[(image, S2, E)] for one reaction, or None."""
    p = f'{H}/band_s2_v2/{rx}/band_s2.txt'
    if not os.path.exists(p):
        return None
    out = []
    for line in open(p, errors='replace'):
        f = line.split()
        if len(f) < 4 or f[2] == 'nan' or f[3] == 'nan':
            continue
        try:
            out.append((int(f[1]), float(f[2]), float(f[3])))
        except ValueError:
            continue
    return sorted(out) or None


def neb_grad(rx):
    """Gradient of the converged NEB-TS, from the sweep."""
    for d in (f'{H}/orca_freq/nebts_{rx}', f'{H}/orca_irc/nebts_{rx}'):
        p = f'{d}/engrad.engrad'
        if not os.path.exists(p):
            continue
        nums, L = [], open(p, errors='replace').read().split('\n')
        for i, line in enumerate(L):
            if 'current gradient' in line.lower():
                j = i + 2
                while j < len(L) and not L[j].startswith('#'):
                    s = L[j].strip()
                    if s:
                        nums.append(float(s))
                    j += 1
                break
        if nums:
            return float(np.abs(np.array(nums)).max() * 51.42208)
    return None


def rd(p):
    if not os.path.exists(p):
        return None
    L = open(p, errors='replace').read().split('\n')
    n = int(L[0].split()[0])
    x = []
    for line in L[2:2 + n]:
        f = line.split()
        if len(f) >= 4:
            x.append([float(v) for v in f[1:4]])
    return np.array(x)


def kabsch(A, B):
    if A is None or B is None or len(A) != len(B):
        return None
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    V, S, W = np.linalg.svd(Ac.T @ Bc)
    D = np.diag([1., 1., np.sign(np.linalg.det(V @ W))])
    return float(np.sqrt((((Ac @ (V @ D @ W)) - Bc) ** 2).sum() / len(A)))


def main():
    measured = sorted(os.path.basename(os.path.dirname(p))
                      for p in glob.glob(f'{H}/band_s2_v2/rxn*/band_s2.txt'))
    rxns = [r for r in MR if r in measured]
    extra = [r for r in measured if r not in MR]
    if extra:
        print(f'  ausgeschlossen, weil einreferenziert: {" ".join(extra)}')
    checks.expect(rxns, len(MR), 'MR-Reaktionen mit gemessenem Band',
                  warn_only=True)
    checks.header(__file__,
                  inputs=[f'{H}/band_s2_v2', f'{H}/orca_freq',
                          f'{H}/bs_uks_neb_results', f'{H}/orca_neb_results'],
                  note='<S^2> je Bild aus der validierten Nachmessung '
                       '(Kontrolle bestanden, 1 SCF-Zyklus).')

    print('=' * 100)
    print('DAS PROFIL: wo entlang des Bandes sitzt die Brechung')
    print('=' * 100)
    print('Bild 0 ist das Edukt, das letzte das Produkt.  Der Gipfel ist das')
    print('Bild mit der hoechsten Energie -- daraus wird das Climbing Image')
    print('und damit der Uebergangszustand.')
    print()
    print(f'{"rxn":<9}{"Profil <S^2> je Bild":<44}{"Gipfel":>7}'
          f'{"<S^2> dort":>11}{"Grad NEB":>10}   Urteil')
    print('-' * 100)

    rows, tops = [], []
    for rx in rxns:
        b = band(rx)
        if not b:
            continue
        s2 = [x[1] for x in b]
        en = [x[2] for x in b]
        top = int(np.argmax(en))
        s2top = s2[top]
        g = neb_grad(rx)

        # a compact glyph profile: . below threshold, # above
        prof = ''.join('#' if v > BREAK else ('-' if v > 0.05 else '.')
                       for v in s2)
        prof = prof[:top] + prof[top].upper().replace('.', 'o') \
            .replace('-', '=').replace('#', '@') + prof[top + 1:]

        if s2top > BREAK:
            urteil = 'Gipfel gebrochen'
        elif max(s2) > BREAK:
            urteil = '*** bricht, aber NICHT am Gipfel ***'
        else:
            urteil = 'nirgends gebrochen'
        print(f'{rx:<9}{prof:<44}{top:>7}{s2top:>11.3f}'
              f'{(f"{g:.3f}" if g is not None else "—"):>10}   {urteil}')
        rows.append((rx, s2top, max(s2), g))
        tops.append(s2top)

    print()
    print('  Legende:  . <0.05   - 0.05-0.3   # >0.3      '
          'o = @ markieren den Gipfel')
    print()

    checks.sentinel(tops, '<S^2> am Gipfel')

    a = [r for r in rows if r[1] > BREAK]
    b_ = [r for r in rows if r[1] <= BREAK < r[2]]
    c = [r for r in rows if r[2] <= BREAK]
    print()
    print(f'  Gipfel gebrochen             {len(a):>3} von {len(rows)}')
    print(f'  bricht, aber nicht am Gipfel {len(b_):>3}   '
          f'{" ".join(r[0] for r in b_)}')
    print(f'  nirgends gebrochen           {len(c):>3}   '
          f'{" ".join(r[0] for r in c)}')

    print()
    print('=' * 100)
    print('HAENGT DAS ERGEBNIS DES NEB DARAN')
    print('=' * 100)
    print('Ein Band, dessen Gipfel auf der restringierten Flaeche liegt, sollte')
    print('gegen den restringierten Sattelpunkt laufen.')
    print()
    print(f'{"Gruppe":<32}{"n":>4}{"Grad NEB median":>17}'
          f'{"stationaer":>12}{"RMSD zum RKS-TS":>18}')
    print('-' * 100)
    for lab, grp in (('Gipfel gebrochen', a),
                     ('Gipfel restringiert', b_ + c)):
        gs = [r[3] for r in grp if r[3] is not None]
        ds = []
        for r in grp:
            g = (glob.glob(f'{H}/bs_uks_neb_results/{r[0]}/*NEB-TS_converged.xyz')
                 + glob.glob(f'{H}/bs_uks_neb_results/{r[0]}/*NEB-CI_converged.xyz'))
            if g:
                d = kabsch(rd(g[0]),
                           rd(f'{H}/orca_neb_results/{r[0]}/transition_state.xyz'))
                if d is not None:
                    ds.append(d)
        print(f'{lab:<32}{len(grp):>4}'
              f'{(f"{np.median(gs):.3f}" if gs else "—"):>17}'
              f'{sum(1 for x in gs if x < 0.15):>7} von {len(gs):<4}'
              f'{(f"{np.median(ds):.4f}" if ds else "—"):>18}')

    print()
    print('Einzeln, damit die Gruppen nachpruefbar bleiben:')
    for lab, grp in (('Gipfel gebrochen', a), ('Gipfel restringiert', b_ + c)):
        print(f'  {lab}:')
        for rx, s2t, s2m, g in grp:
            print(f'     {rx:<9} <S^2>_Gipfel {s2t:.3f}  max {s2m:.3f}  '
                  f'Grad {"—" if g is None else f"{g:.3f}"}')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
