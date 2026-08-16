"""Is the barrier top of a mixed band an artefact of the seam between sheets?

The profile analysis found that whether the NEB converges is decided by the
spin state of its highest image, not by whether the band broke somewhere, and
that in all four bands which broke away from the top, the top sits exactly one
image beside the broken region.

The proposed reason: a broken image lies lower in energy than the restricted
solution at the same geometry -- that is what "broken is the ground state"
means.  A band whose images sit on different sheets therefore has a depression
over the broken stretch, and the maximum of that mixed curve is pushed onto a
restricted image at the edge.  The climbing image then climbs the seam instead
of the barrier.

This checks the claim on the energies rather than asserting it:

  1. how far, in images, is the maximum from the nearest seam
  2. how large is the energy step across a seam against the steps within a sheet
  3. does the broken stretch sit lower than a smooth path through it would

Point 3 is the one that could refute it.  If the broken images are not
depressed relative to their neighbours, the seam cannot be creating a false
maximum and the correlation from the profile analysis needs another
explanation.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checks

H = '/home/energy/s242862'
HA_MEV = 27211.386
BREAK = 0.3


def band(rx):
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
    out.sort()
    return out or None


def main():
    # same restriction as band_profile.py: the directory holds three
    # single-reference runs that do not belong in an MR statistic
    MR = ['rxn7949', 'rxn8832', 'rxn1320', 'rxn4113', 'rxn8885', 'rxn6196',
          'rxn0346', 'rxn4518', 'rxn3107', 'rxn8837', 'rxn7060', 'rxn5691',
          'rxn1283', 'rxn8827', 'rxn4522', 'rxn1147', 'rxn0894', 'rxn7957',
          'rxn5690']
    measured = {os.path.basename(os.path.dirname(p))
                for p in glob.glob(f'{H}/band_s2_v2/rxn*/band_s2.txt')}
    rxns = [r for r in MR if r in measured]
    checks.expect(rxns, len(MR), 'MR-Reaktionen mit gemessenem Band',
                  warn_only=True)
    checks.header(__file__, inputs=[f'{H}/band_s2_v2'],
                  note='Energien und <S^2> je Bild aus der validierten '
                       'Nachmessung.')

    print('=' * 104)
    print('1  DAS ENERGIEPROFIL, BILDWEISE, MIT BLATTZUGEHOERIGKEIT')
    print('=' * 104)
    print('Energie relativ zu Bild 0, in meV.  B = gebrochen (<S^2> > 0.3),')
    print('r = restringiert.  Der Gipfel ist markiert.')
    print()

    seam_steps, inner_steps, dists, mixed = [], [], [], []
    for rx in rxns:
        b = band(rx)
        if not b:
            continue
        s2 = np.array([x[1] for x in b])
        e = np.array([x[2] for x in b])
        e = (e - e[0]) * HA_MEV
        lab = np.where(s2 > BREAK, 1, 0)
        top = int(np.argmax(e))
        if lab.min() == lab.max():
            continue                      # not mixed, nothing to say about seams
        mixed.append(rx)

        seams = [i for i in range(len(lab) - 1) if lab[i] != lab[i + 1]]
        d = min(abs(top - s) for s in seams) if seams else None
        if d is not None:
            dists.append(d)

        print(f'  {rx}')
        row1 = '    Blatt  ' + ' '.join(
            ('B' if l else 'r') for l in lab)
        row2 = '    E      ' + ' '.join(f'{v:>6.0f}' for v in e)
        # the label row has to line up with the six-wide energy columns
        row1 = '    Blatt  ' + ' '.join(
            f'{("B" if l else "r"):>6}' for l in lab)
        row3 = '    Gipfel ' + ' '.join(
            f'{("^^" if i == top else ""):>6}' for i in range(len(e)))
        print(row1)
        print(row2)
        print(row3)
        print(f'    Naht bei Bild {seams}, Gipfel bei {top}, '
              f'Abstand {d}')

        for i in range(len(e) - 1):
            step = abs(e[i + 1] - e[i])
            (seam_steps if i in seams else inner_steps).append(step)
        print()

    print('=' * 104)
    print('2  WIE GROSS IST DER SCHRITT UEBER DIE NAHT')
    print('=' * 104)
    if seam_steps and inner_steps:
        print(f'  ueber eine Naht      n={len(seam_steps):<3} '
              f'median {np.median(seam_steps):8.1f} meV   '
              f'max {max(seam_steps):8.1f}')
        print(f'  innerhalb eines Blatts n={len(inner_steps):<3} '
              f'median {np.median(inner_steps):8.1f} meV   '
              f'max {max(inner_steps):8.1f}')
        r = np.median(seam_steps) / max(np.median(inner_steps), 1e-9)
        print(f'  Verhaeltnis der Mediane  {r:.2f}')
        print()
        if r > 1.5:
            print('  Der Schritt ueber die Naht ist groesser -- konsistent mit')
            print('  einem Versatz zwischen den Blaettern.')
        else:
            print('  Kein auffaelliger Sprung an der Naht.  Die Erklaerung')
            print('  ueber ein kuenstliches Maximum traegt so nicht.')

    print()
    print('=' * 104)
    print('3  LIEGT DER GIPFEL AN DER NAHT')
    print('=' * 104)
    if dists:
        print(f'  gemischte Baender: {len(mixed)}')
        print(f'  Abstand Gipfel zur naechsten Naht, in Bildern:')
        for d in sorted(set(dists)):
            print(f'     {d}:  {dists.count(d)}x')
        print(f'  median {np.median(dists):.1f}')
        print()
        if np.median(dists) <= 1:
            print('  Der Gipfel sitzt an oder neben der Naht.  Bei acht Bildern')
            print('  ist das genau die Aufloesung, bei der sich Naht und Sattel')
            print('  nicht trennen lassen.')

    print()
    print('=' * 104)
    print('4  SITZT DIE GEBROCHENE STRECKE TIEFER ALS EIN GLATTER PFAD')
    print('=' * 104)
    print('Fuer jede gebrochene Strecke: die Energie ihrer Bilder gegen die')
    print('Gerade zwischen den beiden restringierten Nachbarn.  Negative Werte')
    print('heissen, die gebrochene Loesung zieht das Profil nach unten -- die')
    print('Voraussetzung fuer ein kuenstliches Maximum daneben.')
    print()
    print(f'  {"rxn":<9}{"Bilder der Strecke":<22}{"max Absenkung [meV]":>22}')
    print('  ' + '-' * 60)
    drops = []
    for rx in mixed:
        b = band(rx)
        s2 = np.array([x[1] for x in b])
        e = np.array([x[2] for x in b])
        e = (e - e[0]) * HA_MEV
        lab = s2 > BREAK
        i = 0
        while i < len(lab):
            if not lab[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(lab) and lab[j + 1]:
                j += 1
            a, c = i - 1, j + 1
            if a >= 0 and c < len(e):
                t = np.linspace(e[a], e[c], c - a + 1)[1:-1]
                dev = (e[i:j + 1] - t).min()
                drops.append(dev)
                print(f'  {rx:<9}{f"{i}-{j}":<22}{dev:>22.1f}')
            else:
                print(f'  {rx:<9}{f"{i}-{j}":<22}{"Rand, kein Vergleich":>22}')
            i = j + 1
    if drops:
        print()
        print(f'  n={len(drops)}   median {np.median(drops):.1f} meV   '
              f'negativ in {sum(1 for d in drops if d < 0)} von {len(drops)}')
        print()
        if np.median(drops) < 0:
            print('  Bestaetigt: die gebrochene Strecke liegt unter der')
            print('  Verbindungslinie ihrer Nachbarn.')
        else:
            print('  NICHT bestaetigt: keine Absenkung.  Die Erklaerung ueber')
            print('  das kuenstliche Maximum ist damit widerlegt.')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
