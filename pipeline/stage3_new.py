"""Stage 3 for every saddle candidate produced overnight.

Seven bands came out of the NEB-CI split with one imaginary mode each, plus
results from the model-path runs, the 16-image runs and the optimisations
started from the highest broken image.  Convergence and a single imaginary
frequency are stages 1 and 2; they were satisfied three times tonight by points
that stage 3 then rejected -- torsions at -47, -69 and -76 cm-1, one of them in
the reactant basin with both reactive bonds still at educt values.

So none of it counts until the imaginary mode is projected onto the bonds that
this reaction actually breaks and forms.

The projection is not reimplemented.  sweep_summary.py mass-weights the
Hessian, projects out translations and rotations and normalises the eigenvector
before measuring; a cartesian version of the same quantity disagreed with it on
6 of 30 verdicts earlier, because hydrogen against carbon is a factor 3.5 in
amplitude.  The thresholds 0.10 and 0.05 belong to the mass-weighted
definition, so its functions are executed from source rather than copied.
"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checks

H = '/home/energy/s242862'

_src = open(f'{H}/sweep_summary.py', errors='replace').read().split('\n')
_cut = next(i for i, l in enumerate(_src) if l.startswith('print('))
exec('\n'.join(_src[:_cut]), globals())
for _n in ('read_xyz', 'read_orca_hess', 'analyse', 'reactive',
           'FRAC_MIN', 'RATE_MIN'):
    assert _n in globals(), f'sweep_summary.py definiert {_n} nicht mehr'

# where the overnight runs put their results, and which file carries the
# optimised structure for each
SETS = [
    ('NEB-CI aufgeteilt', f'{H}/bs_uks_nebci', 'tsopt'),
    ('vom hoechsten gebrochenen Bild', f'{H}/tsopt_broken', 'tsopt'),
    ('Modellpfad', f'{H}/bs_uks_neb_modelpath', 'neb'),
    ('16 Bilder', f'{H}/bs_uks_neb16', 'neb'),
    ('Produktionsniveau wB97M-V/def2-TZVP', f'{H}/bs_uks_nebci_prod',
     'tsopt2'),
]


def orca_nimag(d, kind):
    """How many imaginary modes ORCA itself reports, from the last block.

    Counting them from a re-diagonalised Hessian does not work here.  The
    projection in sweep_summary.py leaves residual rotational modes down to
    about -24 cm-1 on these structures, and its threshold sits at -20, so a
    clean first-order saddle comes out as second-order.  For rxn1320 ORCA
    prints six exact zeros and one mode at -404.3, while the re-diagonalisation
    gives -403.9 and -23.6.

    The projection is still trustworthy for the eigenVECTOR, which is what
    stage 3 needs -- the mode fraction and the bond rate are unaffected by a
    stray mode 17 times softer.  So the count comes from ORCA and the direction
    from the Hessian.
    """
    p = f'{d}/{kind}.out' if kind.startswith('tsopt') else None
    if p is None or not os.path.exists(p):
        return None
    t = open(p, errors='replace').read()
    i = t.rfind('VIBRATIONAL FREQUENCIES')
    if i < 0:
        return None
    fr = [float(m.group(1)) for m in
          re.finditer(r'^\s*\d+:\s+(-?\d+\.\d+)\s+cm', t[i:], re.M)]
    return sum(1 for v in fr if v < -1.0)


def last_hess(d, kind):
    """The Hessian of the final structure, not of the initial guess.

    OptTS writes the starting Hessian from Calc_Hess as well; taking the first
    one makes a converged saddle look like a higher-order one.
    """
    cands = ([f'{d}/{kind}.hess'] if kind.startswith('tsopt')
             else sorted(glob.glob(f'{d}/*.hess')))
    for p in cands:
        if os.path.exists(p):
            return p
    return None


def geom_for(d, kind):
    if kind.startswith('tsopt'):
        for f in (f'{kind}.xyz',):
            if os.path.exists(f'{d}/{f}'):
                return f'{d}/{f}'
    for pat in ('*NEB-TS_converged.xyz', '*NEB-CI_converged.xyz'):
        g = glob.glob(f'{d}/{pat}')
        if g:
            return g[0]
    return None


def main():
    checks.header(__file__,
                  inputs=[s[1] for s in SETS] + [f'{H}/sweep_summary.py'],
                  note=f'Stufe 3: Modenanteil >= {FRAC_MIN}, '
                       f'Bindungsrate >= {RATE_MIN}, massengewichtet.')

    total = {}
    for label, root, kind in SETS:
        print()
        print('=' * 92)
        print(label)
        print('=' * 92)
        print(f'{"rxn":<12}{"n_imag":>7}{"v_imag":>10}{"Anteil":>8}{"Rate":>8}'
              f'   Urteil')
        print('-' * 92)
        rows = []
        for d in sorted(glob.glob(f'{root}/*/')):
            name = os.path.basename(d.rstrip('/'))
            m = re.search(r'(rxn\d+)', name)
            if not m:
                continue
            rx = m.group(1)
            hp = last_hess(d.rstrip('/'), kind)
            gp = geom_for(d.rstrip('/'), kind)
            if hp is None or gp is None:
                print(f'{name:<12}{"—":>7}{"—":>10}{"—":>8}{"—":>8}'
                      f'   keine Hesse oder Struktur')
                continue
            sym, xyz = read_xyz(gp)
            a = analyse(read_orca_hess(hp), sym, xyz, reactive(rx))
            fr, rt = a.get('frac'), a.get('maxrate')
            im = a['imag']
            nim_orca = orca_nimag(d.rstrip('/'), kind)
            nim = nim_orca if nim_orca is not None else a['n_imag']
            src = '' if nim_orca is not None else '  (aus der Hesse gezaehlt)'
            if nim != 1:
                v = f'{nim} imaginaere Moden{src}'
            elif fr is None:
                v = 'keine reaktiven Bindungen hinterlegt'
            elif fr >= FRAC_MIN and rt >= RATE_MIN:
                v = '*** BESTEHT ALLE DREI STUFEN ***'
            else:
                v = 'Mode gehoert nicht zu dieser Reaktion'
            print(f'{name:<12}{nim:>7}{im:>10.1f}'
                  f'{(f"{fr:.2f}" if fr is not None else "—"):>8}'
                  f'{(f"{rt:.3f}" if rt is not None else "—"):>8}   {v}')
            rows.append((rx, v.startswith('***')))
        ok = sum(1 for _, k in rows if k)
        total[label] = (ok, len(rows))
        print(f'\n  {ok} von {len(rows)} bestehen alle drei Stufen')

    print()
    print('=' * 92)
    print('BILANZ')
    print('=' * 92)
    for k, (a, b) in total.items():
        print(f'  {k:<36} {a:>3} von {b}')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
