"""Four additive tables for the chapter.  Nothing existing is redefined.

The stage-3 machinery is NOT reimplemented here.  An earlier version of this
script parsed ORCA's $normal_modes (cartesian displacements) and disagreed with
sweep_summary.txt on 6 of ~30 verdicts -- because sweep_summary mass-weights the
Hessian, projects out translations and rotations, and normalises the resulting
eigenvector.  Hydrogen against carbon is a factor 3.5 in amplitude, so the two
"fractions" are different quantities, and the thresholds 0.10 / 0.05 are
calibrated to the mass-weighted one.

So this script executes the definition part of sweep_summary.py and calls its
functions.  That is uglier than importing, but sweep_summary.py writes its
report at module level, so importing it would run the whole thing.  Cutting at
the first top-level print gives the definitions and nothing else, and it
guarantees the two files cannot drift apart.
"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checks

H = '/home/energy/s242862'
HA_EV = 27.211386245988

_src = open(f'{H}/sweep_summary.py', errors='replace').read().split('\n')
_cut = next(i for i, l in enumerate(_src) if l.startswith('print('))
exec('\n'.join(_src[:_cut]), globals())
for _n in ('read_xyz', 'read_orca_hess', 'analyse', 'reactive', 'orca',
           'stab_grad', 'MODELDIR', 'GRAD_OK', 'FRAC_MIN', 'RATE_MIN', 'MR'):
    assert _n in globals(), f'sweep_summary.py no longer defines {_n}'

CANDS = [('nebts_%s', 'UKS-NEB'), ('ours_%s', 'unsere'),
         ('%s_UMA-S', 'UMA-S'), ('%s_UMA-M', 'UMA-M'), ('%s_eSEN', 'eSEN'),
         ('tsopt_%s_UMA-M', 'TSopt/M')]

MRO = ['rxn7949', 'rxn8832', 'rxn1320', 'rxn4113', 'rxn8885', 'rxn6196',
       'rxn0346', 'rxn4518', 'rxn3107', 'rxn8837', 'rxn7060', 'rxn5691',
       'rxn1283', 'rxn8827', 'rxn4522', 'rxn1147', 'rxn0894', 'rxn7957',
       'rxn5690']


def locate(label, rx):
    """(hessian path, geometry path, gradient) for a candidate label.

    Three storage layouts had to be covered.  orca_freq is the sweep; orca_irc
    holds our structures at rxn1147 and rxn7957 because their Hessians were
    computed inside the IRC job; freq_at_model holds the model geometries as a
    numpy Hessian with the geometry in the model's own NEB directory.  Missing
    any of the three silently drops cells -- and the ones it drops are not
    random, they are the reactions that got extra attention.
    """
    o = orca(label)
    if o:
        d = (f'{H}/orca_freq/{label}' if os.path.isdir(f'{H}/orca_freq/{label}')
             else f'{H}/orca_irc/{label}')
        return o['hess'], f'{d}/start.xyz', o['grad']
    m = re.match(r'ours_(rxn\d+)$', label)
    if m:
        d = f'{H}/orca_irc/{m.group(1)}_ours'
        if os.path.exists(f'{d}/numfreq.hess'):
            o2 = orca(f'{m.group(1)}_ours')
            return (f'{d}/numfreq.hess', f'{d}/start.xyz',
                    o2['grad'] if o2 else None)
    m = re.match(r'(rxn\d+)_(UMA-S|UMA-M|eSEN)$', label)
    if m:
        rxx, mdl = m.group(1), m.group(2)
        p = f'{H}/freq_at_model/{label}/hessian.npy'
        g = f'{H}/{MODELDIR[mdl]}/{rxx}/transition_state.xyz'
        if os.path.exists(p) and os.path.exists(g):
            return p, g, stab_grad(rxx, mdl)
    return None, None, None


def assess(label, rx):
    """(stage, gradient, n_imag, imag, frac, rate); stage in none/a/b/c."""
    hp, gp, g = locate(label, rx)
    if hp is None:
        return (None,) * 6
    if g is None:
        return 'nograd', None, None, None, None, None
    if g >= GRAD_OK:
        return 'none', g, None, None, None, None
    sym, xyz = read_xyz(gp)
    hs = read_orca_hess(hp) if hp.endswith('.hess') else np.load(hp)
    a = analyse(hs, sym, xyz, reactive(rx))
    nim, im = a['n_imag'], a['imag']
    if nim != 1:
        return 'a', g, nim, im, a.get('frac'), a.get('maxrate')
    fr, rt = a.get('frac'), a.get('maxrate')
    if fr is None:
        return 'b', g, nim, im, None, None
    return ('c' if (fr >= FRAC_MIN and rt >= RATE_MIN) else 'b'), g, nim, im, fr, rt


def fmt(v, w, d=3):
    return f'{v:>{w}.{d}f}' if isinstance(v, (int, float)) else f'{"—":>{w}}'


def kabsch(A, B):
    if A is None or B is None or len(A) != len(B):
        return None
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    V, S, W = np.linalg.svd(Ac.T @ Bc)
    D = np.diag([1., 1., np.sign(np.linalg.det(V @ W))])
    return float(np.sqrt((((Ac @ (V @ D @ W)) - Bc) ** 2).sum() / len(A)))


def xyz_of(p):
    return read_xyz(p)[1] if os.path.exists(p) else None


def reactant_energy(rx):
    for f in ('sp.out', 'rks.out'):
        p = f'{H}/orca_endpoint/{rx}_reactant/{f}'
        if os.path.exists(p):
            m = re.findall(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)',
                           open(p, errors='replace').read())
            if m:
                return float(m[-1])
    return None


def main():
    checks.header(__file__,
                  inputs=[f'{H}/orca_freq', f'{H}/orca_irc',
                          f'{H}/freq_at_model', f'{H}/bs_uks_neb_results',
                          f'{H}/sweep_summary.py', f'{H}/sweep_summary.txt'],
                  note='Stufe 3 kommt aus sweep_summary.py, nicht aus einer '
                       'zweiten Implementierung.')

    print('=' * 96)
    print('A  DIE FEHLENDEN ZELLEN IN T1')
    print('=' * 96)
    print(f'{"Zelle":<24}{"Grad":>8}{"n_imag":>8}{"v_imag":>10}{"Anteil":>8}'
          f'{"Rate":>8}{"Barriere":>10}  Stufe')
    print('-' * 96)
    for rx in ('rxn1147', 'rxn7957'):
        st, g, nim, im, fr, rt = assess(f'ours_{rx}', rx)
        hp, _, _ = locate(f'ours_{rx}', rx)
        e0 = reactant_energy(rx)
        d = os.path.dirname(hp) if hp else None
        e = None
        if d and os.path.exists(f'{d}/bs_sp.out'):
            m = re.findall(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)',
                           open(f'{d}/bs_sp.out', errors='replace').read())
            e = float(m[-1]) if m else None
        b = (e - e0) * HA_EV if (e is not None and e0 is not None) else None
        print(f'{rx + " / unsere":<24}{fmt(g, 8)}'
              f'{nim if nim is not None else "—":>8}{fmt(im, 10, 1)}'
              f'{fmt(fr, 8, 2)}{fmt(rt, 8, 3)}{fmt(b, 10)}  {st}')

    print()
    print('Nach Einbeziehen von orca_irc und freq_at_model noch fehlend:')
    gone = 0
    for rx in MRO:
        for pat, nm in CANDS:
            if locate(pat % rx, rx)[0] is None:
                gone += 1
                print(f'   {rx:<9} {nm:<9} ({pat % rx})')
    print(f'   {gone} Zellen von {len(MRO) * len(CANDS)}')

    print()
    print('=' * 96)
    print('B  KUMULATIVE DREISTUFENBILANZ')
    print('=' * 96)
    print(f'a  stationaer, Gradient < {GRAD_OK} eV/A')
    print('b  zusaetzlich genau eine imaginaere Mode')
    print(f'c  zusaetzlich Modenanteil >= {FRAC_MIN} und Rate >= {RATE_MIN}')
    print()
    print(f'{"Methode":<12}{"geprueft":>10}{"Stufe a":>10}{"Stufe b":>10}'
          f'{"Stufe c":>10}   verloren a->b   b->c')
    print('-' * 96)
    detail = {}
    for pat, nm in CANDS:
        n = a = b = c = 0
        rows = []
        for rx in MRO:
            st, g, nim, im, fr, rt = assess(pat % rx, rx)
            if st is None:
                continue
            n += 1
            a += st in ('a', 'b', 'c')
            b += st in ('b', 'c')
            c += st == 'c'
            rows.append((rx, st, g, nim, im, fr, rt))
        detail[nm] = rows
        print(f'{nm:<12}{n:>10}{a:>10}{b:>10}{c:>10}{a - b:>16}{b - c:>7}')

    print()
    # The comparison that found the mass-weighting bug.  It stays, and it is
    # now the shared guard rather than a hand-rolled loop, so a future table
    # cannot quietly skip it.
    sw = {}
    for line in open(f'{H}/sweep_summary.txt', errors='replace'):
        m = re.match(r'\s*(rxn\d+)\s+(UMA-S|UMA-M|eSEN)\s+.*?(\S.*\S)\s*$', line)
        if m:
            sw[f'{m.group(1)}_{m.group(2)}'] = 'CLEARS ALL THREE' in m.group(3)
        m = re.match(r'\s*((?:nebts|tsopt)_\S+)\s+.*?(\S.*\S)\s*$', line)
        if m:
            sw[m.group(1)] = 'CLEARS ALL THREE' in m.group(2)
    mine = {pat % rx: (st == 'c')
            for pat, nm in CANDS for rx, st, *_ in detail[nm]}
    checks.crosscheck(mine, sw, 'Stufe-c-Urteile gegen sweep_summary.txt')

    print()
    print('Zeilen, die Stufe b bestehen und an Stufe c scheitern:')
    for pat, nm in CANDS:
        for rx, st, g, nim, im, fr, rt in detail[nm]:
            if st == 'b' and fr is not None:
                print(f'   {rx:<9} {nm:<9} v {im:>8.1f}  Anteil {fr:.2f}  Rate {rt:.3f}')

    print()
    print('=' * 96)
    print('C  TRIAGE, NEU AUSGEZAEHLT GEGEN STUFE c')
    print('=' * 96)
    print(f'{"Lauf":<26}{"Start":>8}{"Ergebnis":>10}{"v_imag":>10}{"Anteil":>8}'
          f'{"Rate":>8}  Stufe')
    print('-' * 96)
    lo, hi, nog = [], [], []
    for d0 in sorted(glob.glob(f'{H}/orca_freq/tsopt_rxn*')):
        lbl = os.path.basename(d0)
        m = re.match(r'tsopt_(rxn\d+)_(.+)$', lbl)
        rx, mdl = m.group(1), m.group(2)
        gstart = locate(f'{rx}_{mdl}', rx)[2]
        st, g, nim, im, fr, rt = assess(lbl, rx)
        print(f'{lbl:<26}{fmt(gstart, 8)}{fmt(g, 10)}{fmt(im, 10, 1)}'
              f'{fmt(fr, 8, 2)}{fmt(rt, 8, 3)}  {st}')
        (nog if gstart is None else (lo if gstart < 0.25 else hi)).append(
            (lbl, gstart, st))
    for name, bucket in (('unter 0.25 eV/A', lo), ('ueber 0.25 eV/A', hi),
                         ('Startgradient unbekannt', nog)):
        ok = [x for x in bucket if x[2] == 'c']
        print(f'\n{name}:  {len(ok)} von {len(bucket)} bestehen Stufe c')
        for lbl, gs, st in bucket:
            print(f'     {"JA  " if st == "c" else "NEIN"} {lbl:<26}'
                  f'Start {fmt(gs, 6)}  -> {st}')

    print()
    print('=' * 96)
    print('D  ABSTAND DES UKS-NEB VOM RKS-TS')
    print('=' * 96)
    print(f'{"rxn":<9}{"NEB vs RKS-TS":>15}{"NEB vs unsere":>15}{"Grad NEB":>10}')
    print('-' * 96)
    vals = []
    for rx in MRO + ['rxn1150', 'rxn7936', 'rxn7945']:
        g = (glob.glob(f'{H}/bs_uks_neb_results/{rx}/*NEB-TS_converged.xyz')
             + glob.glob(f'{H}/bs_uks_neb_results/{rx}/*NEB-CI_converged.xyz'))
        if not g:
            continue
        nb = xyz_of(g[0])
        ref = xyz_of(f'{H}/orca_neb_results/{rx}/transition_state.xyz')
        hp, gp, _ = locate(f'ours_{rx}', rx)
        ours = xyz_of(gp) if gp else None
        r1 = kabsch(nb, ref)
        print(f'{rx:<9}{fmt(r1, 15, 4)}{fmt(kabsch(nb, ours), 15, 4)}'
              f'{fmt(assess(f"nebts_{rx}", rx)[1], 10)}')
        if r1 is not None:
            vals.append((r1, rx))
    if vals:
        vals.sort()
        v = [x for x, _ in vals]
        print()
        # An RMSD of exactly zero here would mean the two files are the same
        # structure, i.e. a provenance mix-up rather than agreement.
        checks.sentinel(v, 'RMSD NEB gegen RKS-TS')
        print(f'n = {len(v)}   Median {np.median(v):.4f} A   '
              f'unter 0.10 A: {sum(1 for x in v if x < 0.10)}')
        print('am naechsten: ' + ', '.join(f'{r} {x:.4f}' for x, r in vals[:6]))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
