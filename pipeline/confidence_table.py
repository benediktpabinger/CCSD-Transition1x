"""How confident are we, per reaction, that the broken-symmetry structure we
found is the relevant saddle point?

A frequency calculation proves a structure is a first-order saddle point. It
does not prove it is the saddle point the reaction goes through -- that is a
global question and the analysis is local. Four pieces of evidence bear on it:

  freq        exactly one imaginary frequency: it is a saddle at all
  NEB         an independent path search from relaxed endpoints found the same
              structure; a path search explores more of the surface than an
              optimisation started at one point, so agreement is the strongest
              evidence available here
  Richtung    whether the optimisation moved toward stronger or weaker spin
              symmetry breaking. Moving toward weaker while some model geometry
              of the same reaction shows much stronger breaking means the
              optimisation went away from a region that may hold another saddle
  2. Becken   the ratio between the strongest dE_BS across the four geometries
              and the one at the reference. Large means the surface has a
              strongly broken region the reference-started optimisation may
              never have visited

Noise floor from the null measurement is 0.125 A: displacements below that are
not distinguishable from method scatter.
"""
import glob
import json
import os

H = '/home/energy/s242862'
NOISE = 0.125
res = sorted(json.load(open(f'{H}/fod_ranking.json'))['results'],
             key=lambda r: -r['nfod'])
n = len(res)
TOP = [res[i]['rxn'] for i in range(26)]
MID = [res[i - 1]['rxn'] for i in [11, 40, 68, 97, 126, 154, 183, 212, 240, 269]]
LOW = [res[i]['rxn'] for i in range(n - 10, n)]
grp = {}
for r in TOP: grp[r] = 'high'
for r in MID: grp.setdefault(r, 'mid')
for r in LOW: grp.setdefault(r, 'low')
nf = {x['rxn']: x['nfod'] for x in res}

import numpy as np


def read_xyz(p):
    L = open(p).read().split('\n')
    m = int(L[0].split()[0])
    return np.array([[float(v) for v in l.split()[1:4]] for l in L[2:2 + m]])


def kabsch(A, B):
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    V, S, W = np.linalg.svd(Ac.T @ Bc)
    D = np.diag([1., 1., np.sign(np.linalg.det(V @ W))])
    return float(np.sqrt((((Ac @ (V @ D @ W)) - Bc) ** 2).sum() / len(A)))


def ts_file(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        for pat in ('ts', 'final', 'opt'):
            for f in glob.glob(f'{H}/{d}/{rx}/*.xyz'):
                if pat in os.path.basename(f).lower():
                    return f, d
    return None, None


def freq_of(rx):
    for d in ('bs_freq', 'bs_freq_v2', 'bs_freq_fromneb'):
        p = f'{H}/{d}/{rx}/result.json'
        if os.path.exists(p):
            j = json.load(open(p))
            if 'n_imag' in j:
                return j['n_imag'], (j.get('imag_freq') or [None])[0]
    return None, None


def s2_final(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        p = f'{H}/{d}/{rx}/result.json'
        if os.path.exists(p):
            j = json.load(open(p))
            if j.get('s2_final') is not None:
                return j['s2_final']
    return None


rows = []
for rx in sorted(grp, key=lambda r: -nf[r]):
    p = f'{H}/stab_pipeline/{rx}/result.json'
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    ref = next((g for g in d['geometries'] if g['source'] == 'RKS-ref'), None)
    if not ref or ref.get('ext_stable') is not False:
        continue
    bsr = ref.get('bs') or {}
    ref_de, ref_s2 = abs(bsr.get('de_meV') or 0), bsr.get('s2') or 0
    strongest = 0.0
    for g in d['geometries']:
        if g['source'] == 'RKS-ref':
            continue
        b = g.get('bs') or {}
        strongest = max(strongest, abs(b.get('de_meV') or 0))
    ratio = strongest / ref_de if ref_de > 0.01 else None

    ni, im = freq_of(rx)
    s2f = s2_final(rx)
    tsf, src = ts_file(rx)
    rmsd_ref = None
    if tsf:
        r = f'{H}/orca_neb_results/{rx}/transition_state.xyz'
        if os.path.exists(r):
            rmsd_ref = kabsch(read_xyz(tsf), read_xyz(r))
    neb = f'{H}/bs_uks_neb_results/{rx}/bs_uks_neb_NEB-TS_converged.xyz'
    d_neb = kabsch(read_xyz(tsf), read_xyz(neb)) if (tsf and os.path.exists(neb)) else None

    # confidence
    if ni != 1:
        conf, why = 'offen', 'kein bestaetigter Sattelpunkt'
    elif d_neb is not None and d_neb < 0.15:
        conf, why = 'hoch', f'NEB bestaetigt ({d_neb:.3f} A)'
    elif ratio and ratio > 20 and s2f is not None and s2f < ref_s2:
        conf, why = 'niedrig', f'Optimierung zu schwaecherer BS, Modelle {ratio:.0f}x staerker'
    elif ratio and ratio > 20:
        conf, why = 'mittel', f'2. Becken moeglich ({ratio:.0f}x)'
    elif rmsd_ref is not None and rmsd_ref < NOISE:
        conf, why = 'mittel', 'Verschiebung im Rauschen, geometrisch folgenlos'
    else:
        conf, why = 'mittel', 'Frequenz bestaetigt, keine unabhaengige Gegenprobe'
    rows.append({'rxn': rx, 'grp': grp[rx], 'ref_de': ref_de, 'ref_s2': ref_s2,
                 's2f': s2f, 'ratio': ratio, 'ni': ni, 'im': im,
                 'rmsd_ref': rmsd_ref, 'd_neb': d_neb, 'src': src,
                 'conf': conf, 'why': why})

ORDER = {'hoch': 0, 'mittel': 1, 'niedrig': 2, 'offen': 3}
rows.sort(key=lambda r: (ORDER[r['conf']], -(r['ref_de'])))

print(f"{'rxn':<9}{'grp':<5}{'dE_ref':>8}{'S2_ref':>7}{'S2_end':>7}"
      f"{'v_imag':>8}{'RMSD_ref':>9}{'d(NEB)':>8}{'Faktor':>8}  {'Sicherheit':<9} Begruendung")
print('-' * 122)
for r in rows:
    def f(v, p=3):
        return '--' if v is None else '{:.{}f}'.format(v, p)
    im = '--' if r['im'] is None else '{:.0f}'.format(r['im'])
    ra = '--' if r['ratio'] is None else '{:.0f}x'.format(r['ratio'])
    print('{:<9}{:<5}{:>8.1f}{:>7.3f}{:>7}{:>8}{:>9}{:>8}{:>8}  {:<9} {}'.format(
        r['rxn'], r['grp'], r['ref_de'], r['ref_s2'], f(r['s2f']), im,
        f(r['rmsd_ref']), f(r['d_neb']), ra, r['conf'], r['why']))

print()
for c in ('hoch', 'mittel', 'niedrig', 'offen'):
    s = [r['rxn'] for r in rows if r['conf'] == c]
    print(f'  {c:<9} {len(s):>2}   ' + ' '.join(s))
