"""Does external SCF stability of the reference separate model TS accuracy
better than the FOD multireference diagnostic?

Uses only existing data: stability_pipeline output (job 10691631), the model NEB
transition states, and the confirmed BS-optimised TS geometries (job 10692887).
No new quantum chemistry.
"""
import glob
import json
import os

import numpy as np

H = '/home/energy/s242862'
MODELS = {'UMA-S': 'uma_neb_results', 'UMA-M': 'uma_m_neb_results',
          'eSEN': 'esen_neb_results', 'MACE': 'mace_bare_neb_results',
          'MACE+delta': 'mace_delta_neb_results_fw2'}
THR = 0.3


# ---------------------------------------------------------------- geometry
def read_xyz(p):
    L = open(p).read().split('\n')
    n = int(L[0].split()[0])
    sym, xyz = [], []
    for line in L[2:2 + n]:
        f = line.split()
        sym.append(f[0])
        xyz.append([float(x) for x in f[1:4]])
    return sym, np.array(xyz)


def kabsch(A, B):
    A, B = A - A.mean(0), B - B.mean(0)
    V, S, W = np.linalg.svd(A.T @ B)
    D = np.diag([1., 1., np.sign(np.linalg.det(V @ W))])
    return float(np.sqrt((((A @ (V @ D @ W)) - B) ** 2).sum() / len(A)))


def rmsd_files(p, q):
    if not (os.path.exists(p) and os.path.exists(q)):
        return None, 'Datei fehlt'
    s1, x1 = read_xyz(p)
    s2, x2 = read_xyz(q)
    if s1 != s2:
        return None, 'Atomreihenfolge/Zusammensetzung weicht ab'
    return kabsch(x1, x2), None


# ---------------------------------------------------------------- stats
def stats(v):
    v = np.asarray([x for x in v if x is not None], float)
    if not len(v):
        return None
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    return {'n': len(v), 'median': q2, 'mean': v.mean(), 'min': v.min(),
            'max': v.max(), 'q1': q1, 'q3': q3,
            'n_gt': int((v > THR).sum())}


def prow(label, s):
    if s is None:
        print(f'{label:<26} --')
        return
    print(f"{label:<26}{s['n']:>5}{s['median']:>9.4f}{s['mean']:>9.4f}"
          f"{s['min']:>9.4f}{s['q1']:>9.4f}{s['q3']:>9.4f}{s['max']:>9.4f}"
          f"{s['n_gt']:>8}")


HDR = (f"{'':<26}{'n':>5}{'median':>9}{'mean':>9}{'min':>9}{'Q1':>9}"
       f"{'Q3':>9}{'max':>9}{'>0.3':>8}")


def auc(scores, labels):
    """Mann-Whitney AUC. scores: higher = predicted positive. labels: bool."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return None
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[y].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


# ---------------------------------------------------------------- Step 1
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

rows, missing = [], []
for rx in sorted(grp, key=lambda r: -nf[r]):
    p = f'{H}/stab_pipeline/{rx}/result.json'
    if not os.path.exists(p):
        missing.append((rx, '-', 'kein stab_pipeline-Ergebnis'))
        continue
    d = json.load(open(p))
    ref_stable = ref_lmin = None
    for g in d['geometries']:
        if g['source'] == 'RKS-ref':
            ref_stable = g.get('ext_stable')
            ref_lmin = g.get('lmin_ext')
    ref_xyz = f'{H}/orca_neb_results/{rx}/transition_state.xyz'
    for m, dirname in MODELS.items():
        mp = f'{H}/{dirname}/{rx}/transition_state.xyz'
        r, err = rmsd_files(ref_xyz, mp)
        if err:
            missing.append((rx, m, err))
        rows.append({'rxn': rx, 'group': grp[rx], 'nfod': nf[rx],
                     'ext_stable': ref_stable,
                     'lmin_ext': ref_lmin if isinstance(ref_lmin, float) else None,
                     'model': m, 'rmsd': r})

have = [r for r in rows if r['rmsd'] is not None]
print('=' * 78)
print('SCHRITT 1 — Zusammenstellung')
print('=' * 78)
print(f'Zeilen gesamt        : {len(rows)}  '
      f'({len(set(r["rxn"] for r in rows))} Reaktionen x {len(MODELS)} Modelle)')
print(f'Zeilen mit RMSD      : {len(have)}')
print(f'Zeilen ohne RMSD     : {len(rows) - len(have)}')
mods_ok = {m: sum(1 for r in have if r['model'] == m) for m in MODELS}
print(f'je Modell vorhanden  : {mods_ok}')
if missing:
    print(f'\nfehlend ({len(missing)}):')
    seen = {}
    for rx, m, why in missing:
        seen.setdefault((m, why), []).append(rx)
    for (m, why), rl in sorted(seen.items()):
        print(f'  {m:<11} {why:<40} {len(rl)} Rkt.: {" ".join(rl[:8])}'
              + ('...' if len(rl) > 8 else ''))

# models that actually have data
ACTIVE = [m for m in MODELS if mods_ok[m] > 0]

# ---------------------------------------------------------------- Step 2
print()
print('=' * 78)
print('SCHRITT 2 — Aufteilung nach externer Stabilitaet der Referenz')
print('=' * 78)
print(HDR)
for lab, want in (('extern STABIL', True), ('extern INSTABIL', False)):
    prow(lab + ' (gepoolt)',
         stats([r['rmsd'] for r in have if r['ext_stable'] is want]))
print()
for m in ACTIVE:
    print(f'-- {m} --')
    print(HDR)
    for lab, want in (('  stabil', True), ('  instabil', False)):
        prow(lab, stats([r['rmsd'] for r in have
                         if r['model'] == m and r['ext_stable'] is want]))

# ---------------------------------------------------------------- Step 3
print()
print('=' * 78)
print('SCHRITT 3a — Aufteilung nach FOD-Gruppe')
print('=' * 78)
print(HDR)
for g in ('high', 'mid', 'low'):
    prow(f'{g} (gepoolt)', stats([r['rmsd'] for r in have if r['group'] == g]))
print()
for m in ACTIVE:
    print(f'-- {m} --')
    print(HDR)
    for g in ('high', 'mid', 'low'):
        prow(f'  {g}', stats([r['rmsd'] for r in have
                              if r['model'] == m and r['group'] == g]))

print()
print('=' * 78)
print('SCHRITT 3b — N_FOD-Schwellensweep (gepoolt ueber alle Modelle)')
print('=' * 78)
print(f"{'Schwelle':>9}{'n oben':>8}{'med oben':>10}{'n unten':>9}"
      f"{'med unten':>11}{'Verhaeltnis':>13}")
sweep = []
for t in np.arange(0.10, 1.1001, 0.05):
    hi = [r['rmsd'] for r in have if r['nfod'] > t]
    lo = [r['rmsd'] for r in have if r['nfod'] <= t]
    if not hi or not lo:
        continue
    mh, ml = float(np.median(hi)), float(np.median(lo))
    ratio = mh / ml if ml > 0 else float('inf')
    sweep.append((t, len(hi), mh, len(lo), ml, ratio))
    print(f'{t:>9.2f}{len(hi):>8}{mh:>10.4f}{len(lo):>9}{ml:>11.4f}'
          f'{ratio:>13.2f}')
best = max(sweep, key=lambda x: x[5]) if sweep else None
if best:
    print(f'\nbeste Trennung bei N_FOD = {best[0]:.2f}  '
          f'(Verhaeltnis {best[5]:.2f})')

# ---------------------------------------------------------------- Step 4
print()
print('=' * 78)
print('SCHRITT 4 — Trennguete')
print('=' * 78)
st = [r['rmsd'] for r in have if r['ext_stable'] is True]
un = [r['rmsd'] for r in have if r['ext_stable'] is False]
m_st, m_un = float(np.median(st)), float(np.median(un))
print(f'Praediktor A: ext_stable (binaer)')
print(f'  Median stabil   {m_st:.4f}   Median instabil {m_un:.4f}   '
      f'Verhaeltnis {m_un/m_st:.2f}')
print(f'  Ueberlappung: {sum(1 for v in st if v > m_un)} von {len(st)} '
      f'"gut"-Zeilen ueber dem Median der "schlecht"-Gruppe')
print(f'               {sum(1 for v in un if v < m_st)} von {len(un)} '
      f'"schlecht"-Zeilen unter dem Median der "gut"-Gruppe')

if best:
    t = best[0]
    hi = [r['rmsd'] for r in have if r['nfod'] > t]
    lo = [r['rmsd'] for r in have if r['nfod'] <= t]
    mh, ml = float(np.median(hi)), float(np.median(lo))
    print(f'\nPraediktor B: N_FOD > {t:.2f}')
    print(f'  Median unten    {ml:.4f}   Median oben     {mh:.4f}   '
          f'Verhaeltnis {mh/ml:.2f}')
    print(f'  Ueberlappung: {sum(1 for v in lo if v > mh)} von {len(lo)} '
          f'"gut"-Zeilen ueber dem Median der "schlecht"-Gruppe')
    print(f'               {sum(1 for v in hi if v < ml)} von {len(hi)} '
          f'"schlecht"-Zeilen unter dem Median der "gut"-Gruppe')

y = [r['rmsd'] > THR for r in have]
print(f'\nAUC fuer die Vorhersage RMSD > {THR} A   '
      f'(Positive: {sum(y)} von {len(y)} Zeilen)')
a_ext = auc([0.0 if r['ext_stable'] else 1.0 for r in have], y)
a_fod = auc([r['nfod'] for r in have], y)
lam_rows = [r for r in have if r['lmin_ext'] is not None]
a_lam = auc([-r['lmin_ext'] for r in lam_rows],
            [r['rmsd'] > THR for r in lam_rows])
print(f'  ext_stable (binaer)      {a_ext:.4f}')
print(f'  N_FOD (kontinuierlich)   {a_fod:.4f}')
print(f'  -lambda_min_ext (kont.)  {a_lam:.4f}   (n={len(lam_rows)})')
print()
print(f"{'':<12}{'AUC ext':>10}{'AUC N_FOD':>11}{'AUC -lmin':>11}")
for m in ACTIVE:
    sub = [r for r in have if r['model'] == m]
    ys = [r['rmsd'] > THR for r in sub]
    if len(set(ys)) < 2:
        print(f'{m:<12}   (alle Zeilen gleiche Klasse)')
        continue
    ae = auc([0.0 if r['ext_stable'] else 1.0 for r in sub], ys)
    af = auc([r['nfod'] for r in sub], ys)
    sl = [r for r in sub if r['lmin_ext'] is not None]
    al = auc([-r['lmin_ext'] for r in sl], [r['rmsd'] > THR for r in sl])
    print(f'{m:<12}{ae:>10.4f}{af:>11.4f}'
          + (f'{al:>11.4f}' if al is not None else f'{"--":>11}'))

# ---------------------------------------------------------------- Step 5
print()
print('=' * 78)
print('SCHRITT 5 — nur die High-MR-Gruppe')
print('=' * 78)
hi_rows = [r for r in have if r['group'] == 'high']
print(HDR)
for lab, want in (('stabil (gepoolt)', True), ('instabil (gepoolt)', False)):
    prow(lab, stats([r['rmsd'] for r in hi_rows if r['ext_stable'] is want]))
print()
for m in ACTIVE:
    print(f'-- {m} --')
    print(HDR)
    for lab, want in (('  stabil', True), ('  instabil', False)):
        prow(lab, stats([r['rmsd'] for r in hi_rows
                         if r['model'] == m and r['ext_stable'] is want]))

print('\nExtern STABILE Reaktionen innerhalb der High-Gruppe:')
hs = sorted({r['rxn'] for r in hi_rows if r['ext_stable'] is True},
            key=lambda x: -nf[x])
head = f"{'rxn':<10}{'N_FOD':>8}" + ''.join(f'{m:>12}' for m in ACTIVE)
print(head)
for rx in hs:
    line = f'{rx:<10}{nf[rx]:>8.4f}'
    for m in ACTIVE:
        v = next((r['rmsd'] for r in hi_rows
                  if r['rxn'] == rx and r['model'] == m), None)
        line += f'{v:>12.4f}' if v is not None else f'{"--":>12}'
    print(line)

# ---------------------------------------------------------------- Step 6
print()
print('=' * 78)
print('SCHRITT 6 — Kontrolle gegen die korrigierte (BS) Referenz')
print('=' * 78)
conf = []
for p in sorted(glob.glob(f'{H}/bs_freq/*/result.json')):
    d = json.load(open(p))
    if d.get('n_imag') == 1:
        conf.append(d['rxn'])
print(f'bestaetigte BS-TS (genau 1 imaginaere Frequenz): {len(conf)}  '
      f'-> {" ".join(conf)}')


def bs_ts(rxn):
    c = glob.glob(f'{H}/bs_tsopt_batch/{rxn}/*.xyz')
    for pat in ('ts', 'final', 'opt'):
        for f in c:
            if pat in os.path.basename(f).lower():
                return f
    return c[0] if c else None


print()
head = (f"{'rxn':<10}{'Modell':<12}{'vs RKS-ref':>12}{'vs BS-TS':>11}"
        f"{'kleiner':>10}")
print(head)
pair = []
for rx in conf:
    ts = bs_ts(rx)
    for m in ACTIVE:
        mp = f'{H}/{MODELS[m]}/{rx}/transition_state.xyz'
        r_ref = next((r['rmsd'] for r in have
                      if r['rxn'] == rx and r['model'] == m), None)
        r_bs, err = rmsd_files(ts, mp) if ts else (None, 'kein BS-TS')
        if r_ref is None or r_bs is None:
            continue
        pair.append((rx, m, r_ref, r_bs))
        print(f'{rx:<10}{m:<12}{r_ref:>12.4f}{r_bs:>11.4f}'
              f'{("BS-TS" if r_bs < r_ref else "RKS-ref"):>10}')

if pair:
    print()
    print(HDR)
    prow('vs RKS-Referenz', stats([p[2] for p in pair]))
    prow('vs BS-TS', stats([p[3] for p in pair]))
    nb = sum(1 for p in pair if p[3] < p[2])
    print(f'\nBS-TS naeher in {nb} von {len(pair)} Zeilen')
    print()
    for m in ACTIVE:
        s = [p for p in pair if p[1] == m]
        if not s:
            continue
        k = sum(1 for p in s if p[3] < p[2])
        print(f'  {m:<12} BS-TS naeher in {k}/{len(s)}   '
              f'median vs RKS {np.median([p[2] for p in s]):.4f}  '
              f'vs BS {np.median([p[3] for p in s]):.4f}')

# ---------------------------------------------------------------- appendix
print()
print('=' * 78)
print('ANHANG — Zeilentabelle aus Schritt 1')
print('=' * 78)
print(f"{'rxn':<10}{'grp':<6}{'N_FOD':>8}{'ext':>10}{'lmin_ext':>11}"
      f"{'Modell':<12}{'RMSD':>9}")
for r in rows:
    e = {True: 'stabil', False: 'instabil', None: '?'}[r['ext_stable']]
    lm = f"{r['lmin_ext']:.5f}" if r['lmin_ext'] is not None else '-'
    rm = f"{r['rmsd']:.4f}" if r['rmsd'] is not None else 'FEHLT'
    print(f"{r['rxn']:<10}{r['group']:<6}{r['nfod']:>8.4f}{e:>10}{lm:>11}  "
          f"{r['model']:<12}{rm:>9}")
