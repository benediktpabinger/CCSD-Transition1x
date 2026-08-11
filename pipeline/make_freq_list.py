"""Build the task list for the frequency sweep, from the data rather than by hand.

Every structure that exists and has no Hessian goes in. Deciding by hand which
ones "need" one is what produced the current state: 15 model frequencies exist,
they were chosen by two different criteria on two different nights, and the
reactions left out include the ones where our own structure is known to be wrong.

Including geometries that are not stationary is deliberate. The Hessian there
does not prove anything about transition states -- n_imag at a point with a
large gradient means nothing -- but it is exactly what ORCA's OptTS needs as a
starting Hessian, so nothing computed here is wasted if the point turns out to
need optimising afterwards.

Writes one line per task: <label> <geometry path>
"""
import glob
import json
import os

H = '/home/energy/s242862'
MODELS = {'UMA-S': 'uma_neb_results', 'UMA-M': 'uma_m_neb_results',
          'eSEN': 'esen_neb_results'}
OUT = f'{H}/freq_tasks.txt'

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


def cls_of(rx):
    p = f'{H}/stab_pipeline/{rx}/result.json'
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    ref = {g['source']: g for g in d['geometries']}.get('RKS-ref')
    if not ref or ref.get('ext_stable') is None:
        return None
    return 'MR' if ref['ext_stable'] is False else 'simple'


def grad_of(rx, m):
    p = f'{H}/stab_pipeline/{rx}/result.json'
    if not os.path.exists(p):
        return None
    g = {x['source']: x for x in json.load(open(p))['geometries']}.get(m)
    if not g or g.get('ext_stable') is None:
        return None
    if g['ext_stable']:
        return (g.get('rks_grad') or {}).get('max_evang')
    return ((g.get('bs') or {}).get('bs_grad') or {}).get('max_evang')


def done(label):
    """A Hessian already exists for this structure, in either code."""
    return (os.path.exists(f'{H}/freq_at_model/{label}/hessian.npy')
            or os.path.exists(f'{H}/orca_freq/{label}/numfreq.hess'))


tasks = []

# 1. every model geometry of the multireference group
for rx in sorted(grp, key=lambda r: r):
    if cls_of(rx) != 'MR':
        continue
    for m, dn in MODELS.items():
        g = f'{H}/{dn}/{rx}/transition_state.xyz'
        lbl = f'{rx}_{m}'
        if os.path.exists(g) and not done(lbl):
            tasks.append((lbl, g, 'MR-model', grad_of(rx, m)))

# 2. one model geometry per single-reference reaction, as the control group.
#    The three models agree to 0.0045 A median there, so they are one structure.
for rx in sorted(grp, key=lambda r: r):
    if cls_of(rx) != 'simple':
        continue
    best = None
    for m in MODELS:
        gr = grad_of(rx, m)
        if gr is not None and (best is None or gr < best[1]):
            best = (m, gr)
    if not best:
        continue
    g = f'{H}/{MODELS[best[0]]}/{rx}/transition_state.xyz'
    lbl = f'{rx}_{best[0]}'
    if os.path.exists(g) and not done(lbl):
        tasks.append((lbl, g, 'simple-control', best[1]))

# 3. the structures a from-model optimisation produced and nobody classified
for d in sorted(glob.glob(f'{H}/tsopt_from_model/*/')):
    tag = os.path.basename(os.path.dirname(d))
    xs = [f for f in sorted(glob.glob(f'{d}/*.xyz'))
          if any(p in os.path.basename(f).lower() for p in ('ts', 'final', 'opt'))]
    if not xs:
        continue
    lbl = f'tsopt_{tag}'
    if not done(lbl):
        tasks.append((lbl, xs[0], 'new-from-tsopt', None))

# 4. the ORCA NEB-TS structures, which no stage has ever been applied to
for d in sorted(glob.glob(f'{H}/bs_uks_neb_results/rxn*/')):
    rx = os.path.basename(os.path.dirname(d))
    g = sorted(glob.glob(f'{d}/*NEB-TS_converged.xyz'))
    if not g:
        continue
    lbl = f'nebts_{rx}'
    if not done(lbl):
        tasks.append((lbl, g[0], 'neb-ts', None))

with open(OUT, 'w') as fh:
    for lbl, g, kind, gr in tasks:
        fh.write(f'{lbl} {g}\n')

print(f'{len(tasks)} tasks -> {OUT}')
print(f'array range: 0-{len(tasks) - 1}')
print()
byk = {}
for lbl, g, kind, gr in tasks:
    byk.setdefault(kind, []).append((lbl, gr))
for k, v in byk.items():
    print(f'{k:<18}{len(v):>4}')
print()
for k, v in byk.items():
    print(f'--- {k}')
    for lbl, gr in v:
        print(f'    {lbl:<22}{"" if gr is None else f"grad {gr:.3f}"}')
