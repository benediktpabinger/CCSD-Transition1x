"""The ORCA 6.0.0 acceptance test ("gate"): the private ORCA 6.0.0 install on el9
against (a) our ORCA 5.0.4 runs and (b) the OMol25 labels, at the 45 stored
Transition1x transition states, same protocol in all three.

Inputs (cluster)
    ~/orca_hinge_t1x/<rxn>/    ORCA 5.0.4  (pipeline/job_hinge_t1x.sh, job 10773547)
    ~/orca6_hinge_t1x/<rxn>/   ORCA 6.0.0  (~/orca6_hinge_t1x/job_hinge_t1x_orca6.sh)
        rks_sp.out  RKS + EnGrad;  uks_sp.out  UKS + STABPerform;  uks_engrad.out  EnGrad on those orbitals
    results/omol25_label_compare.csv   OMol25 label (ORCA 6.0.0, Levine et al. 2025) at the same geometry:
        s2_omol, e_omol_ev (total energy), fmax_omol (largest force component, eV/A)

Output  results/orca6_gate.csv, one row per reaction, and a verdict on stdout.
Pass criteria (the margins ORCA 5.0.4 already met against the OMol25 labels in the thesis):
    |E_BS(6) - E_BS(5)|      <= 0.1 meV       |E_BS(6) - E_omol|  <= 0.1 meV
    |F_BS(6) - F_BS(5)|      <= 0.02 eV/A     |F_BS(6) - F_omol|  <= 0.02 eV/A
    same group (closed-shell / broken-symmetry at <S^2> = 0.05) in all three
Exit code 1 if any reaction fails or is missing, so a job can gate on it.

    python pipeline/orca6_gate.py            (on the cluster; needs numpy)
"""
import csv
import glob
import os
import re
import sys
import numpy as np

H = os.path.expanduser('~')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVA = 51.42208
HA_EV = 27.211386245988
S2_BREAK = 0.05
TOL_E_MEV, TOL_F = 0.1, 0.02
E_RE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')
S2_RE = re.compile(r'Expectation value of <S\*\*2>\s*:\s*([-\d.]+)')


def sp(p):
    if not os.path.exists(p):
        return None
    t = open(p, errors='replace').read()
    if 'ORCA TERMINATED NORMALLY' not in t:
        return None
    e, s2 = E_RE.findall(t), S2_RE.findall(t)
    return dict(e=float(e[-1]) if e else None, s2=float(s2[-1]) if s2 else None)


def gradmax(p):
    t = open(p, errors='replace').read()
    i = t.find('CARTESIAN GRADIENT')
    if i < 0:
        return None
    G = []
    for line in t[i:].split('\n')[3:]:
        f = line.split()
        if len(f) < 6:
            break
        try:
            G.append([float(v) for v in f[3:6]])
        except ValueError:
            break
    return float(np.abs(np.array(G) * EVA).max()) if G else None


def run(d):
    uks, ueg = sp(f'{d}/uks_sp.out'), sp(f'{d}/uks_engrad.out')
    if uks is None or ueg is None:
        return None
    return dict(s2=uks['s2'], e_bs_ev=ueg['e'] * HA_EV, f_bs=gradmax(f'{d}/uks_engrad.out'))


omol = {r['rxn']: r for r in csv.DictReader(open(os.path.join(REPO, 'results', 'omol25_label_compare.csv')))}
rows, missing, fails = [], [], []
for d5 in sorted(glob.glob(f'{H}/orca_hinge_t1x/rxn*')):
    rx = os.path.basename(d5)
    r5, r6 = run(d5), run(f'{H}/orca6_hinge_t1x/{rx}')
    if r5 is None or r6 is None or rx not in omol:
        missing.append(rx)
        continue
    o = omol[rx]
    g = lambda s2: 'bs' if abs(s2) > S2_BREAK else 'closed'
    row = dict(rxn=rx, group5=g(r5['s2']), group6=g(r6['s2']), group_omol=g(float(o['s2_omol'])),
               s2_5=r5['s2'], s2_6=r6['s2'], s2_omol=float(o['s2_omol']),
               dE65_mev=(r6['e_bs_ev'] - r5['e_bs_ev']) * 1e3,
               dE6omol_mev=(r6['e_bs_ev'] - float(o['e_omol_ev'])) * 1e3,
               dF65=r6['f_bs'] - r5['f_bs'], dF6omol=r6['f_bs'] - float(o['fmax_omol']),
               f_bs_6=r6['f_bs'])
    row['pass'] = int(row['group5'] == row['group6'] == row['group_omol']
                      and abs(row['dE65_mev']) <= TOL_E_MEV and abs(row['dE6omol_mev']) <= TOL_E_MEV
                      and abs(row['dF65']) <= TOL_F and abs(row['dF6omol']) <= TOL_F)
    rows.append(row)
    if not row['pass']:
        fails.append(rx)

COLS = ['rxn', 'group5', 'group6', 'group_omol', 's2_5', 's2_6', 's2_omol',
        'dE65_mev', 'dE6omol_mev', 'dF65', 'dF6omol', 'f_bs_6', 'pass']
with open(os.path.join(REPO, 'results', 'orca6_gate.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, COLS)
    w.writeheader()
    for r in rows:
        w.writerow({c: (('%.6f' % r[c]) if isinstance(r[c], float) else r[c]) for c in COLS})

print('ORCA 6.0.0 GATE: %d reactions compared, %d missing, %d fail' % (len(rows), len(missing), len(fails)))
for k, lab in (('dE65_mev', 'E_BS  6.0.0 vs 5.0.4  [meV]'), ('dE6omol_mev', 'E_BS  6.0.0 vs OMol25 [meV]'),
               ('dF65', 'F_BS  6.0.0 vs 5.0.4  [eV/A]'), ('dF6omol', 'F_BS  6.0.0 vs OMol25 [eV/A]')):
    a = np.abs([r[k] for r in rows])
    if len(a):
        print('  %-30s median %.4f  max %.4f' % (lab, np.median(a), a.max()))
for grp in ('closed', 'bs'):
    print('  group %-7s 5.0.4 %2d  6.0.0 %2d  OMol25 %2d' % (grp, *[sum(r[c] == grp for r in rows)
                                                            for c in ('group5', 'group6', 'group_omol')]))
if missing:
    print('  missing:', ' '.join(missing))
if fails:
    print('  FAIL:', ' '.join(fails))
    for rx in fails:
        r = next(r for r in rows if r['rxn'] == rx)
        print('    %s  groups %s/%s/%s  dE65 %.3f  dE6omol %.3f meV  dF65 %.4f  dF6omol %.4f eV/A'
              % (rx, r['group5'], r['group6'], r['group_omol'], r['dE65_mev'], r['dE6omol_mev'], r['dF65'], r['dF6omol']))
sys.exit(1 if (fails or missing) else 0)
