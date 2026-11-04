"""Grades every run of the DFT refinement pilot (D32, D36) and lists every exclusion.

One row per run folder, ~/marks_pilot/<rxn>/ (DFT starting Hessian) and
~/marks_pilot/<rxn>/modelhess/ (model starting Hessian). The rule, in this order:

  solution switch  stability analysis at the saddle finds the OptTS solution unstable, the run
                   crosses between closed-shell and broken-symmetry (<S^2> 0.01) during OptTS,
                   or <S^2> at the saddle and in the stability run differ by more than 0.01.
                   Protocol artefact: excluded, rerun with the stability loop.
  dft_not_converged  OptTS without "THE OPTIMIZATION HAS CONVERGED" (250 steps)  counts, fails
  dft_minimum / dft_higher_order  finished frequency run with 0 or >= 2 imaginary modes  counts, fails
  route / one_end / same_end / elsewhere  both IRC ends relaxed with DFT (job_irc_ends.sh) and
                   compared with reactant and product by bond graph (pilot_check.bonds, H-H rule
                   of D35); route passes, the rest count and fail
  missing or crashed steps, unconverged end optimisations: technical, excluded with the reason

Energies only from gradient-type runs (OptTS and end Opt, SCNL), never from the stability
single points, which treat VV10 non-self-consistently (about 1 meV apart).

Output results/marks_pilot.csv, results/marks_pilot_excluded.csv

    python pipeline/marks_collect.py
    python pipeline/marks_collect.py --submit-ends   # also submit job_irc_ends.sh where due
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys

from ase.io import read
from ase.units import Hartree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_check import bonds  # noqa: E402

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PILOT = f'{H}/marks_pilot'
S2_CUT = 0.01
DONE = 'ORCA TERMINATED NORMALLY'


def text(p):
    return open(p, errors='replace').read() if os.path.exists(p) else ''


def s2(t):
    v = re.findall(r'Expectation value of <S\*\*2>\s*:\s*(-?[\d.]+)', t)
    return [float(x) for x in v]


def energy(p):
    assert os.path.basename(p) in ('optts.out', 'opt.out'), 'energies only from gradient-type runs'
    v = re.findall(r'FINAL SINGLE POINT ENERGY\s+(-?[\d.]+)', text(p))
    return float(v[-1]) if v else None


def grade(d):
    root = d[:-len('/modelhess')] if d.endswith('/modelhess') else d
    meta = json.load(open(f'{root}/model.json'))
    r = dict(run=os.path.relpath(d, PILOT), rxn=meta['rxn'], hess='model' if d != root else 'dft',
             kind=meta['kind'], group=meta['group'], nfod=meta['nfod'], cls_model=meta['cls_model'])
    t_opt, t_stab, t_freq, t_irc = (text(f'{d}/{n}.out') for n in ('optts', 'stab', 'freq', 'irc'))
    rounds = text(f'{d}/stab_rounds').strip()
    r.update(stab_rounds=int(rounds) if rounds else 0, s2_sp=(s2(text(f'{d}/uks_sp.out')) or [''])[-1])

    def out(outcome, counts, reason=''):
        r.update(outcome=outcome, counts=int(counts), passed=int(outcome == 'route'), reason=reason)
        return r

    if not t_opt:
        return out('pending', False, 'no optts.out')
    v = s2(t_opt)
    r.update(optts_cycles=t_opt.count('GEOMETRY OPTIMIZATION CYCLE'),
             optts_conv=int('THE OPTIMIZATION HAS CONVERGED' in t_opt), s2_ts=v[-1] if v else '')
    if not v or DONE not in t_opt or not os.path.exists(f'{d}/optts.xyz'):
        return out('technical', False, 'OptTS crashed: ' + ('; '.join(re.findall(r'\[file .*?\]: (.*)', t_opt)[:1]) or 'no SCF'))
    if DONE not in t_stab:
        return out('technical', False, 'no finished stability run at the saddle')
    vs = s2(t_stab)
    r.update(s2_stab=vs[-1], stab_unstable=int('wavefunction is unstable' in t_stab))
    if r['stab_unstable'] or (v[0] > S2_CUT) != (v[-1] > S2_CUT) or abs(vs[-1] - v[-1]) > S2_CUT:
        return out('solution_switch', False, f'<S^2> OptTS {v[0]:.3f} -> {v[-1]:.3f}, stability run {vs[-1]:.3f}'
                   + (', unstable' if r['stab_unstable'] else ''))
    if not r['optts_conv']:
        return out('dft_not_converged', True, f"{r['optts_cycles']} cycles")
    r['e_ts_eh'] = energy(f'{d}/optts.out')
    if DONE not in t_freq:
        return out('technical', False, 'no finished frequency run')
    r['n_imag'] = t_freq.count('***imaginary mode***')
    if r['n_imag'] != 1:
        return out('dft_minimum' if r['n_imag'] == 0 else 'dft_higher_order', True, f"{r['n_imag']} imaginary modes")
    if DONE not in t_irc:
        return out('technical', False, 'IRC did not finish')
    r['irc_conv_dirs'] = t_irc.count('THE IRC HAS CONVERGED')
    if not all(os.path.exists(f'{d}/end_{e}/opt.out') for e in 'FB'):
        return out('ends_pending', False, 'IRC ends not relaxed yet')
    Rx, Px = read(f'{root}/reactant.xyz'), read(f'{root}/product.xyz')
    bR, bP = bonds(Rx.numbers, Rx.positions), bonds(Px.numbers, Px.positions)
    ends = []
    for e in 'FB':
        t = text(f'{d}/end_{e}/opt.out')
        conv = 'THE OPTIMIZATION HAS CONVERGED' in t
        r[f'end_{e}_conv'] = int(conv)
        if not conv:
            return out('technical', False, f'end {e} optimisation not converged')
        a = read(f'{d}/end_{e}/opt.xyz')
        b = bonds(a.numbers, a.positions)
        ends.append('R' if b == bR else 'P' if b == bP else 'X')
        r[f'end_{e}'] = ends[-1]
        r[f'end_{e}_diff_r'] = ' '.join('%d-%d' % p for p in sorted(b ^ bR))
        r[f'end_{e}_diff_p'] = ' '.join('%d-%d' % p for p in sorted(b ^ bP))
        r[f'barrier_{e}_ev'] = (r['e_ts_eh'] - energy(f'{d}/end_{e}/opt.out')) * Hartree
    if set(ends) == {'R', 'P'}:
        return out('route', True)
    if ends[0] == ends[1] and ends[0] != 'X':
        return out('same_end', True)
    return out('one_end' if 'R' in ends or 'P' in ends else 'elsewhere', True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--submit-ends', action='store_true')
    a = ap.parse_args()
    dirs = sorted(glob.glob(f'{PILOT}/rxn*/model.json'))
    dirs = [os.path.dirname(p) for p in dirs]
    dirs += sorted(glob.glob(f'{PILOT}/rxn*/modelhess'))
    rows = [grade(d) for d in dirs]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    for name, R in (('marks_pilot', rows), ('marks_pilot_excluded', [r for r in rows if not r['counts']])):
        with open(f'{REPO}/results/{name}.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys, restval='')
            w.writeheader()
            for r in R:
                w.writerow({k: ('%.5f' % v if isinstance(v, float) else v) for k, v in r.items()})
    for r in rows:
        print('%-18s %-5s %-8s %-17s %s' % (r['run'], r['hess'], r['kind'], r['outcome'], r['reason']))
    print()
    for hess in ('dft', 'model'):
        for kind in ('fail', 'pass', 'baseline'):
            R = [r for r in rows if r['hess'] == hess and r['kind'] == kind]
            if R:
                c = [r for r in R if r['counts']]
                print('%-5s %-8s runs %2d  counted %2d  route %2d  excluded %2d' % (
                    hess, kind, len(R), len(c), sum(r['passed'] for r in R), len(R) - len(c)))
    if a.submit_ends:
        for d, r in zip(dirs, rows):
            if r['outcome'] == 'ends_pending' and not os.path.exists(f'{d}/ends.submitted'):
                subprocess.run(['sbatch', f'--export=ALL,DIR={d}', f'{REPO}/pipeline/job_irc_ends.sh'], check=True)
                open(f'{d}/ends.submitted', 'w').close()
                print('ends submitted', r['run'])
