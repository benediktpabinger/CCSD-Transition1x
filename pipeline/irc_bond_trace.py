"""Read an ORCA IRC trajectory and print how the reactive bonds move along it.

The endpoint of an IRC is the usual thing people report, but it is not what
decides the two contested cases here. Those turn on whether a rival structure
lies *on the descending branch* of the path -- past the transition state --
rather than at a saddle of its own. So the output is the bond trace: both
reactive distances at every point of the path, plus the RMSD to the rival
structure, so it is visible whether the path runs through it.

If the minimum RMSD to the rival along the path is small, the rival is on this
path, downhill of this saddle, and is not a transition state of this reaction.
That is the statement the bond-length judgement was standing in for.

Usage: python irc_bond_trace.py <rxn> <ours|UMA-S|UMA-M|eSEN> [rival]
"""
import glob
import json
import os
import sys

import numpy as np

H = '/home/energy/s242862'
MODEL_DIR = {'UMA-S': 'uma_neb_results', 'UMA-M': 'uma_m_neb_results',
             'eSEN': 'esen_neb_results'}


def read_multi_xyz(p):
    """Every frame of a concatenated xyz trajectory."""
    toks = open(p).read().split('\n')
    frames, i = [], 0
    while i < len(toks):
        if not toks[i].strip():
            i += 1
            continue
        try:
            n = int(toks[i].split()[0])
        except (ValueError, IndexError):
            i += 1
            continue
        title = toks[i + 1] if i + 1 < len(toks) else ''
        sym, xyz = [], []
        for line in toks[i + 2:i + 2 + n]:
            f = line.split()
            if len(f) < 4:
                break
            sym.append(f[0]); xyz.append([float(x) for x in f[1:4]])
        if len(xyz) == n:
            frames.append((sym, np.array(xyz), title))
        i += 2 + n
    return frames


def read_xyz(p):
    return read_multi_xyz(p)[0][:2]


def kabsch(A, B):
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    V, S, W = np.linalg.svd(Ac.T @ Bc)
    D = np.diag([1., 1., np.sign(np.linalg.det(V @ W))])
    return float(np.sqrt((((Ac @ (V @ D @ W)) - Bc) ** 2).sum() / len(A)))


def reactive(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        p = f'{H}/{d}/{rx}/result.json'
        if os.path.exists(p):
            rb = json.load(open(p)).get('reactive_bonds')
            if rb:
                return [(e['pair'][0], e['pair'][1], e['sym']) for e in rb[:2]]
    return []


def geometry_of(rx, src):
    if src == 'ours':
        for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
            for f in sorted(glob.glob(f'{H}/{d}/{rx}/*.xyz')):
                if any(p in os.path.basename(f).lower()
                       for p in ('ts', 'final', 'opt')):
                    return f
        return None
    return f'{H}/{MODEL_DIR[src]}/{rx}/transition_state.xyz'


def main(rx, src, rival=None):
    W = f'{H}/orca_irc/{rx}_{src}'
    pairs = reactive(rx)
    if not pairs:
        print(f'{rx}: no reactive bonds recorded'); return 1

    refs = {}
    for lab in ('reactant', 'product'):
        p = f'{H}/orca_neb_results/{rx}/{lab}.xyz'
        if os.path.exists(p):
            refs[lab] = read_xyz(p)[1]
    if rival:
        g = geometry_of(rx, rival)
        if g and os.path.exists(g):
            refs[rival] = read_xyz(g)[1]

    cands = sorted(glob.glob(f'{W}/*IRC_Full_trj.xyz')
                   or glob.glob(f'{W}/*IRC*trj.xyz'))
    if not cands:
        print(f'{W}: no IRC trajectory'); return 1
    frames = read_multi_xyz(cands[0])
    print(f'=== {rx} [{src}]   {len(frames)} points from '
          f'{os.path.basename(cands[0])}')
    print(f'    reactive bonds: ' + ', '.join(nm for _, _, nm in pairs))
    hdr = '    idx  ' + '  '.join(f'{nm:>9}' for _, _, nm in pairs)
    hdr += '  ' + '  '.join(f'{k[:9]:>9}' for k in refs)
    print(hdr)
    best = {k: (9e9, -1) for k in refs}
    for i, (sym, xyz, _) in enumerate(frames):
        d = [np.linalg.norm(xyz[a] - xyz[b]) for a, b, _ in pairs]
        r = {k: kabsch(xyz, v) for k, v in refs.items()}
        for k, v in r.items():
            if v < best[k][0]:
                best[k] = (v, i)
        if i % max(1, len(frames) // 40) == 0 or i == len(frames) - 1:
            print(f'    {i:4d}  ' + '  '.join(f'{x:9.4f}' for x in d)
                  + '  ' + '  '.join(f'{r[k]:9.4f}' for k in refs))
    print()
    for k, (v, i) in best.items():
        print(f'    closest approach to {k:<10} {v:7.4f} A at point {i} '
              f'of {len(frames) - 1}')
    if rival and rival in best:
        v, i = best[rival]
        mid = len(frames) // 2
        side = 'forward' if i > mid else 'backward'
        print(f'\n    the {rival} structure sits {v:.4f} A from the path, on '
              f'the {side} branch')
        print('    a small value means it lies on this reaction path downhill '
              'of this saddle,\n    and is therefore not a transition state '
              'of this reaction')
    return 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
