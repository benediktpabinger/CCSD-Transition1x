"""The two structures of ours that exist but were never given a Hessian.

make_freq_list_ours.py built its list from the reactions that already had a
PySCF Hessian, so a reaction that never had one could not acquire one -- the
list was keyed on the wrong thing. rxn1283 and rxn4522 fell through that gap
and show up as the only two "?" cells in the figure: a structure exists and
nobody tested it.

rxn4522 comes with a caveat worth carrying: its optimisation ran into the
walltime after 332 steps, so the geometry is the last step rather than a
converged one. The gradient from stage 1b says whether that matters.
"""
import glob
import json
import os

H = '/home/energy/s242862'
OUT = f'{H}/freq_tasks_gaps.txt'
RX = ('rxn1283', 'rxn4522')


def our_geometry(rx):
    """Same resolution order as everywhere else, so this is the same structure
    the tables and the figure call ours."""
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        rp = f'{H}/{d}/{rx}/result.json'
        if not os.path.exists(rp):
            continue
        j = json.load(open(rp))
        if j.get('e_uks_final') is None:
            continue
        for f in sorted(glob.glob(f'{H}/{d}/{rx}/*.xyz')):
            if any(p in os.path.basename(f).lower()
                   for p in ('ts', 'final', 'opt')):
                return f, d.replace('bs_tsopt_', ''), j
    return None, None, None


tasks = []
for rx in RX:
    g, origin, j = our_geometry(rx)
    lbl = f'ours_{rx}'
    if not g:
        print(f'{rx}: no structure found'); continue
    if os.path.exists(f'{H}/orca_freq/{lbl}/numfreq.hess'):
        print(f'{rx}: already done'); continue
    tasks.append((lbl, g))
    print(f'{rx}  from {origin}  status={j.get("status")}  '
          f'steps={j.get("n_geom_steps")}  S2={j.get("s2_final")}')
    print(f'     {g}')

with open(OUT, 'w') as fh:
    for lbl, g in tasks:
        fh.write(f'{lbl} {g}\n')
print(f'\n{len(tasks)} tasks -> {OUT}')
if tasks:
    print(f'array range: 0-{len(tasks) - 1}')
