"""Task list for putting our own transition states through ORCA.

Sixteen of our broken-symmetry saddles carry a PySCF numerical Hessian, and
every stage-2 and stage-3 verdict in the working document rests on them. Two
have been checked against ORCA -- rxn1147 and rxn7957, where the imaginary modes
agreed to a 0.9994 overlap. The other fourteen have never been checked against
anything.

The second reason is practical: ORCA's IRC needs an ORCA .hess. Producing them
now means an IRC anywhere in the set can start immediately instead of spending
half an hour on a Hessian first.

Writes one line per task: <label> <geometry path>
"""
import glob
import json
import os

H = '/home/energy/s242862'
OUT = f'{H}/freq_tasks_ours.txt'
FREQDIRS = ('bs_freq_fromneb', 'bs_freq_v2', 'bs_freq')
OPTDIRS = ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch')


def our_geometry(rx):
    """The structure the Hessian was actually built on, same order as elsewhere."""
    for d in OPTDIRS:
        rp = f'{H}/{d}/{rx}/result.json'
        if not os.path.exists(rp):
            continue
        j = json.load(open(rp))
        if j.get('e_uks_final') is None:
            continue
        for f in sorted(glob.glob(f'{H}/{d}/{rx}/*.xyz')):
            if any(p in os.path.basename(f).lower()
                   for p in ('ts', 'final', 'opt')):
                return f, d.replace('bs_tsopt_', '')
    return None, None


rxns = set()
for fd in FREQDIRS:
    for p in glob.glob(f'{H}/{fd}/rxn*/hessian.npy'):
        rxns.add(os.path.basename(os.path.dirname(p)))

tasks = []
for rx in sorted(rxns):
    g, origin = our_geometry(rx)
    if not g:
        print(f'{rx}: no geometry found, skipped')
        continue
    lbl = f'ours_{rx}'
    if os.path.exists(f'{H}/orca_freq/{lbl}/numfreq.hess'):
        continue
    # already done under a different label by the IRC preparation
    if os.path.exists(f'{H}/orca_irc/{rx}_ours/numfreq.hess'):
        print(f'{rx}: already cross-checked in orca_irc, skipped')
        continue
    tasks.append((lbl, g, origin))

with open(OUT, 'w') as fh:
    for lbl, g, origin in tasks:
        fh.write(f'{lbl} {g}\n')

print(f'\n{len(tasks)} tasks -> {OUT}')
print(f'array range: 0-{len(tasks) - 1}')
for lbl, g, origin in tasks:
    print(f'  {lbl:<16}from {origin}')
