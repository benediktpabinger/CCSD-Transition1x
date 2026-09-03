# -*- coding: utf-8 -*-
"""NEB-driven barriers + TS RMSD: bare MACE vs old head vs fixed head, by MR tier."""
import json, os
import numpy as np

D = r"C:\Users\PABING~1\AppData\Local\Temp\claude\c--Transition-1X-Transition-1x-Transition1x\60c2b781-b6da-49f8-932a-8cdb5275db7b\scratchpad\results"
neb = json.load(open(os.path.join(D, 'neb_barriers_collected.json')))
bm  = {r['rxn']: r for r in json.load(open(os.path.join(D, 'full_benchmark_results.json')))['reactions']}

TOP10    = {'rxn7949','rxn8832','rxn1320','rxn4113','rxn8885','rxn7945','rxn7937','rxn6196','rxn0346','rxn1150'}
BOTTOM10 = {'rxn9246','rxn4498','rxn1061','rxn4003','rxn4004','rxn4063','rxn4114','rxn4060','rxn1961','rxn1962'}
tier = lambda r: 'high' if r in TOP10 else 'low' if r in BOTTOM10 else 'mid'

METHODS = [('bare_mace', 'MACE'), ('old_fw2', 'MACE+D old'), ('fixed', 'MACE+D fixed')]

# reactions where the fixed run has finished (converged or not) -> common comparison set
done_fixed = [r for r, v in neb['fixed'].items() if 'fwd_meV' in v and v.get('converged')]
print(f"fixed-head NEB finished for {len(done_fixed)}/30 reactions; "
      f"converged: {sum(neb['fixed'][r]['converged'] for r in done_fixed)}")
for tag, lab in METHODS:
    conv = sum(1 for r in done_fixed if neb[tag].get(r, {}).get('converged'))
    print(f"  {lab:14s} converged {conv}/{len(done_fixed)} on this set")


def show(sel, title):
    print(f"\n=== {title} (n={len(sel)}) — NEB-driven forward barrier vs wB97M-V NEB reference ===")
    print(f"{'method':14s} {'MAE meV':>8} {'bias meV':>9} {'|err|>200':>10} {'TS-RMSD mean':>13} {'TS-RMSD med':>12} {'n_rmsd':>7}")
    for tag, lab in METHODS:
        errs, rmsds = [], []
        for r in sel:
            v = neb[tag].get(r, {})
            ref = bm[r].get('neb_wb97m_fwd_meV')
            if 'fwd_meV' in v and ref is not None:
                errs.append(v['fwd_meV'] - ref)
            x = v.get('ts_rmsd_A')
            if isinstance(x, (int, float)):
                rmsds.append(x)
        errs = np.array(errs); rmsds = np.array(rmsds)
        if len(errs) == 0:
            continue
        print(f"{lab:14s} {np.abs(errs).mean():8.1f} {errs.mean():+9.1f} {(np.abs(errs) > 200).sum():10d} "
              f"{(rmsds.mean() if len(rmsds) else float('nan')):13.3f} "
              f"{(np.median(rmsds) if len(rmsds) else float('nan')):12.3f} {len(rmsds):7d}")


show(done_fixed, 'ALL finished')
for t in ('high', 'mid', 'low'):
    sel = [r for r in done_fixed if tier(r) == t]
    if sel:
        show(sel, f'tier {t}')

# vs CCSD(T) where available
sel = [r for r in done_fixed if bm[r].get('ccsdt_fwd_meV') is not None]
if sel:
    print(f"\n=== vs CCSD(T) (n={len(sel)}) ===")
    for tag, lab in METHODS:
        errs = np.array([neb[tag][r]['fwd_meV'] - bm[r]['ccsdt_fwd_meV'] for r in sel if 'fwd_meV' in neb[tag].get(r, {})])
        print(f"{lab:14s} MAE {np.abs(errs).mean():7.1f}  bias {errs.mean():+7.1f}")

# per-reaction table
print(f"\n{'rxn':9s} {'tier':5s} {'ref':>7} {'MACE':>7} {'old':>7} {'fixed':>7} | {'rmsd MACE':>9} {'rmsd old':>8} {'rmsd fix':>8}")
for r in done_fixed:
    ref = bm[r].get('neb_wb97m_fwd_meV')
    vals = [neb[t].get(r, {}).get('fwd_meV') for t, _ in METHODS]
    rm   = [neb[t].get(r, {}).get('ts_rmsd_A') for t, _ in METHODS]
    f = lambda x: f"{x:7.0f}" if isinstance(x, (int, float)) else f"{'-':>7}"
    g = lambda x: f"{x:8.3f}" if isinstance(x, (int, float)) else f"{str(x)[:8]:>8}"
    print(f"{r:9s} {tier(r):5s} {f(ref)} {f(vals[0])} {f(vals[1])} {f(vals[2])} | {g(rm[0]):>9} {g(rm[1])} {g(rm[2])}")
