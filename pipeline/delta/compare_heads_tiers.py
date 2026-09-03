# -*- coding: utf-8 -*-
"""Old vs fixed head on the 30-reaction SP benchmark, with the true MR tiers."""
import json, os
import numpy as np

D = r"C:\Users\PABING~1\AppData\Local\Temp\claude\c--Transition-1X-Transition-1x-Transition1x\60c2b781-b6da-49f8-932a-8cdb5275db7b\scratchpad\results"
old = {r['rxn']: r for r in json.load(open(os.path.join(D, 'eval_benchmark_sp_fw2_full.json')))['reactions']}
new = {r['rxn']: r for r in json.load(open(os.path.join(D, 'eval_benchmark_sp_fixed_full.json')))['reactions']}
rxns = list(old)
TOP10    = {'rxn7949','rxn8832','rxn1320','rxn4113','rxn8885','rxn7945','rxn7937','rxn6196','rxn0346','rxn1150'}
BOTTOM10 = {'rxn9246','rxn4498','rxn1061','rxn4003','rxn4004','rxn4063','rxn4114','rxn4060','rxn1961','rxn1962'}
TIER = {r: ('high' if r in TOP10 else 'low' if r in BOTTOM10 else 'mid') for r in rxns}


def barrier(e):
    e = np.array(e); return float((e - e[0]).max() * 1000)


def stats(sel):
    ref = {r: barrier(old[r]['e_wb97m_eV']) for r in sel}
    out = {}
    for label, src, ekey, fkey, bkey in [
        ('wB97X-D3 (DFT)', old, 'emae_wb97x_meV', 'fmae_wb97x_meVA', 'e_wb97x_eV'),
        ('MACE',           old, 'emae_mace_meV',  'fmae_mace_meVA',  'e_mace_eV'),
        ('MACE+D old',     old, 'emae_delta_meV', 'fmae_delta_meVA', 'e_delta_eV'),
        ('MACE+D fixed',   new, 'emae_delta_meV', 'fmae_delta_meVA', 'e_delta_eV'),
    ]:
        emae = np.mean([src[r][ekey] for r in sel]); fmae = np.mean([src[r][fkey] for r in sel])
        berr = np.array([barrier(src[r][bkey]) - ref[r] for r in sel])
        out[label] = (emae, fmae, np.abs(berr).mean(), berr.mean())
    return out


for title, sel in [('ALL 30', rxns)] + [(f'tier {t}', [r for r in rxns if TIER[r] == t]) for t in ('high', 'mid', 'low')]:
    print(f"\n=== {title} (n={len(sel)}) ===")
    print(f"{'method':16s} {'eMAE meV':>10} {'fMAE meV/A':>11} {'barrier MAE':>12} {'barrier bias':>13}")
    for k, (e, f, bm, bb) in stats(sel).items():
        print(f"{k:16s} {e:10.1f} {f:11.1f} {bm:12.1f} {bb:+13.1f}")
