import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open(r'c:\Transition 1X\Transition 1x\Transition1x\full_benchmark_results.json') as f:
    bm = json.load(f)
bm_by_rxn = {r['rxn']: r for r in bm['reactions']}

TOP10    = {'rxn7949','rxn8832','rxn1320','rxn4113','rxn8885','rxn7945','rxn7937','rxn6196','rxn0346','rxn1150'}
BOTTOM10 = {'rxn9246','rxn4498','rxn1061','rxn4003','rxn4004','rxn4063','rxn4114','rxn4060','rxn1961','rxn1962'}
MR = {r: ('High' if r in TOP10 else ('Low' if r in BOTTOM10 else 'Mid')) for r in bm_by_rxn}

METHODS = {
    'UMA-S':       'uma_neb_fwd_meV',
    'UMA-M':       'uma_m_neb_fwd_meV',
    'eSEN':        'esen_neb_fwd_meV',
    'MACE+δ fw2':  'delta_fw2_neb_fwd_meV',
}
METHOD_COLOR = {'UMA-S': 'tab:blue', 'UMA-M': 'tab:purple', 'eSEN': 'tab:brown', 'MACE+δ fw2': 'tab:red'}
CATS = ['High', 'Mid', 'Low']
CAT_BINS = {
    'High': np.linspace(-1000, 1000, 17),
    'Mid':  np.linspace(-250, 250, 17),
    'Low':  np.linspace(-150, 150, 17),
}

errors = {cat: {m: [] for m in METHODS} for cat in CATS}
for rxn, r in bm_by_rxn.items():
    cat = MR.get(rxn)
    if cat is None:
        continue
    ref = r.get('neb_wb97m_fwd_meV')
    if ref is None:
        continue
    for m, key in METHODS.items():
        pred = r.get(key)
        if pred is not None:
            errors[cat][m].append(pred - ref)

fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=False)
for i, cat in enumerate(CATS):
    bins = CAT_BINS[cat]
    for j, m in enumerate(METHODS):
        ax = axes[i, j]
        vals = np.array(errors[cat][m])
        ax.hist(vals, bins=bins, color=METHOD_COLOR[m], edgecolor='black', alpha=0.85)
        ax.axvline(0, color='k', linestyle='--', linewidth=1)
        mae = np.mean(np.abs(vals))
        bias = np.mean(vals)
        ax.set_title(f'{m}\nMAE={mae:.0f}  bias={bias:+.0f} meV', fontsize=10)
        if i == 2:
            ax.set_xlabel('Predicted - wB97M-V (meV)', fontsize=9)
        if j == 0:
            ax.set_ylabel(f'{cat} MR\nCount', fontsize=10.5, weight='bold')
        ax.tick_params(labelsize=8)

fig.suptitle('Forward barrier error vs ORCA ωB97M-V NEB reference, split by MR category', fontsize=14, weight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = r'c:\Transition 1X\Transition 1x\Transition1x\benchmark_plots\barrier_error_histogram_by_mr.png'
fig.savefig(out, dpi=150)
print('Saved', out)
