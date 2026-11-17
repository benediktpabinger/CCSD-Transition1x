"""Version 2: axis says Transition1x transition state, title without "all" (10 009 of 10 073 have both labels).

Figure of the reactions: N_FOD over all 10 009 Transition1x transition states found in OMol25,
closed-shell vs broken-symmetry by the OMol25 label (<S^2> > 0.05 broken-symmetry), with the
share of broken-symmetry reactions per bin in a second panel and the key percentages printed.

Output figures/paper/fig_landscape.{png,pdf}; numbers printed for the caption.

    python pipeline/plot_landscape.py
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
OUT = os.path.join(REPO, 'figures', 'paper')
INK, INK2, GRID = '#1f1f1e', '#5f5e58', '#e4e3df'
CS, BS = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.size': 9.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
                     'font.family': 'DejaVu Sans'})

fod = {r['rxn']: float(r['nfod']) for r in csv.DictReader(open(f'{RES}/fod_ts_all.csv'))}
s2 = {r['rxn']: float(r['s2']) for r in csv.DictReader(open(f'{RES}/omol25_ts_all.csv')) if r['found'] == '1'}
rx = [k for k in s2 if k in fod]
n = np.array([fod[k] for k in rx])
b = np.array([s2[k] > 0.05 for k in rx])
N = len(n)
share = lambda sel: 100 * sel.sum() / N   # noqa: E731
print(f'{N} transition states; broken-symmetry {share(b):.1f} %; N_FOD >= 0.5 {share(n >= .5):.1f} %; '
      f'N_FOD >= 0.8 {share(n >= .8):.1f} %; N_FOD >= 1 {share(n >= 1):.1f} % ({(n >= 1).sum()})')
for lo in (0.5, 0.8, 1.0):
    s = n >= lo
    print(f'  broken-symmetry among N_FOD >= {lo}: {100 * b[s].mean():.0f} %')

edges = np.arange(0, 1.85, 0.05)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True, gridspec_kw=dict(height_ratios=(2.2, 1), hspace=0.12))
hc, _ = np.histogram(n[~b], edges)
hb, _ = np.histogram(n[b], edges)
w = edges[1] - edges[0]
a1.bar(edges[:-1], hc, width=w, align='edge', color=CS, edgecolor='white', lw=0.6, label='Closed-shell')
a1.bar(edges[:-1], hb, width=w, align='edge', bottom=hc, color=BS, edgecolor='white', lw=0.6, label='Broken-symmetry')
a1.set_ylabel('Transition states')
a1.grid(axis='y', color=GRID, lw=0.8)
a1.set_axisbelow(True)
a1.legend(frameon=False, loc='upper right')
a1.set_title(f'Multireference character of {N:,} Transition1x transition states'.replace(',', ' '), fontsize=10.5, loc='left')
for lo, y in ((0.5, 0.62), (1.0, 0.36)):
    a1.axvline(lo, color=INK2, lw=1, ls='--')
    a1.text(lo + 0.02, a1.get_ylim()[1] * y, f'{share(n >= lo):.0f} % of all\nhave $N_\\mathrm{{FOD}}\\geq {lo:g}$',
            fontsize=8.5, color=INK, va='top')

frac = np.where(hc + hb >= 10, 100 * hb / np.maximum(hc + hb, 1), np.nan)
mid = edges[:-1] + w / 2
a2.plot(mid, frac, color=BS, lw=2, marker='o', ms=3.5)
a2.set_ylim(0, 100)
a2.set_ylabel('Broken-symmetry\nshare (%)')
a2.grid(axis='y', color=GRID, lw=0.8)
a2.set_axisbelow(True)
for lo in (0.5, 1.0):
    a2.axvline(lo, color=INK2, lw=1, ls='--')
a2.set_xlabel(r'$N_\mathrm{FOD}$ (multireference character) at the Transition1x transition state')
a2.set_xlim(0, 1.8)
fig.text(0.99, 0.005, f'Spin class from the OMol25 label; {share(b):.1f} % of all transition states are broken-symmetry.',
         ha='right', fontsize=7.5, color=INK2)
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/fig_landscape_v2.{ext}', dpi=200, bbox_inches='tight')
print('wrote fig_landscape')
