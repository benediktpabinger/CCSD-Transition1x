import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

METHODS = ['UMA-S', 'UMA-M', 'eSEN', 'MACE+δ fw2']

RMSD = {
    'High': [0.150, 0.189, 0.287, 0.134],
    'Mid':  [0.036, 0.033, 0.032, 0.116],
    'Low':  [0.007, 0.007, 0.008, 0.053],
}
MAE = {
    'High': [323, 400, 373, 272],
    'Mid':  [12, 12, 10, 159],
    'Low':  [4, 3, 5, 89],
}

def best_idx(vals):
    return int(np.argmin(vals))

fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(3, 1, height_ratios=[0.55, 0.55, 1.1], hspace=0.55)

def draw_table(ax, data, title, fmt, unit):
    ax.axis('off')
    rows = list(data.keys())
    n_cols = len(METHODS)
    cell_text = []
    for r in rows:
        vals = data[r]
        b = best_idx(vals)
        cell_text.append([fmt(v) + (' ★' if i == b else '') for i, v in enumerate(vals)])
    table = ax.table(cellText=cell_text, rowLabels=rows, colLabels=METHODS,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#cccccc')
        if row == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', weight='bold')
        elif col == -1:
            cell.set_facecolor('#ecf0f1')
            cell.set_text_props(weight='bold')
        else:
            vals = data[rows[row-1]]
            b = best_idx(vals)
            if col == b:
                cell.set_facecolor('#d4efdf')
                cell.set_text_props(weight='bold')
    ax.set_title(title, fontsize=13, weight='bold', pad=14)

ax1 = fig.add_subplot(gs[0])
draw_table(ax1, RMSD, 'TS Geometry RMSD vs ORCA ωB97M-V NEB, by MR category  (Å, lower=better, ★=best)',
           lambda v: f'{v:.3f}', 'Å')

ax2 = fig.add_subplot(gs[1])
draw_table(ax2, MAE, 'Forward Barrier MAE vs ωB97M-V reference, by MR category  (meV, lower=better, ★=best)',
           lambda v: f'{v:.0f}', 'meV')

ax3 = fig.add_subplot(gs[2])
ax3.axis('off')
explainer = (
    "WHAT THIS SHOWS\n"
    "30 benchmark reactions from Transition1x, split into High / Mid / Low multireference (MR) character (10 each, by Fractional\n"
    "Occupation Density). Four NEB-derived TS-finding methods are compared against the reference ORCA ωB97M-V/def2-TZVP NEB:\n"
    "two general-purpose foundation models (UMA-S, UMA-M), one foundation model fine-tuned for transition states (eSEN), and\n"
    "MACE with a delta-correction head trained to shift its energies toward ωB97M-V (MACE+δ fw2).\n\n"
    "THE PATTERN\n"
    "On Mid/Low-MR reactions (mostly single-reference character), the three foundation models are essentially exact\n"
    "(RMSD ≤ 0.04 Å, MAE ≤ 12 meV) — MACE+δ is 3–10× worse here. On High-MR reactions (genuine multireference\n"
    "character), the ranking flips: MACE+δ has the lowest RMSD and lowest barrier MAE of the four.\n\n"
    "IMPORTANT CAVEAT\n"
    "This apparent High-MR win for MACE+δ should be read carefully: when checked against actual CASSCF+NEVPT2 (true\n"
    "multireference) transition states — available for only 4 of the 10 High-MR reactions so far — MACE+δ is actually the\n"
    "WORST of the four methods, while UMA-S is statistically tied with ORCA itself. This suggests MACE+δ has learned to\n"
    "mimic ωB97M-V's specific (and imperfect) description of these systems, rather than capturing true multireference\n"
    "character. The 4-reaction subset is also not representative of the full 10 (CASSCF convergence wasn't random), so\n"
    "this picture may still shift once the other 6 High-MR CASSCF optimizations converge."
)
ax3.text(0.0, 1.0, explainer, transform=ax3.transAxes, fontsize=10.5,
         va='top', ha='left', family='monospace',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#fdfefe', edgecolor='#aab7b8'))

fig.suptitle('MLIP / Foundation Model Benchmark on Transition1x: TS Geometry & Barrier Accuracy by MR Category',
             fontsize=14.5, weight='bold', y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(r'c:\Transition 1X\Transition 1x\Transition1x\benchmark_plots\summary_figure.png', dpi=160, bbox_inches='tight')
print('Saved summary_figure.png')
