import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], height_ratios=[1, 1], hspace=0.35, wspace=0.3)

# ---------- Panel 1: architecture flow diagram ----------
ax = fig.add_subplot(gs[0, 0])
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

boxes = [
    (5, 9.3, 'Geometry', '#ecf0f1'),
    (5, 8.0, 'MACE encoder (FROZEN)\nmace_t1x_p10_compiled.model', '#aed6f1'),
    (5, 6.5, 'node_feats  [N_atoms x 17408]\nHIDDEN_IRREPS = 1024x0e+1024x1o+1024x2e+1024x3o', '#d6eaf8'),
    (5, 5.0, 'slice [:, 1024:]  ->  higher-order feats [N_atoms x 16384]\n(directional/angular info, geometry-sensitive)', '#fcf3cf'),
    (5, 3.6, 'NonLinearReadoutBlock (trainable)\nMLP_IRREPS = 64x0e,  gate = SiLU', '#f5b7b1'),
    (5, 2.2, 'per-atom delta  [N_atoms x 1]', '#e8daef'),
    (5, 1.0, 'sum over atoms -> delta_total [eV]\ndelta = E(wB97M-V/def2-TZVP) - E(wB97X-D3/6-31G(d))', '#a9dfbf'),
]
for x, y, text, color in boxes:
    ax.add_patch(mpatches.FancyBboxPatch((x-3.6, y-0.55), 7.2, 1.05, boxstyle='round,pad=0.08',
                                          facecolor=color, edgecolor='black', linewidth=0.8))
    ax.text(x, y, text, ha='center', va='center', fontsize=9.5)
for (x1, y1, *_), (x2, y2, *_) in zip(boxes[:-1], boxes[1:]):
    ax.annotate('', xy=(x2, y2+0.55), xytext=(x1, y1-0.55),
                arrowprops=dict(arrowstyle='-|>', color='black', lw=1.2))
ax.set_title('Architecture: frozen MACE + trainable delta head', fontsize=12.5, weight='bold')

# ---------- Panel 2: training sampling diagram ----------
ax = fig.add_subplot(gs[0, 1])
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.set_title('Training data: stratified TS-centered sampling', fontsize=12.5, weight='bold')

import numpy as np
xs = np.linspace(0.5, 9.5, 400)
n = len(xs)
ts_idx = int(n*0.45)
ys = np.piecewise(xs, [xs <= xs[ts_idx], xs > xs[ts_idx]],
                   [lambda x: 1 + 4*((x-xs[0])/(xs[ts_idx]-xs[0])),
                    lambda x: 5 - 3.5*((x-xs[ts_idx])/(xs[-1]-xs[ts_idx]))])
ax.plot(xs, ys, color='black', linewidth=1.5)
seg_bounds = [0, n//4, ts_idx, ts_idx + (n-ts_idx)//2, n-1]
seg_colors = ['#aed6f1', '#f9e79f', '#f5b7b1', '#abebc6']
seg_labels = ['seg1\nreactant\nvalley', 'seg2\napproach\nto barrier', 'seg3\nimmediate\npost-TS', 'seg4\nproduct\nvalley']
for i in range(4):
    a, b = seg_bounds[i], seg_bounds[i+1]
    ax.axvspan(xs[a], xs[b], color=seg_colors[i], alpha=0.4)
    pts = np.linspace(a, b-1, 5).astype(int)
    ax.scatter(xs[pts], ys[pts], color='black', s=18, zorder=5)
    ax.text((xs[a]+xs[b])/2, 0.3, seg_labels[i], ha='center', fontsize=8)
ax.annotate('TS\n(argmax wB97X energy)', xy=(xs[ts_idx], ys[ts_idx]), xytext=(xs[ts_idx], 6.3),
            ha='center', fontsize=8.5, arrowprops=dict(arrowstyle='-|>', color='black', lw=1))
ax.text(5, -0.6, '5000 reactions (seed=42)  x  20 geoms/reaction (5 per segment)  ~80,600 total training geoms',
        ha='center', fontsize=9.5, style='italic')

# ---------- Panel 3: training hyperparameters table ----------
ax = fig.add_subplot(gs[1, 0])
ax.axis('off')
ax.set_title('Training configuration', fontsize=12.5, weight='bold', pad=10)
rows = [
    ('Target', 'delta = E(wB97M-V/def2-TZVP) - E(wB97X-D3/6-31G(d))'),
    ('MACE backbone', 'frozen throughout; only head trains'),
    ('Loss', 'Huber(delta_E, d=0.1eV) + force_weight x Huber(delta_F, d=0.1eV)'),
    ('Force weight (this run)', '2.0  (swept: 0.5 / 1.0 / 2.0 in parallel)'),
    ('Optimizer', 'Adam, lr=1e-3'),
    ('Batch size', '64'),
    ('LR schedule', 'ReduceLROnPlateau (patience=10, factor=0.5)'),
    ('Stopping', 'early stop when lr < 1e-6  (max 200 epochs)'),
    ('Checkpoint selection', 'best val_f_f (force loss on NEB-path force val set)'),
    ('Hardware', 'H100/H200 GPU (TorchScript MACE targets sm_90a)'),
]
table = ax.table(cellText=[[k, v] for k, v in rows], colLabels=['Parameter', 'Value'],
                  loc='center', cellLoc='left', colWidths=[0.32, 0.68])
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1, 1.55)
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#cccccc')
    if row == 0:
        cell.set_facecolor('#2c3e50'); cell.set_text_props(color='white', weight='bold')
    elif col == 0:
        cell.set_facecolor('#ecf0f1'); cell.set_text_props(weight='bold')

# ---------- Panel 4: validation design ----------
ax = fig.add_subplot(gs[1, 1])
ax.axis('off')
ax.set_title('Validation design', fontsize=12.5, weight='bold', pad=10)
text = (
    "TWO SEPARATE VAL SETS, DIFFERENT JOBS\n\n"
    "val_sample (1024 geoms, fixed at epoch 1)\n"
    "  -> drives LR scheduler (val_loss)\n"
    "  -> broad mix: energy-only + force-labeled geoms\n"
    "  -> stable signal for \"is the model still improving?\"\n\n"
    "val_f_sample (~2240 geoms, force-labeled only)\n"
    "  -> drives checkpoint saving (val_f_f)\n"
    "  -> Group A (174 rxns): last 10 NEB images,\n"
    "     wB97M-V EnGrad computed specifically for this\n"
    "  -> Group B (51 rxns): T1x geometries, full forces\n"
    "  -> targeted at actual NEB inference conditions\n\n"
    "WHY SPLIT THEM\n"
    "val_loss is energy-dominated (large ~3eV offset\n"
    "converges fast); using it to pick checkpoints would\n"
    "ignore force quality, which matters most for NEB path\n"
    "optimization. val_f_f isolates exactly that signal."
)
ax.text(0.02, 0.97, text, transform=ax.transAxes, fontsize=9.3, va='top', ha='left', family='monospace',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#fdfefe', edgecolor='#aab7b8'))

fig.suptitle('MACE+δ Correction Head — Architecture, Sampling, and Training Configuration (v2, fw=2.0)',
             fontsize=15, weight='bold', y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.965])
out = r'c:\Transition 1X\Transition 1x\Transition1x\benchmark_plots\delta_head_training_summary.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print('Saved', out)
