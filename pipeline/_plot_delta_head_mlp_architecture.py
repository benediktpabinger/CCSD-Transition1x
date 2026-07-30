import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = r'c:\Transition 1X\Transition 1x\Transition1x\benchmark_plots\delta_head_mlp_architecture.png'

fig, ax = plt.subplots(figsize=(9, 14))
ax.set_xlim(0, 10)
ax.set_ylim(0, 34)
ax.axis('off')

def textbox(x_center, y_center, text, fc='#dde7f5', ec='black', fontsize=10,
            weight='normal', width_frac=0.92, pad=0.6):
    ax.text(x_center, y_center, text, ha='center', va='center', fontsize=fontsize,
            weight=weight, linespacing=1.5,
            bbox=dict(boxstyle=f'round,pad={pad}', fc=fc, ec=ec, lw=1.2),
            wrap=True)

def arrow(x, y1, y2):
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='-|>', lw=1.4, color='black'))

# Title
ax.text(5, 33.2, 'Delta Head MLP — Actual Architecture (verified by inspection)',
        ha='center', va='center', fontsize=13, weight='bold')
ax.text(5, 32.4, 'NonLinearReadoutBlock(irreps_in=1024x0e+1024x1o+1024x2e+1024x3o, MLP_irreps=64x0e, gate=SiLU)',
        ha='center', va='center', fontsize=8.5, style='italic', color='dimgray')

# Input row label
ax.text(5, 30.6, 'Input: node_feats[:, 1024:]  —  16,384-dim per atom',
        ha='center', fontsize=10, weight='bold')

# 4 sub-channel boxes
sub_y = 29.0
xs = [0.85, 3.1, 5.35, 7.6]
labels = ['1024 x 0e\n(scalar)\nUSED', '1024 x 1o\n(vector)\nunused',
          '1024 x 2e\n(rank-2)\nunused', '1024 x 3o\n(rank-3)\nunused']
colors = ['#bbdca8', '#e8e8e8', '#e8e8e8', '#e8e8e8']
for x, lab, c in zip(xs, labels, colors):
    textbox(x, sub_y, lab, fc=c, fontsize=8.5, pad=0.5)
ax.text(5, sub_y - 1.4, 'only the scalar (0e) block has any weight connection — the other three are mathematically present\nbut never influence the output',
        ha='center', fontsize=8, color='firebrick')

arrow(0.85, sub_y - 2.0, 26.6)

# linear_1
y1 = 25.0
textbox(5, y1, 'linear_1 : Linear(1024 -> 64), no bias\n65,536 params\nOnly instruction: i_in=0 (0e) -> i_out=0 (0e)',
        fc='#cfe0f5', fontsize=9.5, pad=0.7)
arrow(5, y1 - 1.4, 21.6)

# SiLU
y2 = 20.6
textbox(5, y2, 'SiLU(x) = x * sigmoid(x)\napplied elementwise to all 64 hidden units',
        fc='#fde9b6', fontsize=9.5, pad=0.6)
arrow(5, y2 - 1.2, 17.8)

# linear_2
y3 = 16.8
textbox(5, y3, 'linear_2 : Linear(64 -> 1), no bias\n64 params',
        fc='#cfe0f5', fontsize=9.5, pad=0.6)
arrow(5, y3 - 1.0, 14.4)

# output
y4 = 13.6
textbox(5, y4, 'per-atom delta  [1 scalar, eV]', fc='#bbdca8', fontsize=9.5, weight='bold', pad=0.5)
arrow(5, y4 - 0.8, 11.8)

# sum
y5 = 11.0
textbox(5, y5, 'sum over atoms -> delta_total [eV]', fc='#bbdca8', fontsize=9.5, pad=0.5)

# totals
y6 = 8.6
textbox(5, y6, 'Total trainable parameters: 65,536 + 64 = 65,600\n(matches earlier reported parameter count exactly)',
        fc='white', ec='black', fontsize=10.5, weight='bold', pad=0.7)

# key finding callout
y7 = 3.6
textbox(5, y7,
    'KEY FINDING\n\n'
    'Functionally this is a plain 1024->64->1 MLP with SiLU.\n'
    'Despite the 16,384-dim "equivariant" input, 3 of its 4\n'
    'irrep blocks (1o, 2e, 3o = all directional/angular info)\n'
    'carry zero weight and never reach the output.\n\n'
    'Cause: equivariant Linear layers only connect features\n'
    'of the SAME rotation type. MLP_irreps="64x0e" forces a\n'
    'pure-scalar hidden layer, so only 0e->0e is a legal path.\n\n'
    'Implication: the delta head never actually "sees" detailed\n'
    'bond-angle/directional geometry, only invariant scalar\n'
    'features. A plausible contributor to the NEB-optimization\n'
    'roughness diagnosed separately.',
    fc='#fce8e6', ec='firebrick', fontsize=9.5, pad=0.8)

fig.savefig(OUT, dpi=160, bbox_inches='tight')
print(f'Saved: {OUT}')
