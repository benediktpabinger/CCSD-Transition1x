"""
Print MACE model architecture info needed to construct the delta head.
Run on cluster: python inspect_mace_model.py
"""
import torch

MODEL = '/home/energy/s242862/mace_t1x_p10_epoch291.model'

m = torch.load(MODEL, map_location='cpu', weights_only=False)

print(f'type:             {type(m).__name__}')
print(f'num_interactions: {m.num_interactions}')
print(f'hidden_irreps:    {m.hidden_irreps}')
print(f'MLP_irreps:       {getattr(m, "MLP_irreps", "N/A")}')
print(f'num_heads:        {getattr(m, "num_heads", 1)}')
print(f'atomic_numbers:   {m.atomic_numbers.tolist()}')
print(f'r_max:            {m.r_max}')
print()
print('readouts:')
for i, r in enumerate(m.readouts):
    print(f'  [{i}] {r}')
print()

# node_feats shape: cat of num_interactions tensors, each hidden_irreps.dim
# Confirm by looking at the last readout input irreps
last = m.readouts[-1]
if hasattr(last, 'linear_1'):
    print(f'last readout linear_1 irreps_in:  {last.linear_1.irreps_in}')
    print(f'last readout linear_1 irreps_out: {last.linear_1.irreps_out}')
elif hasattr(last, 'linear'):
    print(f'last readout linear irreps_in:  {last.linear.irreps_in}')
    print(f'last readout linear irreps_out: {last.linear.irreps_out}')

# node_feats_out = cat(node_feats_concat) across all interactions
print(f'\nnode_feats_out dim (hidden_irreps.dim * num_interactions): '
      f'{m.hidden_irreps.dim} * {m.num_interactions} = '
      f'{m.hidden_irreps.dim * m.num_interactions}')
