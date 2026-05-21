import paramiko, json, os
import numpy as np

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username='s242862', password='Butterbrot9797')

sftp = ssh.open_sftp()
sftp.get('/home/energy/s242862/delta_head/eval_neb_results.json',
         'delta_head_eval_neb_results.json')
sftp.close()
ssh.close()
print('Downloaded.')

with open('delta_head_eval_neb_results.json') as f:
    data = json.load(f)

rxns = data['reactions']

fwd_true  = np.array([r['fwd_true_meV']  for r in rxns])
fwd_mace  = np.array([r['fwd_mace_meV']  for r in rxns])
fwd_delta = np.array([r['fwd_delta_meV'] for r in rxns])
rev_true  = np.array([r['rev_true_meV']  for r in rxns])
rev_mace  = np.array([r['rev_mace_meV']  for r in rxns])
rev_delta = np.array([r['rev_delta_meV'] for r in rxns])

fwd_err_mace  = fwd_delta - fwd_true    # mace+delta error
fwd_err_base  = fwd_mace  - fwd_true    # mace-only error
rev_err_mace  = rev_delta - rev_true
rev_err_base  = rev_mace  - rev_true

fwd_improvement = np.abs(fwd_err_base) - np.abs(fwd_err_mace)   # +ve = delta helped
rev_improvement = np.abs(rev_err_base) - np.abs(rev_err_mace)

print(f'\n=== Forward barrier (n={len(rxns)}) ===')
print(f'Delta correction mean:  {(fwd_delta - fwd_mace).mean():.1f} meV')
print(f'Delta correction std:   {(fwd_delta - fwd_mace).std():.1f} meV')
print(f'Improved by delta:      {(fwd_improvement > 0).sum()} / {len(rxns)}')
print(f'Hurt by delta:          {(fwd_improvement < 0).sum()} / {len(rxns)}')
print(f'Mean improvement:       {fwd_improvement.mean():.1f} meV')

print(f'\n=== Reverse barrier (n={len(rxns)}) ===')
print(f'Delta correction mean:  {(rev_delta - rev_mace).mean():.1f} meV')
print(f'Delta correction std:   {(rev_delta - rev_mace).std():.1f} meV')
print(f'Improved by delta:      {(rev_improvement > 0).sum()} / {len(rxns)}')
print(f'Hurt by delta:          {(rev_improvement < 0).sum()} / {len(rxns)}')
print(f'Mean improvement:       {rev_improvement.mean():.1f} meV')

print(f'\n=== Signed errors (bias) ===')
print(f'Forward — MACE bias:     {fwd_err_base.mean():.1f} meV  (std {fwd_err_base.std():.1f})')
print(f'Forward — delta bias:    {fwd_err_mace.mean():.1f} meV  (std {fwd_err_mace.std():.1f})')
print(f'Reverse — MACE bias:     {rev_err_base.mean():.1f} meV  (std {rev_err_base.std():.1f})')
print(f'Reverse — delta bias:    {rev_err_mace.mean():.1f} meV  (std {rev_err_mace.std():.1f})')

print(f'\n=== Delta correction per image (TS vs endpoints) ===')
# Per-reaction: delta correction = delta_pred(TS) - delta_pred(reactant)
# We don't have per-image delta, but we can infer:
# fwd_delta - fwd_mace = [delta(TS) - delta(R)] in meV
# rev_delta - rev_mace = [delta(TS) - delta(P)] in meV
delta_ts_minus_r = fwd_delta - fwd_mace   # delta correction at TS relative to reactant
delta_ts_minus_p = rev_delta - rev_mace   # delta correction at TS relative to product
delta_r_minus_p  = delta_ts_minus_r - delta_ts_minus_p  # delta(R) - delta(P)

print(f'delta(TS) - delta(R) mean: {delta_ts_minus_r.mean():.1f} meV  std {delta_ts_minus_r.std():.1f}')
print(f'delta(TS) - delta(P) mean: {delta_ts_minus_p.mean():.1f} meV  std {delta_ts_minus_p.std():.1f}')
print(f'delta(R)  - delta(P) mean: {delta_r_minus_p.mean():.1f} meV  std {delta_r_minus_p.std():.1f}')

print(f'\n=== Worst reverse degradations ===')
worst_rev = sorted(zip(rev_improvement, rxns), key=lambda x: x[0])[:10]
for imp, r in worst_rev:
    print(f"  {r['rxn']:12s}  rev_true={r['rev_true_meV']:6.0f}  rev_mace={r['rev_mace_meV']:6.0f}  rev_delta={r['rev_delta_meV']:6.0f}  change={-imp:+.0f} meV")
