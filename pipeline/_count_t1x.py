import paramiko, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username=os.environ['SSH_USER'], password=os.environ['SSH_PASS'])

def run(cmd):
    _, out, _ = ssh.exec_command(cmd, get_pty=True)
    return out.read().decode('utf-8', errors='replace').strip()

script = r"""
import h5py, numpy as np

f = h5py.File('/home/energy/s242862/data/Transition1x.h5', 'r')

total_rxns = 0
total_geoms = 0

for split in f.keys():
    split_rxns = 0
    split_geoms = 0
    for formula in f[split].keys():
        for rxn in f[split][formula].keys():
            g = f[split][formula][rxn]
            n = g['positions'].shape[0]
            split_rxns += 1
            split_geoms += n
    total_rxns += split_rxns
    total_geoms += split_geoms
    print(f'{split:6s}: {split_rxns:6d} reactions, {split_geoms:8d} geometries')

print(f'{"TOTAL":6s}: {total_rxns:6d} reactions, {total_geoms:8d} geometries')

# sample one reaction to show geoms per rxn
split = 'train'
formula = list(f[split].keys())[0]
rxn = list(f[split][formula].keys())[0]
n = f[split][formula][rxn]['positions'].shape[0]
print(f'\nSample: {rxn} has {n} geometries')
"""

print(run(f'module load Python/3.13.5-GCCcore-14.3.0 && python3 -c "{script}" 2>&1'))
ssh.close()
