import paramiko, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username='s242862',
            key_filename=os.path.expanduser('~/.ssh/dtu_key'))

def run(cmd):
    _, out, err = ssh.exec_command(cmd, get_pty=True)
    return out.read().decode('utf-8', errors='replace').strip()

remote_script = r"""
import h5py, numpy as np

T1X_H5 = '/home/energy/s242862/data/Transition1x.h5'

REACTIONS_23 = [
    # High(orig)
    'rxn7949', 'rxn8832', 'rxn1320', 'rxn4113', 'rxn8885',
    'rxn7945', 'rxn7937', 'rxn6196', 'rxn0346', 'rxn1150',
    # next-HIGH
    'rxn0896', 'rxn4518', 'rxn3107', 'rxn8837', 'rxn7060',
    'rxn8827', 'rxn4522', 'rxn7936', 'rxn1147', 'rxn0101',
    'rxn10005', 'rxn10054', 'rxn7957',
]

RELIABILITY = {
    'rxn7949':  'reliable',  'rxn8832':  'reliable',  'rxn1320': 'reliable',
    'rxn8885':  'reliable',  'rxn7945':  'reliable',  'rxn6196': 'reliable',
    'rxn1150':  'reliable',  'rxn3107':  'reliable',  'rxn7936': 'reliable',
    'rxn1147':  'reliable',  'rxn7957':  'reliable',
    'rxn4113':  'caveat',    'rxn7937':  'caveat',    'rxn0346': 'caveat',
    'rxn0896':  'caveat',    'rxn7060':  'caveat',    'rxn8827': 'caveat',
    'rxn10005': 'caveat',    'rxn10054': 'caveat',
    'rxn8837':  'borderline','rxn4522':  'borderline',
    'rxn4518':  'excluded',  'rxn0101':  'excluded',
}

# Covalent radii (Angstrom) — from Alvarez 2008
COV_RADII = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 16: 1.05, 17: 1.02}
ELEM = {1:'H', 6:'C', 7:'N', 8:'O', 9:'F', 16:'S', 17:'Cl'}
BOND_TOL = 1.25  # covalent radii sum * tolerance

def bond_exists(d, z1, z2):
    r1 = COV_RADII.get(z1, 0.7)
    r2 = COV_RADII.get(z2, 0.7)
    return d < (r1 + r2) * BOND_TOL

def find_rxn(h5, rxn):
    for split in ('test', 'val', 'train'):
        if split not in h5: continue
        for formula in h5[split].keys():
            if rxn in h5[split][formula]:
                grp = h5[split][formula][rxn]
                r_pos = np.array(grp['reactant']['positions'][0])
                p_pos = np.array(grp['product']['positions'][0])
                nums  = np.array(grp['reactant']['atomic_numbers'])
                return r_pos, p_pos, nums
    return None, None, None

def analyze_bonds(r_pos, p_pos, nums):
    n = len(nums)
    breaking, forming = [], []
    for i in range(n):
        for j in range(i+1, n):
            dr = np.linalg.norm(r_pos[i] - r_pos[j])
            dp = np.linalg.norm(p_pos[i] - p_pos[j])
            r_bond = bond_exists(dr, nums[i], nums[j])
            p_bond = bond_exists(dp, nums[i], nums[j])
            delta = abs(dr - dp)
            if delta < 0.25: continue  # no significant change
            e1, e2 = ELEM.get(nums[i],'?'), ELEM.get(nums[j],'?')
            pair = f'{e1}-{e2}' if e1 <= e2 else f'{e2}-{e1}'
            if r_bond and not p_bond:
                breaking.append((pair, dr, dp, delta))
            elif p_bond and not r_bond:
                forming.append((pair, dr, dp, delta))
            elif delta > 0.25 and (r_bond or p_bond):
                # significant change but both exist — partial bond order change
                if dr > dp:
                    forming.append((pair, dr, dp, delta))
                else:
                    breaking.append((pair, dr, dp, delta))
    return breaking, forming

def classify_bond(pair):
    # Is this bond type captured by C 2pz / N 2p / O 2pz AVAS?
    pi_pairs = {'C-C', 'C-N', 'C-O', 'N-N', 'N-O', 'O-O'}
    if pair in pi_pairs:
        return 'pi-ok'   # C/N/O p orbitals cover this
    elif 'H' in pair:
        return 'sigma-H' # C-H, N-H, O-H — sigma, not in pz
    else:
        return 'sigma'   # other sigma bonds

with h5py.File(T1X_H5, 'r') as h5:
    for status in ('reliable', 'caveat', 'borderline', 'excluded'):
        rxns = [r for r in REACTIONS_23 if RELIABILITY[r] == status]
        if not rxns: continue
        print(f'\n{"="*70}')
        print(f'{status.upper()} ({len(rxns)})')
        print('='*70)
        for rxn in rxns:
            r_pos, p_pos, nums = find_rxn(h5, rxn)
            if r_pos is None:
                print(f'{rxn}: not found'); continue
            breaking, forming = analyze_bonds(r_pos, p_pos, nums)
            # Sort by delta
            breaking.sort(key=lambda x: -x[3])
            forming.sort(key=lambda x: -x[3])
            b_str = ', '.join(f'{p}({d:.2f})' for p,dr,dp,d in breaking[:3])
            f_str = ', '.join(f'{p}({d:.2f})' for p,dr,dp,d in forming[:3])
            # Flag sigma bonds
            sigma_warn = []
            for pair, *_ in breaking + forming:
                c = classify_bond(pair)
                if c in ('sigma-H', 'sigma') and pair not in sigma_warn:
                    sigma_warn.append(pair)
            warn = f'  *** SIGMA: {", ".join(sigma_warn)}' if sigma_warn else ''
            print(f'{rxn:10s}  break: {b_str or "-":30s}  form: {f_str or "-":30s}{warn}')
"""

sftp = ssh.open_sftp()
with sftp.file('/home/energy/s242862/pipeline/_analyze_reaction_bonds_remote.py', 'w') as f:
    f.write(remote_script)
sftp.close()

print(run('module load Python/3.11.3-GCCcore-12.3.0 && python3 /home/energy/s242862/pipeline/_analyze_reaction_bonds_remote.py 2>&1'))
ssh.close()
