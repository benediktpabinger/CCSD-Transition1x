import paramiko, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('slid.fysik.dtu.dk', username='s242862', key_filename=r'C:\Users\PabingerBenedikt\.ssh\dtu_key', timeout=30)

REACTIONS = [
    'rxn7949','rxn8832','rxn1320','rxn4113','rxn8885',
    'rxn7945','rxn7937','rxn6196','rxn0346','rxn1150',
    'rxn9246','rxn4498','rxn1061','rxn4003','rxn4004',
    'rxn4063','rxn4114','rxn4060','rxn1961','rxn1962',
    'rxn0896','rxn1154','rxn5690','rxn4513','rxn7955',
    'rxn4519','rxn4500','rxn2553','rxn8829','rxn1155',
]

remote_script = '''
import json
import os

REACTIONS = ''' + repr(REACTIONS) + '''
BARE = "/home/energy/s242862/mace_bare_neb_results"
DELTA = "/home/energy/s242862/mace_delta_neb_results_fw2"

print(f'{"rxn":<10} {"bare n_iter":>11} {"bare final_fmax":>16} {"delta n_iter":>12} {"delta final_fmax":>17} {"bare converged":>14} {"delta converged":>15}')
for rxn in REACTIONS:
    try:
        bare_fmaxs = json.load(open(f"{BARE}/{rxn}/fmaxs.json"))
        bare_n = len(bare_fmaxs)
        bare_final = bare_fmaxs[-1]
        bare_conv = os.path.exists(f"{BARE}/{rxn}/converged")
    except Exception:
        bare_n, bare_final, bare_conv = None, None, None
    try:
        delta_fmaxs = json.load(open(f"{DELTA}/{rxn}/fmaxs.json"))
        delta_n = len(delta_fmaxs)
        delta_final = delta_fmaxs[-1]
        delta_conv = os.path.exists(f"{DELTA}/{rxn}/converged")
    except Exception:
        delta_n, delta_final, delta_conv = None, None, None

    def f(x):
        return f"{x:.4f}" if isinstance(x, float) else "N/A"
    def i(x):
        return str(x) if x is not None else "N/A"

    print(f"{rxn:<10} {i(bare_n):>11} {f(bare_final):>16} {i(delta_n):>12} {f(delta_final):>17} {str(bare_conv):>14} {str(delta_conv):>15}")
'''

sftp = client.open_sftp()
with sftp.file('/home/energy/s242862/pipeline/_compare_neb_convergence_remote.py', 'w') as f:
    f.write(remote_script)
sftp.close()

_, out, err = client.exec_command(
    'module load Python/3.13.5-GCCcore-14.3.0 2>/dev/null; python3 /home/energy/s242862/pipeline/_compare_neb_convergence_remote.py'
)
print(out.read().decode('utf-8', errors='replace'))
e = err.read().decode('utf-8', errors='replace')
if e:
    print('STDERR:', e[-2000:])
client.close()
