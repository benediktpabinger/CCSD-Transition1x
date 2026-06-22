import paramiko, sys, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username='s242862', key_filename=r'C:\Users\PabingerBenedikt\.ssh\dtu_key', timeout=30)

RESULTS = '/home/energy/s242862/mr_benchmark/results'

def load(rxn, tag):
    suffix = f'_{tag}' if tag else ''
    path = f'{RESULTS}/{rxn}_ccsdt{suffix}.json'
    sftp = ssh.open_sftp()
    try:
        with sftp.open(path) as f:
            return json.load(f)
    except IOError:
        return None
    finally:
        sftp.close()

for rxn, tag, label in [('rxn7949', 'delta_fw2', 'MACE+delta fw2'), ('rxn8885', 'uma_s', 'UMA-S')]:
    print(f'=== {rxn} ({label} geometry) vs ORCA geometry ===')
    alt  = load(rxn, tag)
    orca = load(rxn, '')
    for geom in ['reactant', 'transition_state', 'product']:
        a = alt['geometries'].get(geom, {})
        o = orca['geometries'].get(geom, {})
        print(f'  {geom}:')
        print(f'    {label:<16} e_hf={a.get("e_hf_Ha")}  e_ccsd={a.get("e_ccsd_Ha")}  e_t={a.get("e_t_Ha")}  e_ccsdt_eV={a.get("e_ccsdt_eV")}  err={a.get("error")}')
        print(f'    {"ORCA":<16} e_hf={o.get("e_hf_Ha")}  e_ccsd={o.get("e_ccsd_Ha")}  e_t={o.get("e_t_Ha")}  e_ccsdt_eV={o.get("e_ccsdt_eV")}  err={o.get("error")}')
    print()

ssh.close()
