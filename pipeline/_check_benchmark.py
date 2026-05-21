import paramiko, json

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username='s242862', password='Butterbrot9797')
def run(cmd):
    _, out, _ = ssh.exec_command(cmd)
    return out.read().decode('utf-8', errors='replace').strip()

BOTTOM10 = ['rxn9246','rxn4498','rxn1061','rxn4003','rxn4004','rxn4063','rxn4114','rxn4060','rxn1961','rxn1962']
MIDDLE10 = ['rxn0896','rxn1154','rxn5690','rxn4513','rxn7955','rxn4519','rxn4500','rxn2553','rxn8829','rxn1155']
NEW20    = BOTTOM10 + MIDDLE10

print('=== squeue ===')
print(run('squeue -u s242862 -h'))

print()
print(f"{'Rxn':12s}  {'Group':7s}  {'CCSD(T)':>8}  {'NEVPT2':>8}")
print('-' * 45)

for rxn in NEW20:
    group = 'bottom' if rxn in BOTTOM10 else 'middle'
    ccsdt_path  = f'/home/energy/s242862/mr_benchmark/results/{rxn}_ccsdt.json'
    nevpt2_path = f'/home/energy/s242862/nevpt2_results/{rxn}_pyscf_avas/nevpt2_results.json'

    ccsdt_done = run(f'test -f {ccsdt_path} && echo yes || echo no')
    nevpt2_done = run(f'test -f {nevpt2_path} && echo yes || echo no')

    # If done, show barrier
    if ccsdt_done == 'yes':
        raw = run(f'cat {ccsdt_path}')
        try:
            d = json.loads(raw)
            fwd = d.get('barrier_fwd_meV', '?')
            ccsdt_str = f'{fwd:.0f}meV' if isinstance(fwd, float) else 'err'
        except:
            ccsdt_str = 'parse err'
    else:
        ccsdt_str = 'pending'

    if nevpt2_done == 'yes':
        raw = run(f'cat {nevpt2_path}')
        try:
            d = json.loads(raw)
            fwd = d.get('barrier_forward_meV', '?')
            nevpt2_str = f'{fwd:.0f}meV' if isinstance(fwd, float) else 'err'
        except:
            nevpt2_str = 'parse err'
    else:
        nevpt2_str = 'pending'

    print(f'{rxn:12s}  {group:7s}  {ccsdt_str:>8}  {nevpt2_str:>8}')

ssh.close()
