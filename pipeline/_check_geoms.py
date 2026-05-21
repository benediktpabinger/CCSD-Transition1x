import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username='s242862', password='Butterbrot9797')
def run(cmd):
    _, out, _ = ssh.exec_command(cmd)
    return out.read().decode('utf-8', errors='replace').strip()

BOTTOM10 = ['rxn9246','rxn4498','rxn1061','rxn4003','rxn4004','rxn4063','rxn4114','rxn4060','rxn1961','rxn1962']
MIDDLE10 = ['rxn0896','rxn1154','rxn5690','rxn4513','rxn7955','rxn4519','rxn4500','rxn2553','rxn8829','rxn1155']

all_rxns = BOTTOM10 + MIDDLE10

print(f"{'Rxn':12s}  {'reactant':>9}  {'ts':>9}  {'product':>9}")
print('-' * 48)
for rxn in all_rxns:
    base = f'/home/energy/s242862/orca_neb_results/{rxn}'
    r = run(f'test -f {base}/reactant.xyz && echo yes || echo no')
    t = run(f'test -f {base}/transition_state.xyz && echo yes || echo no')
    p = run(f'test -f {base}/product.xyz && echo yes || echo no')
    flag = '' if r == t == p == 'yes' else '  <-- MISSING'
    print(f'{rxn:12s}  {r:>9}  {t:>9}  {p:>9}{flag}')

ssh.close()
