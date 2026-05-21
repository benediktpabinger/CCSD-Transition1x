import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username='s242862', password='Butterbrot9797')
def run(cmd):
    _, out, _ = ssh.exec_command(cmd, get_pty=True)
    return out.read().decode('utf-8', errors='replace').strip()

print('Total NEVPT2 results:', run('ls /home/energy/s242862/nevpt2_results/ | wc -l'))
print()
print('Retry job results:')
RETRY = ['rxn4498','rxn1061','rxn4003','rxn4060','rxn1962','rxn1154','rxn5690','rxn4519','rxn4500']
for rxn in RETRY:
    done = run(f'test -f /home/energy/s242862/nevpt2_results/{rxn}_pyscf_avas/nevpt2_results.json && echo done || echo missing')
    print(f'  {rxn}: {done}')

print()
print('Latest retry log tail:')
print(run('ls -t /home/energy/s242862/logs/mr_nevpt2_retry_*.log | head -1 | xargs tail -8'))
ssh.close()
