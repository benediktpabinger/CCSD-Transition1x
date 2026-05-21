import paramiko, json

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username='s242862', password='Butterbrot9797')
def run(cmd):
    _, out, _ = ssh.exec_command(cmd)
    return out.read().decode('utf-8', errors='replace').strip()

data = json.loads(run('cat /home/energy/s242862/fod_results/fod_ranking.json'))
results = data['results']
print(f'Total reactions: {len(results)}')
print()
for i, r in enumerate(results):
    print(f'{i+1:3d}  {r["rxn"]}  {r["nfod"]:.6f}')

ssh.close()
