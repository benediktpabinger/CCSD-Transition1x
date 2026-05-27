import paramiko, os, json

HOST = 'slid.fysik.dtu.dk'
USER = os.environ.get('SSH_USER', 's242862')
PASS = os.environ.get('SSH_PASS')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASS)
sftp = ssh.open_sftp()

with sftp.open('/home/energy/s242862/fod_results/fod_ranking.json') as f:
    data = json.load(f)

with open('fod_ranking.json', 'w') as f:
    json.dump(data, f, indent=2)

results = data['results']
print(f'Downloaded: {len(results)} reactions, {len(data["errors"])} errors')
print()
print('Top 15:')
for i, r in enumerate(results[:15], 1):
    print(f'  {i:3d}  {r["rxn"]}  NFOD={r["nfod"]:.4f}')
print()
print('Bottom 15:')
for i, r in enumerate(results[-15:], len(results)-14):
    print(f'  {i:3d}  {r["rxn"]}  NFOD={r["nfod"]:.4f}')
print()
print('Middle (around rank 135-145):')
mid = len(results) // 2
for i, r in enumerate(results[mid-5:mid+6], mid-4):
    print(f'  {i:3d}  {r["rxn"]}  NFOD={r["nfod"]:.4f}')

sftp.close()
ssh.close()
