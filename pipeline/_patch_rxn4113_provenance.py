import paramiko, json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('slid.fysik.dtu.dk', username='s242862', key_filename=r'C:\Users\PabingerBenedikt\.ssh\dtu_key', timeout=30)

path = '/home/energy/s242862/nevpt2_optts_results/rxn4113_avas/nevpt2_optts_results.json'
sftp = client.open_sftp()
with sftp.open(path) as f:
    data = json.load(f)

# Add provenance fields that today's pipeline writes automatically.
# Values based on known history: this result came from the original
# first-run (threshold=0.2, before pruning/mc2step were added).
# threshold=0.4 gives a degenerate (16e,8o) AVAS space for this molecule
# (nelecas=2*ncas, all orbitals fully occupied) — hence threshold=0.2 is
# the correct setting, producing a meaningful (16e,10o) active space.
data['avas_threshold']  = 0.2
data['avas_ncas']       = data['ncas']       # AVAS and final space are the same (no pruning)
data['avas_nelecas']    = data['nelecas']
data['prune_enabled']   = False
data['prune_info']      = {
    'attempted': False,
    'note': 'Pruning not yet implemented when this result was produced. '
            'threshold=0.4 yields degenerate (16e,8o) AVAS space for rxn4113 '
            '(nelecas=2*ncas); threshold=0.2 required to obtain meaningful active space.'
}
data['solver_used_ts']      = 'mc1step'
data['solver_used_ts_opt']  = 'mc1step'
data['provenance_note']     = (
    'Result from original first run (pre-2026-06-17 fixes). '
    'All other 9 reactions use threshold=0.4 with pruning+mc2step fallback (jobs 10515621/10518016). '
    'rxn4113 is an exception: threshold=0.4 gives degenerate AVAS active space, '
    'so threshold=0.2 (producing (16e,10o)) is used instead.'
)

with sftp.open(path, 'w') as f:
    json.dump(data, f, indent=2)
sftp.close()
client.close()

print('Patched rxn4113 provenance fields:')
print(f'  avas_threshold = {data["avas_threshold"]}')
print(f'  avas_ncas/nelecas = ({data["avas_ncas"]}o, {data["avas_nelecas"]}e)')
print(f'  prune_enabled = {data["prune_enabled"]}')
print(f'  solver_used_ts = {data["solver_used_ts"]}')
