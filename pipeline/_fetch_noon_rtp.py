"""
Read nevpt2_optts_results.json for all next-HIGH reactions on the cluster
and extract NOON (natural orbital occupation) data at R, TS, P.

Prints n_frac (0.05 < occ < 1.95) and key occupation pairs at each point
so we can assess active-space plausibility for the reliability table.

Run locally — connects via SSH key.
"""
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
import json, os, sys

BASE     = '/home/energy/s242862/nevpt2_optts_results'
BM_PATH  = '/home/energy/s242862/delta_head/full_benchmark_results.json'

# All 23 reactions with a converged OptTS (in order: High(orig) then next-HIGH)
ALL23 = [
    # High(orig)
    'rxn7949', 'rxn8832', 'rxn1320', 'rxn4113', 'rxn8885',
    'rxn7945', 'rxn7937', 'rxn6196', 'rxn0346', 'rxn1150',
    # next-HIGH
    'rxn0896', 'rxn4518', 'rxn3107', 'rxn8837', 'rxn7060',
    'rxn8827', 'rxn4522', 'rxn7936', 'rxn1147', 'rxn0101',
    'rxn10005', 'rxn10054', 'rxn7957',
]

STATUS = {
    'rxn7949': 'reliable',   'rxn8832': 'reliable',   'rxn1320': 'reliable*',
    'rxn4113': 'excl-nevpt2','rxn8885': 'reliable',   'rxn7945': 'reliable',
    'rxn7937': 'caveat',     'rxn6196': 'reliable',   'rxn0346': 'caveat',
    'rxn1150': 'reliable*',  'rxn0896': 'caveat',     'rxn4518': 'excl-geo',
    'rxn3107': 'reliable',   'rxn8837': 'borderline', 'rxn7060': 'caveat',
    'rxn8827': 'caveat',     'rxn4522': 'excl-geo',   'rxn7936': 'reliable',
    'rxn1147': 'reliable*',  'rxn0101': 'excl-geo',   'rxn10005':'caveat',
    'rxn10054':'caveat',     'rxn7957': 'reliable',
}

# CCSD(T) forward barriers from full_benchmark_results.json (meV)
CCSDT = {}
if os.path.exists(BM_PATH):
    with open(BM_PATH) as f:
        bm = json.load(f)
    for r in bm.get('reactions', []):
        if r.get('ccsd_fwd_meV') is not None:
            CCSDT[r['rxn']] = r['ccsd_fwd_meV']

def summarise_occ(nat_occ):
    n_frac = sum(1 for o in nat_occ if 0.05 < o < 1.95)
    n_low  = sum(1 for o in nat_occ if o <= 0.05)
    n_high = sum(1 for o in nat_occ if o >= 1.95)
    frac_occs = sorted([o for o in nat_occ if 0.05 < o < 1.95], reverse=True)
    if frac_occs:
        key = f'{frac_occs[0]:.3f}'
        if len(frac_occs) > 1:
            key += f' / {frac_occs[-1]:.3f}'
    else:
        key = '—'
    return n_frac, n_low, n_high, key

def load_json(rxn):
    path = os.path.join(BASE, f'{rxn}_avas', 'nevpt2_optts_results.json')
    if not os.path.exists(path):
        return None, f'not found: {path}'
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        by_label = {d['label']: d for d in data}
    elif isinstance(data, dict):
        by_label = data.get('points') and {d['label']: d for d in data['points']} or data
    else:
        return None, f'unknown type {type(data)}'
    r  = by_label.get('reactant') or by_label.get('R') or by_label.get('r')
    ts = (by_label.get('ts') or by_label.get('TS')
          or by_label.get('transition_state') or by_label.get('optts'))
    p  = by_label.get('product') or by_label.get('P') or by_label.get('p')
    if not (r and ts and p):
        return None, f'labels={list(by_label.keys())}'
    return (r, ts, p), None

print(f"{'rxn':10s} {'status':12s} {'ncas':5s} {'nel':4s} "
      f"{'frac R':6s} {'frac TS':7s} {'frac P':6s} "
      f"{'TS key pair':17s} {'n<0.05@TS':9s} "
      f"{'MR pattern':12s} {'NEVPT2 fwd':10s} {'CCSDT fwd':9s} {'diff':6s}")
print('-' * 120)

for rxn in ALL23:
    st = STATUS[rxn]
    result, err = load_json(rxn)
    if result is None:
        cct = CCSDT.get(rxn)
        cct_s = f'{cct:.0f}' if cct else '—'
        print(f'{rxn:10s} {st:12s} {"—":5s} {"—":4s} '
              f'{"—":6s} {"—":7s} {"—":6s} '
              f'{"—":17s} {"—":9s} '
              f'{"—":12s} {"—":10s} {cct_s:9s} {"—":6s}   [{err}]')
        continue

    r_d, ts_d, p_d = result
    ncas = ts_d.get('ncas', '?')
    nel  = ts_d.get('nelecas', '?')

    r_nf,  r_nl,  r_nh,  r_key  = summarise_occ(r_d['nat_occ'])
    ts_nf, ts_nl, ts_nh, ts_key = summarise_occ(ts_d['nat_occ'])
    p_nf,  p_nl,  p_nh,  p_key  = summarise_occ(p_d['nat_occ'])

    # MR pattern
    if ts_nf >= r_nf and ts_nf >= p_nf:
        pat = 'TS-max OK'
    elif r_nf == 0:
        pat = '0@R (SR edukt)'
    elif ts_nf < r_nf and ts_nf < p_nf:
        pat = 'TS-min WARN'
    elif ts_nf < r_nf:
        pat = 'R>TS WARN'
    elif ts_nf < p_nf:
        pat = 'P>TS'
    else:
        pat = 'equal'

    # NEVPT2 barrier: e_total@TS - e_total@R (eV → meV)
    nev_fwd = None
    try:
        e_r  = r_d['e_total_eV']
        e_ts = ts_d['e_total_eV']
        e_p  = p_d['e_total_eV']
        nev_fwd = (e_ts - e_r) * 1000   # meV
        nev_rev = (e_ts - e_p) * 1000
    except (KeyError, TypeError):
        pass

    cct = CCSDT.get(rxn)
    diff_s = f'{nev_fwd - cct:+.0f}' if (nev_fwd is not None and cct is not None) else '—'
    nev_s  = f'{nev_fwd:.0f}'         if nev_fwd is not None else '—'
    cct_s  = f'{cct:.0f}'             if cct     is not None else '—'

    print(f'{rxn:10s} {st:12s} {ncas:5} {nel:4} '
          f'{r_nf:6d} {ts_nf:7d} {p_nf:6d} '
          f'{ts_key:17s} {ts_nl:9d} '
          f'{pat:12s} {nev_s:10s} {cct_s:9s} {diff_s:6s}')
    sys.stdout.flush()

print()
print('diff = NEVPT2_fwd - CCSD(T)_fwd  (meV, + means NEVPT2 higher barrier)')
print('n<0.05@TS: number of near-empty orbitals at TS (intruder risk)')
"""

sftp = ssh.open_sftp()
with sftp.file('/home/energy/s242862/pipeline/_fetch_noon_rtp_remote.py', 'w') as f:
    f.write(remote_script)
sftp.close()

print(run('module load Python/3.11.3-GCCcore-12.3.0 && '
          'python3 /home/energy/s242862/pipeline/_fetch_noon_rtp_remote.py 2>&1'))
ssh.close()
