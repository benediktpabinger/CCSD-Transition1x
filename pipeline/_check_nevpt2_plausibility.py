"""
NEVPT2 plausibility check for the 20 new benchmark reactions.
Mirrors the analysis in multireference_screening.md.
"""
import paramiko, json
import numpy as np

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username='s242862', password='Butterbrot9797')

def run(cmd):
    _, out, _ = ssh.exec_command(cmd, get_pty=True)
    return out.read().decode('utf-8', errors='replace').strip()

sftp = ssh.open_sftp()

def load_json(path):
    try:
        with sftp.open(path) as f:
            return json.load(f)
    except Exception:
        return None

BOTTOM10 = ['rxn9246','rxn4498','rxn1061','rxn4003','rxn4004',
            'rxn4063','rxn4114','rxn4060','rxn1961','rxn1962']
MIDDLE10 = ['rxn0896','rxn1154','rxn5690','rxn4513','rxn7955',
            'rxn4519','rxn4500','rxn2553','rxn8829','rxn1155']
NEW20 = BOTTOM10 + MIDDLE10

FRAC_LO, FRAC_HI = 0.05, 1.95

def count_frac(occ):
    return sum(1 for n in occ if FRAC_LO < n < FRAC_HI)

def min_frac_occ(occ):
    fracs = [n for n in occ if FRAC_LO < n < FRAC_HI]
    return min(fracs) if fracs else None

results = []

for rxn in NEW20:
    group = 'bottom' if rxn in BOTTOM10 else 'middle'
    nev = load_json(f'/home/energy/s242862/nevpt2_results/{rxn}_pyscf_avas/nevpt2_results.json')
    cct = load_json(f'/home/energy/s242862/mr_benchmark/results/{rxn}_ccsdt.json')

    cct_fwd = cct.get('barrier_fwd_meV') if cct else None
    cct_rev = cct.get('barrier_rev_meV') if cct else None

    if nev is None:
        results.append({'rxn': rxn, 'group': group, 'status': 'MISSING',
                        'cct_fwd': cct_fwd, 'cct_rev': cct_rev})
        continue

    nev_fwd  = nev.get('barrier_forward_meV')
    nev_rev  = nev.get('barrier_reverse_meV')
    ncas     = nev.get('ncas')
    nelecas  = nev.get('nelecas')
    active   = f'({nelecas}e,{ncas}o)' if ncas and nelecas else '?'

    r_occ  = nev.get('reactant', {}).get('nat_occ', [])
    ts_occ = nev.get('ts', {}).get('nat_occ', [])
    p_occ  = nev.get('product', {}).get('nat_occ', [])

    frac_r  = count_frac(r_occ)
    frac_ts = count_frac(ts_occ)
    frac_p  = count_frac(p_occ)

    flags = []
    if frac_r == 0:
        flags.append('0 frac@R')
    if frac_ts == 0:
        flags.append('0 frac@TS')
    if frac_p == 0:
        flags.append('0 frac@P')
    if nev_rev is not None and nev_rev < 0:
        flags.append(f'neg rev ({nev_rev:.0f}meV)')
    if cct_fwd is not None and nev_fwd is not None and abs(nev_fwd - cct_fwd) > 300:
        flags.append(f'|NEV-CCT|={abs(nev_fwd-cct_fwd):.0f}meV')

    status = 'RED FLAG' if flags else 'Reliable'

    results.append({
        'rxn': rxn, 'group': group, 'status': status, 'active': active,
        'frac_r': frac_r, 'frac_ts': frac_ts, 'frac_p': frac_p,
        'nev_fwd': nev_fwd, 'nev_rev': nev_rev,
        'cct_fwd': cct_fwd, 'cct_rev': cct_rev,
        'flags': flags,
        'r_occ': r_occ, 'ts_occ': ts_occ, 'p_occ': p_occ,
    })

sftp.close()
ssh.close()

# Print summary table
print('=' * 100)
print('NEVPT2 PLAUSIBILITY -- 20 NEW BENCHMARK REACTIONS')
print('=' * 100)
print(f'{"Rxn":12s} {"Grp":6s} {"Status":10s} {"Active":12s} {"fR":>3} {"fTS":>4} {"fP":>3} '
      f'{"NEV_fwd":>8} {"CCT_fwd":>8} {"diff":>6}  Flags')
print('-' * 100)

for r in results:
    nf = r.get('nev_fwd'); cf = r.get('cct_fwd')
    nf_s = f'{nf:.0f}' if nf is not None else 'N/A'
    cf_s = f'{cf:.0f}' if cf is not None else 'N/A'
    diff_s = f'{nf-cf:+.0f}' if (nf and cf) else 'N/A'

    if r['status'] == 'MISSING':
        print(f'{r["rxn"]:12s} {r["group"]:6s} {"MISSING":10s} {"---":12s} {"?":>3} {"?":>4} '
              f'{"?":>3} {"N/A":>8} {cf_s:>8} {"---":>6}')
        continue

    flags_s = '; '.join(r['flags']) if r['flags'] else ''
    print(f'{r["rxn"]:12s} {r["group"]:6s} {r["status"]:10s} {r["active"]:12s} '
          f'{r["frac_r"]:>3} {r["frac_ts"]:>4} {r["frac_p"]:>3} '
          f'{nf_s:>8} {cf_s:>8} {diff_s:>6}  {flags_s}')

# Counts
reliable = [r for r in results if r['status'] == 'Reliable']
red_flag = [r for r in results if r['status'] == 'RED FLAG']
missing  = [r for r in results if r['status'] == 'MISSING']
print(f'\nReliable: {len(reliable)}  |  Red flag: {len(red_flag)}  |  Missing/Failed: {len(missing)}')

# Detail on red flags
if red_flag:
    print('\n-- RED FLAG DETAILS -------------------------------------------------------')
    for r in red_flag:
        print(f'\n{r["rxn"]} ({r["group"]}):')
        print(f'  Active space: {r["active"]}')
        print(f'  Fractional occupations: R={r["frac_r"]}, TS={r["frac_ts"]}, P={r["frac_p"]}')
        if r['ts_occ']:
            key_ts = sorted([n for n in r['ts_occ'] if FRAC_LO < n < FRAC_HI])
            print(f'  TS key occs: {[round(n,3) for n in key_ts]}')
        if r['r_occ']:
            key_r = sorted([n for n in r['r_occ'] if FRAC_LO < n < FRAC_HI])
            print(f'  R  key occs: {[round(n,3) for n in key_r]}')
        print(f'  Flags: {"; ".join(r["flags"])}')
        print(f'  NEVPT2: fwd={r["nev_fwd"]:.0f} meV  rev={r["nev_rev"]:.0f} meV')
        print(f'  CCSD(T): fwd={r["cct_fwd"]:.0f} meV  rev={r["cct_rev"]:.0f} meV')
