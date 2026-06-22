import sqlite3, numpy as np, json

base = '/home/energy/s242862/mace_delta_neb_results/rxn4063'
con = sqlite3.connect(base + '/neb.db')
all_e = [r[0] for r in con.execute('SELECT energy FROM systems ORDER BY id').fetchall()]
con.close()

e = np.array(all_e[-10:])
rel = (e - e[0]) * 1000
fwd = rel.max()
rev = fwd - rel[-1]

with open(base + '/fmaxs.json') as f:
    fmaxs = json.load(f)

print('MACE+delta NEB:  fwd=%.1f  rev=%.1f  fmax=%.4f' % (fwd, rev, fmaxs[-1]))

with open('/home/energy/s242862/delta_head/full_benchmark_results.json') as f:
    bm = {r['rxn']: r for r in json.load(f)['reactions']}

r = bm['rxn4063']
print('wB97M-V (ORCA):  fwd=%.1f' % r['neb_wb97m_fwd_meV'])
print('CCSD(T):         fwd=%.1f' % r['ccsdt_fwd_meV'])
print('eSEN NEB:        fwd=%.1f' % r['esen_neb_fwd_meV'])
print('UMA-s NEB:       fwd=%.1f' % r['uma_neb_fwd_meV'])
print('MACE (SP):       fwd=%.1f' % r['mace_fwd_meV'])
print('MACE+delta (SP): fwd=%.1f' % r['delta_fwd_meV'])
