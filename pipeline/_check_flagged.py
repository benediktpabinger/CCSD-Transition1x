import json, numpy as np

with open('full_benchmark_results.json') as f:
    rxns = json.load(f)['reactions']

TOP10    = {'rxn7949','rxn8832','rxn1320','rxn4113','rxn8885','rxn7945','rxn7937','rxn6196','rxn0346','rxn1150'}
MIDDLE10 = {'rxn0896','rxn1154','rxn5690','rxn4513','rxn7955','rxn4519','rxn4500','rxn2553','rxn8829','rxn1155'}
BOTTOM10 = {'rxn9246','rxn4498','rxn1061','rxn4003','rxn4004','rxn4063','rxn4114','rxn4060','rxn1961','rxn1962'}

HIGH_FMAX = {'rxn8885', 'rxn6196', 'rxn0346', 'rxn4004'}

def mae(a):  return np.abs(a).mean()
def bias(a): return a.mean()

print('eSEN vs wB97M-V -- flagged reactions (fmax > 0.10)')
print()

for label, group in [('High MR', TOP10), ('Mid MR', MIDDLE10), ('Low MR', BOTTOM10)]:
    grxns_all  = [r for r in rxns if r['rxn'] in group
                  and r.get('esen_neb_fwd_meV') and r.get('neb_wb97m_fwd_meV')]
    grxns_good = [r for r in grxns_all if r['rxn'] not in HIGH_FMAX]
    flagged    = [r for r in grxns_all if r['rxn'] in HIGH_FMAX]

    errs_all  = np.array([r['esen_neb_fwd_meV'] - r['neb_wb97m_fwd_meV'] for r in grxns_all])
    errs_good = np.array([r['esen_neb_fwd_meV'] - r['neb_wb97m_fwd_meV'] for r in grxns_good])

    print(f'{label}:')
    print(f'  All (n={len(grxns_all)}):             MAE={mae(errs_all):.1f}  Bias={bias(errs_all):+.1f} meV')
    if len(errs_good):
        print(f'  Excl. flagged (n={len(grxns_good)}):  MAE={mae(errs_good):.1f}  Bias={bias(errs_good):+.1f} meV')
    for r in flagged:
        diff = r['esen_neb_fwd_meV'] - r['neb_wb97m_fwd_meV']
        print(f'  {r["rxn"]}: eSEN={r["esen_neb_fwd_meV"]:.0f}  wB97M={r["neb_wb97m_fwd_meV"]:.0f}  diff={diff:+.0f}  fmax={r["esen_neb_fmax"]:.3f}')
    print()
