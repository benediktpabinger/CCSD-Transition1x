"""Smoke test of stability_pipeline.py on a small basis: does the whole chain
run, are eigenvalues captured, does the lambda guard work, is the orbital file
written?  Chemistry is not under test here, only the plumbing."""
import os, sys, json, importlib.util

sys.path.insert(0, '/home/energy/s242862/pipeline')
spec = importlib.util.spec_from_file_location(
    'sp', '/home/energy/s242862/pipeline/stability_pipeline.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# shrink to a cheap basis; everything else stays as in production
m.BASIS = 'def2-svp'
out = '/home/energy/s242862/stab_pipeline/_test'
os.makedirs(out, exist_ok=True)

print('=== lambda guard ===')
print('  normal  :', m.lam({'ext': [-0.048, 0.1]}, 'ext'))
print('  breakdown:', m.lam({'ext': [-3004.3, -770.0]}, 'ext'))
print('  missing :', m.lam({}, 'ext'))

print('\n=== full chain on rxn1320 / RKS-ref (def2-SVP) ===')
rec = m.run_geometry('rxn1320', 'RKS-ref',
                     '/home/energy/s242862/orca_neb_results/rxn1320/transition_state.xyz',
                     out, 20000)
print()
print(json.dumps({k: v for k, v in rec.items() if k != 'xyz'}, indent=1)[:1400])

print('\n=== Pruefungen ===')
ok = True
def chk(name, cond):
    global ok
    print(f'  {"OK  " if cond else "FAIL"}  {name}')
    ok = ok and cond
chk('RKS konvergiert',        rec.get('rks_converged') is True)
chk('RKS-Gradient da',        bool(rec.get('rks_grad')))
chk('lmin_int erfasst',       rec.get('lmin_int') is not None)
chk('lmin_ext erfasst',       rec.get('lmin_ext') is not None)
bs = rec.get('bs') or {}
chk('BS-Loesung gefunden',    bool(bs) and 'invalid' not in bs)
chk('BS dE < 0',              (bs.get('de_meV') or 1) < 0)
chk('BS S2 > 0.05',           (bs.get('s2') or 0) > 0.05)
chk('BS-Gradient da',         bool(bs.get('bs_grad')))
chk('UKS lmin_int erfasst',   bs.get('uks_lmin_int') is not None)
chk('UKS lmin_ext erfasst',   bs.get('uks_lmin_ext') is not None)
chk('Orbitale gespeichert',   os.path.exists(f'{out}/bs_RKS_ref.npz'))
print('\n' + ('ALLES OK' if ok else 'FEHLER - nicht ausrollen'))
