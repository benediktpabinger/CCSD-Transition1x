"""Is Route 1 on rxn8837/UMA-S bistable?  Re-run the identical calculation and
report which branch the Newton solver lands on, together with the node."""
import os, sys, json, socket, importlib.util

sys.path.insert(0, '/home/energy/s242862/pipeline')
spec = importlib.util.spec_from_file_location(
    'sp', '/home/energy/s242862/pipeline/stability_pipeline.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

XYZ = '/home/energy/s242862/uma_neb_results/rxn8837/transition_state.xyz'
out = f'/home/energy/s242862/repro8837/{os.environ.get("SLURM_ARRAY_TASK_ID","x")}'
os.makedirs(out, exist_ok=True)

rec = m.run_geometry('rxn8837', 'UMA-S', XYZ, out, 50000)
bs = rec.get('bs') or {}
res = {'node': socket.gethostname(),
       'task': os.environ.get('SLURM_ARRAY_TASK_ID'),
       'e_rks': rec.get('e_rks'), 'lmin_ext': rec.get('lmin_ext'),
       'e_uks': bs.get('e_uks'), 'de_meV': bs.get('de_meV'),
       's2': bs.get('s2'), 'route': bs.get('route'),
       'uks_int_stable': bs.get('uks_int_stable')}
print('RESULT ' + json.dumps(res), flush=True)
json.dump(res, open(f'{out}/repro.json', 'w'), indent=1)
