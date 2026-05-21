# Delta Model Pipeline

Scripts for training a delta correction head on top of frozen MACE to predict
E_wB97M-V/def2-TZVP − E_wB97X-D3/6-31G(d).

## Data collection

Training data: 500 randomly sampled T1x training reactions, 10 wB97M-V/def2-TZVP
SPs + gradients each (5,000 SPs total). wB97X-D3/6-31G(d) energy and forces read
from T1x HDF5. Geometries chosen by uniform spacing along the T1x NEB trajectory
so that reactant, TS, and product regions are always covered.

Output per geometry: delta energy (E_wB97M − E_wB97X) and delta forces
(F_wB97M − F_wB97X) — both usable as training targets.

Validation data: 225 val reactions (Groups A and B).
- Group B (51 reactions): full delta energy + delta forces available.
- Group A (174 reactions): delta energy only — wB97M-V forces not stored in neb.db.

## TODO

- Benchmark geometry sampling strategy: uniform spacing (current) vs. fully random
  sampling with a fixed seed. Hypothesis: uniform is better for 10 geometries because
  random risks missing the TS region entirely, but worth verifying once the head is trained.
