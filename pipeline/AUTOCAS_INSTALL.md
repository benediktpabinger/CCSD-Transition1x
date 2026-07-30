# AutoCAS Installation on DTU SLURM Cluster

## What is installed

| Package | Version | Source |
|---------|---------|--------|
| scine-autocas | 3.0.0 | pip (already installed under Python/3.11.3-GCCcore-12.3.0) |
| scine_qcmaquis | 4.0.0 | compiled from source: https://github.com/qcscine/qcmaquis (tag v4.0.0) |

Both are from the Reiher group (ETH Zurich). scine-autocas implements the AutoCAS
algorithm (Stein & Reiher, J. Chem. Theory Comput. 2016, 12, 1760–1771). scine_qcmaquis
provides the DMRG backend.

## Modules required at runtime

```bash
module load Python/3.11.3-GCCcore-12.3.0
module load Boost/1.82.0-GCC-12.3.0
module load GSL/2.7-GCC-12.3.0
module load FlexiBLAS/3.3.1-GCC-12.3.0
module load HDF5/1.14.0-gompi-2023a
```

## Installation steps (already done — for reference)

### 1. Clone qcmaquis source

```bash
git clone --depth 1 https://github.com/qcscine/qcmaquis.git ~/software/qcmaquis_src
# HEAD = tag v4.0.0, commit 59635575f69eabb14cb04c1492226a5fa6c4bf41
```

### 2. Patch the ALPS cmake check

The HDF5 module on the cluster is the MPI-parallel build (gompi toolchain). qcmaquis's
internal ALPS library raises a fatal error if parallel HDF5 is detected without MPI being
explicitly enabled. ALPS does not actually use MPI — the error is overly strict. Fixed by
removing the fatal error check:

```python
# ~/software/qcmaquis_src/src/alps/CMakeLists.txt
# Removed lines 52–56: if(NOT MPI_FOUND) ... message(FATAL_ERROR ...) ... endif()
```

### 3. Configure with cmake

```bash
module purge
module load Python/3.11.3-GCCcore-12.3.0 CMake/3.26.3-GCCcore-12.3.0 \
    Boost/1.82.0-GCC-12.3.0 HDF5/1.14.0-gompi-2023a \
    Eigen/3.4.0-GCCcore-12.3.0 GSL/2.7-GCC-12.3.0 FlexiBLAS/3.3.1-GCC-12.3.0

mkdir -p ~/software/qcmaquis_build2 && cd ~/software/qcmaquis_build2

cmake ~/software/qcmaquis_src \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SYMMETRIES="TwoU1;TwoU1PG;SU2U1;SU2U1PG" \
    -DPYTHON_BINDINGS=ON \
    -DPYTHON_EXECUTABLE=$(which python3) \
    -DENABLE_OMP=ON \
    -DBUILD_DMRG_EVOLVE=ON \
    -DBUILD_TRANSCORRELATED_DMRG=ON
```

### 4. Compile (SLURM job `10620246`, ~40 min on 16 cores, partition xeon24el8)

```bash
cd ~/software/qcmaquis_build2
make -j16
```

### 5. Install Python package

```bash
# Copy compiled extension to user site-packages
cp ~/software/qcmaquis_build2/src/_dmrg.cpython-311-x86_64-linux-gnu.so \
   ~/.local/lib/python3.11/site-packages/

# Install pure-Python package
pip install --user ~/software/qcmaquis_src

# Copy Python source package (pip missed this)
cp -r ~/software/qcmaquis_src/src/python/scine_qcmaquis \
   ~/.local/lib/python3.11/site-packages/
```

## Compatibility patch (already applied — for reference)

scine-autocas 3.0.0 and scine_qcmaquis 4.0.0 were released 15 days apart (Sep 11 and
Sep 26, 2025) and are designed to work together. However, scine-autocas calls
`pyscf_interface.QcMaquis(mol)` while scine_qcmaquis 4.0.0 renamed this class to
`DMRGSolver`. The class `QCMaquis` in 4.0.0 is a different, lower-level object.

**Fix**: added one alias to the installed package:

```python
# ~/.local/lib/python3.11/site-packages/scine_qcmaquis/pyscf_interface/pyscf_interface.py
# (last line)
QcMaquis = DMRGSolver  # scine-autocas 3.0.0 compatibility
```

`DMRGSolver` is the correct pyscf-compatible FCI solver class in scine_qcmaquis 4.0.0.
The patch does not change the algorithm or computation in any way.

## API usage fix: reading results from workflow.run()

`ClassicWorkflow.run()` (in `scine_autocas/workflows/workflow.py`) has **no return statement**.
It stores all results in `workflow.results` dict and returns `None` implicitly. This means
code like the following is always wrong:

```python
result = workflow.run()
if result is None:
    print("no active space")  # ALWAYS executes — bug!
```

The correct way to read the AutoCAS result after calling `workflow.run()`:

```python
workflow.run()  # results stored in workflow.results, not returned

final_occ = workflow.results.get("final_occupation")   # List[int] or None
final_idx = workflow.results.get("final_orbital_indices")  # List[int] or None

if final_occ is None or final_idx is None:
    # SingleReferenceException was raised internally — no active space needed
    print("AutoCAS: no active space selected")
else:
    cas_electrons = sum(int(n) for n in final_occ)
    cas_orbitals  = len(final_occ)
    print(f"AutoCAS result: CAS({cas_electrons},{cas_orbitals})")
    print(f"Active orbital indices (0-based): {list(final_idx)}")
```

`final_occupation` is `None` only when `diagnostics.is_single_reference(s1)` returns True
(all s1 entropies below threshold), in which case `ClassicWorkflow` catches the
`SingleReferenceException` internally and sets `final_orbital_indices = None`.

## Verification

```bash
module load Python/3.11.3-GCCcore-12.3.0 Boost/1.82.0-GCC-12.3.0 \
    GSL/2.7-GCC-12.3.0 FlexiBLAS/3.3.1-GCC-12.3.0 HDF5/1.14.0-gompi-2023a

python3 -c "
import _dmrg; print('_dmrg OK')
import scine_qcmaquis; print('scine_qcmaquis OK')
from scine_autocas import Molecule, Autocas
from scine_autocas.interfaces.pyscf.pyscf import PyscfInterface
from scine_autocas.workflows.conventional import ClassicWorkflow
print('All imports OK')
"
```

Tested end-to-end on H2 (STO-3G): AutoCAS correctly identifies no active space needed
(s1 = 0.068 for both orbitals, below threshold — physically correct for H2 at equilibrium).

## Scaling limits — benchmark on MR reactions (Transition1x)

AutoCAS was tested on the 23 MR benchmark reactions (5–14 atoms, def2-SVP basis).
The valence CAS constructed automatically by `ClassicWorkflow` is very large for these
molecules: rxn7949 (rank-1 MR, NFOD=1.146) required CAS(36,33).

| Reaction | Basis   | Bond dim (m) | Wall time | Result                  |
|----------|---------|-------------|-----------|-------------------------|
| rxn10005 | STO-3G  | 250         | ~10 min   | CAS(2,4) — wrong (basis too minimal) |
| rxn7949  | def2-SVP | 500        | 8 h (timeout) | no sweeps completed |

**Conclusion: AutoCAS is not viable for these molecules with def2-SVP.**

The initial DMRG calculation on CAS(36,33) at m=500 did not complete a single sweep
in 8 hours on an uncontested 8-core xeon24el8 node. The QCMaquis workdir remained
empty throughout — no checkpoint files were written.

Root cause: DMRG cost scales as O(m³ · L · sweeps). At m=500 and L=33 orbitals,
one sweep takes more than 8 hours. The AutoCAS papers (Stein & Reiher 2016) used
CAS spaces up to roughly (20,20) with m=250–500; CAS(36,33) is significantly larger.

STO-3G (minimal basis) is fast enough but gives wrong orbital character — AutoCAS
found CAS(2,4) with occupation [2,0,0,0] for rxn10005, missing all π/π* correlation
that AVAS correctly identifies as CAS(20,13) with TS occupations 1.710/0.313.

**Active space selection for this benchmark uses AVAS instead** (see `mr_casscf_optts.py`).
AVAS (Sayfutyarova et al., JCTC 2017) projects HF MOs onto chemically chosen AO types
(C 2pz, N 2p, O 2pz) by overlap threshold, runs in minutes, and produces physically
correct active spaces validated against FOD diagnostics and NEVPT2 energies.

## Citation

> Stein, C. J.; Reiher, M. Automated Selection of Active Orbital Spaces. *J. Chem. Theory
> Comput.* **2016**, *12*, 1760–1771. https://doi.org/10.1021/acs.jctc.6b00156
