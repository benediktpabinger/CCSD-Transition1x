# ORCA on Niflheim

Two versions exist. Which one a job uses is decided by the partition.

| | ORCA 5.0.4 (thesis) | ORCA 6.0.1 (paper) |
|---|---|---|
| where | central module, el8 nodes only | private install, `~/opt/orca_6.0.1`, el9 nodes |
| partition | `xeon24el8`, `xeon40el8` | `xeon24el9` (and other el9 partitions) |
| activate in a job | `source /etc/profile; module load gompi/2023a ORCA/5.0.4-gompi-2023a` | `source $HOME/opt/orca6.env` |
| binary | `$(which orca)` | `$ORCADIR/orca` (full path, required for parallel runs) |
| MPI | OpenMPI 4.1.5 from gompi/2023a | private OpenMPI 4.1.6 in `~/opt/openmpi-4.1.6` (system gcc 11.5, no module) |
| status | every thesis number; the border zone and the zero sample (2026-10-28/29) | installed 2026-10-29; acceptance test ("gate") at the 45 stored TS, `pipeline/orca6_gate.py` -> `results/orca6_gate.csv`, decision D16 in `docs/paper_decisions.md` |

Why ORCA 6 at all: OMol25's labels are ORCA 6.0.0, and after the el8 reinstall no central ORCA exists on el9 (every
6.x forum build needs OpenMPI 4.1, el9 only has 5.0, so a private MPI was needed either way).

**Why 6.0.1 and not 6.0.0, the exact OMol25 version.** 6.0.0 was installed first (`~/opt/orca_6.0.0`, kept for the
record, do not use). With our protocol (UKS wB97M-V/def2-TZVPD, `STABPerform` + `STABRestartUHFifUnstable`) its SCF
energies matched 5.0.4 to 1e-4 meV, but the stability analysis reported spurious large negative eigenvalues (about
-0.06 Eh for states 5.0.4 calls stable), restarted five times into the same state and aborted: 9 of 17 reactions failed
(run archived in `~/orca600_hinge_t1x`). The 6.0.1 changelog lists "wrong orbitals for non-self-consistent DFT-NL
calculations (wB97M-V, ...)" and two stability-analysis fixes. OMol25 itself was not affected: its route is the 20-degree
rotation, no stability analysis. 6.0.1 is a bug-fix release of the same code, so it is the closest version that runs our
protocol.

## Job skeleton (el9, whole node)

```bash
#SBATCH --partition=xeon24el9
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --mem=120G
source $HOME/opt/orca6.env
$ORCADIR/orca input.inp > input.out 2>&1
```
with `%pal nprocs 24 end` and `%maxcore 4500` in the input. Memory per node is 250 GB on xeon24el9. For the small
Transition1x molecules two ORCA runs per node with `nprocs 12` each are more efficient than one with 24
(`pipeline/job_border_zone.sh` does this; it needs `OMPI_MCA_hwloc_base_binding_policy=none` so the two `mpirun`s do
not bind to the same cores).

## OMol25 protocol (unchanged from the thesis)

`! UKS wB97M-V def2-TZVPD def2/J RIJCOSX TightSCF DEFGRID3`, `%scf Thresh 1e-12  TCut 1e-13  MaxIter 300`,
stability analysis `STABPerform true  STABRestartUHFifUnstable true`. Forces: `EnGrad` on the converged orbitals (`MORead`).
Template: `pipeline/job_hinge_t1x.sh`; its ORCA 6 variant lives in `~/orca6_hinge_t1x/job_hinge_t1x_orca6.sh`.

## Interactive use on the login node

`orca6` (function in `~/.bashrc`) puts ORCA 6 on PATH for that shell. Only for `orca -h` style checks:
the login nodes kill any process after 20 min of CPU time.

## Rebuilding OpenMPI

`~/build/job_build_openmpi416.sh`: configure flags `--with-slurm --without-verbs --without-ofi --without-ucx
--disable-mpi-fortran --with-pmix=internal --with-libevent=internal --with-hwloc=internal`, system gcc 11.5.
Submit to `xeon24el9_test` (10 min limit, usually enough) or `xeon24el9`.
