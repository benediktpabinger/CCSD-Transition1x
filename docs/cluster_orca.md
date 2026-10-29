# ORCA on Niflheim

Two versions exist. Which one a job uses is decided by the partition.

| | ORCA 5.0.4 (thesis) | ORCA 6.0.0 (paper) |
|---|---|---|
| where | central module, el8 nodes only | private install, `~/opt/orca_6.0.0`, el9 nodes |
| partition | `xeon24el8`, `xeon40el8` | `xeon24el9` (and other el9 partitions) |
| activate in a job | `source /etc/profile; module load gompi/2023a ORCA/5.0.4-gompi-2023a` | `source $HOME/opt/orca6.env` |
| binary | `$(which orca)` | `$ORCADIR/orca` (full path, required for parallel runs) |
| MPI | OpenMPI 4.1.5 from gompi/2023a | private OpenMPI 4.1.6 in `~/opt/openmpi-4.1.6` (system gcc, no module) |
| status | every thesis number | installed 2026-10-29; acceptance test = 44-reaction label comparison, see `docs/paper_decisions.md` |

Why 6.0.0 and not newer: OMol25's labels are ORCA 6.0.0, and every ORCA 6.x forum build needs OpenMPI 4.1,
which no el9 tree has, so a private MPI was needed either way. Exact version match costs nothing extra.

## Job skeleton (el9, whole node)

```bash
#SBATCH --partition=xeon24el9
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --mem=120G
source $HOME/opt/orca6.env
$ORCADIR/orca input.inp > input.out 2>&1
```
with `%pal nprocs 24 end` and `%maxcore 4500` in the input. Memory per node is 250 GB on xeon24el9.

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
