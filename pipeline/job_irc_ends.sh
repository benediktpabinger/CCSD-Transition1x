#!/bin/bash
#SBATCH --job-name=ircends
#SBATCH --partition=xeon24el9
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --mem=180G
#SBATCH --output=/home/energy/bepab/logs/ircends_%j.out
#SBATCH --error=/home/energy/bepab/logs/ircends_%j.err

# DFT minimisation of both IRC endpoints of one pilot structure (D32), side by side with
# 12 cores each: UKS + STABPerform at the endpoint, then UKS Opt from those orbitals.
#   sbatch --export=ALL,DIR=$HOME/marks_pilot/rxn7035 pipeline/job_irc_ends.sh

source /etc/profile
source $HOME/opt/orca6.env
ORCA=$ORCADIR/orca
export OMPI_MCA_hwloc_base_binding_policy=none OMPI_MCA_rmaps_base_oversubscribe=true
METHOD='wB97M-V def2-TZVPD def2/J RIJCOSX TightSCF DEFGRID3'
HEAD='%pal nprocs 12 end
%maxcore 3500'
SCF='  Thresh 1e-12
  TCut   1e-13
  MaxIter 300'
DIR=${DIR:?DIR not set}

end_opt () {                        # $1 = F or B
  local W=$DIR/end_$1
  mkdir -p $W && cd $W && cp $DIR/irc_IRC_$1.xyz start.xyz || return 2
  grep -q "ORCA TERMINATED NORMALLY" sp.out 2>/dev/null || {
    printf '! UKS %s\n%s\n%%scf\n%s\n  STABPerform true\n  STABRestartUHFifUnstable true\nend\n* xyzfile 0 1 start.xyz\n' "$METHOD" "$HEAD" "$SCF" > sp.inp
    $ORCA sp.inp > sp.out 2>&1; }
  grep -q "ORCA TERMINATED NORMALLY" opt.out 2>/dev/null || {
    cp sp.gbw opt_start.gbw
    printf '! UKS %s Opt MORead\n%%moinp "opt_start.gbw"\n%s\n%%scf\n%s\nend\n%%geom MaxIter 150 end\n* xyzfile 0 1 start.xyz\n' "$METHOD" "$HEAD" "$SCF" > opt.inp
    $ORCA opt.inp > opt.out 2>&1; }
  find . -maxdepth 1 -type f \( -name '*.tmp' -o -name '*.densities*' -o -name '*_start.gbw' -o -name '*.ges' -o -name '*.bibtex' -o -name '*_property.txt' \) -delete
  echo "end $1: $(grep -c 'THE OPTIMIZATION HAS CONVERGED' opt.out) converged $(date)"
}
( end_opt F ) & ( end_opt B ) & wait
