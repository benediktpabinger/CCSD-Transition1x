#!/bin/bash
#SBATCH --job-name=mace_smoke
#SBATCH --partition=xeon24el8
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/energy/s242862/logs/mace_smoke_%j.log

# Smoke test: convert 100 reactions to extxyz, check output is valid.
# No GPU needed.

set -e

H5=/home/energy/s242862/data/Transition1x.h5
OUT=/home/energy/s242862/data/smoke_test.xyz

module load Python/3.13.5-GCCcore-14.3.0

pip install mace-torch --quiet --user

echo "Converting 100 reactions..."
python /home/energy/s242862/pipeline/convert_h5_to_extxyz.py \
    --h5file ${H5} \
    --output ${OUT} \
    --split train \
    --max-reactions 100 \
    --overwrite

echo "Checking output..."
python -c "
from ase.io import read
configs = read('${OUT}', index=':')
print(f'Read {len(configs)} configs')
print(f'First config: {configs[0].get_chemical_formula()}, energy={configs[0].info[\"energy\"]:.4f} eV')
print(f'Forces shape: {configs[0].arrays[\"forces\"].shape}')
print('OK')
"
