#!/bin/bash
# Run once on the cluster to install fairchem and download the eSEN checkpoint.
# Usage: bash setup_esen.sh

set -e

echo "=== Installing fairchem ==="
pip install fairchem-core --quiet

echo "=== Downloading eSEN checkpoint ==="
mkdir -p ~/checkpoints
python3 -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='facebook/OMol25',
    filename='esen_sm_conserving_all.pt',
    local_dir='/home/energy/s242862/checkpoints',
)
print(f'Downloaded to: {path}')
"

echo "=== Verifying ==="
python3 -c "
from fairchem.core import OCPCalculator
calc = OCPCalculator(
    checkpoint_path='/home/energy/s242862/checkpoints/esen_sm_conserving_all.pt',
    cpu=False,
)
print('eSEN calculator loaded successfully.')
"

echo "Setup complete."
