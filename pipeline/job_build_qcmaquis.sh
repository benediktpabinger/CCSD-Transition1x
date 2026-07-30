#!/bin/bash
#SBATCH --job-name=build_qcmaquis
#SBATCH --partition=xeon24el8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out

set -e

BUILD_DIR=$HOME/software/qcmaquis_build2
SRC_DIR=$HOME/software/qcmaquis_src

echo "=== Loading modules ==="
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CMake/3.26.3-GCCcore-12.3.0
module load Boost/1.82.0-GCC-12.3.0
module load HDF5/1.14.0-gompi-2023a
module load Eigen/3.4.0-GCCcore-12.3.0
module load GSL/2.7-GCC-12.3.0
module load FlexiBLAS/3.3.1-GCC-12.3.0

echo "Python: $(which python3) $(python3 --version)"

# cmake was already configured interactively in $BUILD_DIR with the correct flags.
# Just run the build.
echo "=== Building in $BUILD_DIR ($(nproc) cores) ==="
cd $BUILD_DIR
make -j$(nproc) 2>&1 | tee make_output.txt

echo "=== Build done ==="

# Find the compiled _dmrg Python extension
DMRG_SO=$(find $BUILD_DIR -name '_dmrg*.so' 2>/dev/null | head -1)
if [ -z "$DMRG_SO" ]; then
    echo "ERROR: _dmrg*.so not found after build"
    find $BUILD_DIR -name '*.so' | head -10
    exit 1
fi
echo "Found extension: $DMRG_SO"

# Copy to user site-packages so Python can find it
SITE_PKG=$(python3 -c "import site; print(site.getusersitepackages())")
mkdir -p $SITE_PKG
cp $DMRG_SO $SITE_PKG/
echo "Installed _dmrg to: $SITE_PKG/"

# Install the pure-Python scine_qcmaquis package (no compilation needed)
pip install --user --no-build-isolation --no-binary :all: \
    --config-settings="--build-option=--skip-build" \
    $SRC_DIR 2>&1 | tail -10 || \
pip install --user $SRC_DIR 2>&1 | tail -10

# Verify
python3 -c "import _dmrg; print('_dmrg OK')"
python3 -c "import scine_qcmaquis; print('scine_qcmaquis', scine_qcmaquis.__version__)" || echo "scine_qcmaquis not fully installed yet"

echo "=== Done ==="
