#!/usr/bin/env bash
# WESAD DWN Phase 2 sweep driver.
#
# Runs, in order:
#   1. Noise floor  -- the primary config over 8 seeds, so the seed sigma is
#      measured before any two configs are compared. Nothing downstream is
#      trustworthy until this sigma is known: WESAD LOSO noise is ~1 point and a
#      difference smaller than it is not a result.
#   2. Architecture -- all six candidate configs over 5 seeds, ranked on the
#      balanced-accuracy mean with the noise floor as the yardstick.
#   3. Tau sweep     -- the primary config across tau 0.6..3.0 at k=3, to place
#      the GroupSum temperature (the MNIST default 1.6 was tuned on a different
#      task; small models are tau-sensitive).
#   4. z sweep       -- the primary config across thermometer bits 1..8. z sets
#      the input width (159 features x z), which is the pixels register the FPGA
#      pays for regardless of model size (41% of area on MNIST), so this trades
#      accuracy against the single largest hardware block.
#
# The binary secondary (winner re-run on y_binary) is intentionally NOT here:
# k=2 needs an even final layer, and the winner is not known until step 2. Run it
# by hand once the winner is picked, e.g.:
#   python wesad/training/train_dwn.py --task binary --configs 100-52 --seeds 5
#
# Usage:  bash wesad/training/arch_sweep.sh
# Env:    source the OSS/conda envs first (this script activates conda dwn itself).
set -eo pipefail          # no -u: conda's own activate.d scripts reference
                          # unset vars (e.g. NVCC_PREPEND_FLAGS) and die under -u

cd "$(dirname "$0")/../.."        # repo root (script lives in wesad/training/)
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dwn

CACHE="data/wesad_cache/wesad_features_decimated.npz"   # hardware-realistic (sensor-ODR band-limited)
PRIMARY="100-51"
ALL_CONFIGS="100-51,50-51,100-102,200-102,400-201,200-100-51"

echo "########## 1. noise floor: ${PRIMARY} x 8 seeds ##########"
python wesad/training/train_dwn.py --cache "$CACHE" --task multi \
    --configs "$PRIMARY" --seeds 8

echo "########## 2. architecture sweep: 6 configs x 5 seeds ##########"
python wesad/training/train_dwn.py --cache "$CACHE" --task multi \
    --configs "$ALL_CONFIGS" --seeds 5

echo "########## 3. tau sweep: ${PRIMARY} across 0.6..3.0 x 5 seeds ##########"
for TAU in 0.6 0.8 1.0 1.3 1.6 2.0 2.5 3.0; do
    echo "----- tau=${TAU} -----"
    python wesad/training/train_dwn.py --cache "$CACHE" --task multi \
        --configs "$PRIMARY" --tau "$TAU" --seeds 5
done

echo "########## 4. z sweep: ${PRIMARY} across thermometer bits 1..8 x 5 seeds ##########"
for Z in 1 2 3 4 6 8; do
    echo "----- z=${Z} (input width $((159 * Z)) bits) -----"
    python wesad/training/train_dwn.py --cache "$CACHE" --task multi \
        --configs "$PRIMARY" --num-bits "$Z" --seeds 5
done

echo "########## sweep done -- results under results/wesad-dwn-loso_* ##########"
