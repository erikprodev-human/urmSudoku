#!/bin/bash
set -e
echo "Phase 1: Linear LR Decay (0 -> 100k steps, LR: 1e-4 -> 3e-5)"
python ./2_train_URM_Sudoku.py --lr_from 1e-4 --lr_to 3e-5 --steps 100000
echo "Training Complete!"