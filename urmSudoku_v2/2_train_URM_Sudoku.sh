#!/bin/bash
set -e
echo "Phase 1: Linear LR Decay (0 -> 120k steps, LR: 1e-4 -> 4e-5)"
python ./2_train_URM_Sudoku.py --lr_from 1e-4 --lr_to 4e-5 --steps 120000
echo "Training Complete!"