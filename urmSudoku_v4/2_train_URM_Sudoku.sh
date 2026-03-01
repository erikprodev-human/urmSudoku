#!/bin/bash
set -e
echo "URM Sudoku Training v6"
python train_URM_Universal.py --data_path data/sudoku --lr_from 1e-4 --lr_to 4e-5
echo "Training Complete!"