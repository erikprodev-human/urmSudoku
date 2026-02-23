#!/bin/bash
set -e
echo "URM Sudoku Training v3"
python ./2_train_URM_Sudoku.py \
    --lr_from 1e-4 --lr_to 4e-5 --steps 120000 \
    --attn_dropout 0.1 --mlp_dropout 0.1 \
    --loop_noise_std 0.02 --constraint_loss_weight 0.0 \
    --tta_augments 8 --grad_clip 1.0
echo "Training Complete!"