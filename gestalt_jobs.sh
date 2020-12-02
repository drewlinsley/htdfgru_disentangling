#!/usr/bin/env

CUDA_VISIBLE_DEVICES=0 python run_job.py --no_db --experiment=nist_3_ix2v2_50k_frozen --model=BSDS_vgg_cabc --train=cluttered_nist_3_ix2v2_50k --val=cluttered_nist_3_ix2v2_50k
CUDA_VISIBLE_DEVICES=0 python run_job.py --no_db --experiment=pathfinder_14_frozen --model=BSDS_vgg_cabc --train=curv_contour_length_14_50k --val=curv_contour_length_14_50k

CUDA_VISIBLE_DEVICES=2 python run_job.py --no_db --experiment=nist_3_ix2v2_50k_frozen --model=BSDS_vgg_cabc_unfrozen --train=cluttered_nist_3_ix2v2_50k --val=cluttered_nist_3_ix2v2_50k
CUDA_VISIBLE_DEVICES=3 python run_job.py --no_db --experiment=pathfinder_14_frozen --model=BSDS_vgg_cabc_unfrozen --train=curv_contour_length_14_50k --val=curv_contour_length_14_50k

# CUDA_VISIBLE_DEVICES=0 python run_job.py --experiment=nist_3_ix2v2_50k --model=BSDS_vgg_gestalt --ckpt=stannis_weights/htd_fgru_nist_3_ix2v2_50k_2020_05_22_09_00_31_610470/model_22000.ckpt-22000 --no_db --test --out_dir=mytest

