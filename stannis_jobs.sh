#!/usr/bin/env

CUDA_VISIBLE_DEVICES=0 python run_job.py --no_db --experiment=nist_3_ix2v2_50k --model=htd_fgru --train=cluttered_nist_3_ix2v2_50k --val=cluttered_nist_3_ix2v2_50k
CUDA_VISIBLE_DEVICES=0 python run_job.py --experiment=nist_3_ix2v2_50k --model=htd_fgru --ckpt=stannis_weights/htd_fgru_nist_3_ix2v2_50k_2020_05_22_09_00_31_610470/model_22000.ckpt-22000 --no_db --test --out_dir=mytest

