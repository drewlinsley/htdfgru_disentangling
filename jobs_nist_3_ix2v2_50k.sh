CUDA_VISIBLE_DEVICES=6 python run_job.py --experiment=nist_3_ix2v2_50k --model=h_fgru --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/h_fgru_nist_3_ix2v2_50k_2019_07_28_06_26_50_816317/model_4000.ckpt-4000 --no_db --test --out_dir=test_rebut
CUDA_VISIBLE_DEVICES=6 python run_job.py --experiment=nist_3_ix2v2_50k --model=td_fgru --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/td_fgru_nist_3_ix2v2_50k_2019_07_27_00_36_14_815943/model_6000.ckpt-6000 --no_db --test --out_dir=test_rebut
