# Snemi 001
CUDA_VISIBLE_DEVICES=4 python run_job.py --experiment=snemi_test --model=refactored_v7 --no_db --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/refactored_v7_snemi_combos_2019_05_20_18_11_52_689170/model_300.ckpt-300 --placeholders --test --out_dir=snemi_001 --train=snemi_test --val=snemi_test

# Snemi 010
CUDA_VISIBLE_DEVICES=4 python run_job.py --experiment=snemi_test --model=refactored_v7 --no_db --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/refactored_v7_snemi_combos_2019_05_21_11_13_22_025113/model_1250.ckpt-1250 --placeholders --test --out_dir=snemi_010 --train=snemi_test --val=snemi_test

# Snemi 100
CUDA_VISIBLE_DEVICES=4 python run_job.py --experiment=snemi_test --model=refactored_v7 --no_db --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/refactored_v7_snemi_combos_2019_05_20_18_12_42_453730/model_1800.ckpt-1800 --placeholders --test --out_dir=snemi_100 --train=snemi_test --val=snemi_test

#### UNet
####
# Berson 001
CUDA_VISIBLE_DEVICES=4 python run_job.py --experiment=berson_test --model=seung_unet_per_pixel --no_db --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/seung_unet_per_pixel_berson_combos_2019_05_07_16_11_58_465237/model_0.ckpt-0 --placeholders --test --out_dir=seung_berson_001 --train=berson_test --val=berson_test

# Berson 010
CUDA_VISIBLE_DEVICES=4 python run_job.py --experiment=berson_test --model=seung_unet_per_pixel --no_db --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/seung_unet_per_pixel_berson_combos_2019_05_07_16_13_57_624567/model_500.ckpt-500 --placeholders --test --out_dir=seung_berson_010 --train=berson_test --val=berson_test

# Berson 100
CUDA_VISIBLE_DEVICES=4 python run_job.py --experiment=berson_test --model=seung_unet_per_pixel --no_db --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/seung_unet_per_pixel_berson_combos_2019_05_07_16_16_36_988054/model_1000.ckpt-1000 --placeholders --test --out_dir=seung_berson_100 --train=berson_test --val=berson_test


####
# Snemi 001
CUDA_VISIBLE_DEVICES=4 python run_job.py --experiment=snemi_test --model=seung_unet_per_pixel --no_db --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/seung_unet_per_pixel_snemi_combos_2019_05_07_15_36_35_625574/model_0.ckpt-0 --placeholders --test --out_dir=seung_snemi_001 --train=snemi_test --val=snemi_test

# Snemi 010
CUDA_VISIBLE_DEVICES=4 python run_job.py --experiment=snemi_test --model=seung_unet_per_pixel --no_db --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/seung_unet_per_pixel_snemi_combos_2019_05_07_15_38_35_621129/model_1150.ckpt-1150 --placeholders --test --out_dir=seung_snemi_010 --train=snemi_test --val=snemi_test

# Snemi 100
CUDA_VISIBLE_DEVICES=4 python run_job.py --experiment=snemi_test --model=seung_unet_per_pixel --no_db --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/seung_unet_per_pixel_snemi_combos_2019_05_07_15_22_07_598006/model_850.ckpt-850 --placeholders --test --out_dir=seung_snemi_100 --train=snemi_test --val=snemi_test

