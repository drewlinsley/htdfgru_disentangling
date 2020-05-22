

## Instance
# SNEMI Pretrained
cd /media/data_cifs/cluster_projects/refactor_gammanet
CUDA_VISIBLE_DEVICES=2 python run_job.py --experiment=snemi_combos --train=snemi_001 --val=snemi_001  --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/seung_unet_per_pixel_instance_star_combos_2019_05_06_10_50_53_951630/model_12750.ckpt-12750 --model=seung_unet_per_pixel_instance --no_db

CUDA_VISIBLE_DEVICES=2 python run_job.py --experiment=snemi_combos --train=snemi_010 --val=snemi_010  --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/seung_unet_per_pixel_instance_star_combos_2019_05_06_10_50_53_951630/model_12750.ckpt-12750 --model=seung_unet_per_pixel_instance --no_db

CUDA_VISIBLE_DEVICES=2 python run_job.py --experiment=snemi_combos --train=snemi_100 --val=snemi_100  --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/seung_unet_per_pixel_instance_star_combos_2019_05_06_10_50_53_951630/model_12750.ckpt-12750 --model=seung_unet_per_pixel_instance --no_db

CUDA_VISIBLE_DEVICES=2 python run_job.py --experiment=star_combos --train=snemi_100 --val=snemi_100  --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/seung_unet_per_pixel_instance_star_combos_2019_05_06_10_50_53_951630/model_12750.ckpt-12750 --model=seung_unet_per_pixel_instance --no_db

# SNEMI scratch
CUDA_VISIBLE_DEVICES=2 python run_job.py --experiment=snemi_combos --train=snemi_001 --val=snemi_001 --model=seung_unet_per_pixel_instance --no_db
CUDA_VISIBLE_DEVICES=2 python run_job.py --experiment=snemi_combos --train=snemi_010 --val=snemi_010 --model=seung_unet_per_pixel_instance --no_db
CUDA_VISIBLE_DEVICES=2 python run_job.py --experiment=snemi_combos --train=snemi_100 --val=snemi_100 --model=seung_unet_per_pixel_instance --no_db
CUDA_VISIBLE_DEVICES=2 python run_job.py --experiment=star_combos --train=snemi_100 --val=snemi_100 --model=seung_unet_per_pixel_instance --no_db

# Berson Pretrained
CUDA_VISIBLE_DEVICES=2 python run_job.py --experiment=berson_combos --train=berson_001 --val=berson_001  --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/seung_unet_per_pixel_instance_star_combos_2019_05_06_10_50_53_951630/model_12750.ckpt-12750 --model=seung_unet_per_pixel_instance --no_db

CUDA_VISIBLE_DEVICES=2 python run_job.py --experiment=berson_combos --train=berson_010 --val=berson_010  --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/seung_unet_per_pixel_instance_star_combos_2019_05_06_10_50_53_951630/model_12750.ckpt-12750 --model=seung_unet_per_pixel_instance --no_db

CUDA_VISIBLE_DEVICES=2 python run_job.py --experiment=berson_combos --train=berson_100 --val=berson_100  --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/seung_unet_per_pixel_instance_star_combos_2019_05_06_10_50_53_951630/model_12750.ckpt-12750 --model=seung_unet_per_pixel_instance --no_db

CUDA_VISIBLE_DEVICES=2 python run_job.py --experiment=star_combos --train=berson_100 --val=berson_100  --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/seung_unet_per_pixel_instance_star_combos_2019_05_06_10_50_53_951630/model_12750.ckpt-12750 --model=seung_unet_per_pixel_instance --no_db


# Berson scratch
CUDA_VISIBLE_DEVICES=2 python run_job.py --experiment=berson_combos --train=berson_001 --val=berson_001 --model=seung_unet_per_pixel_instance --no_db
CUDA_VISIBLE_DEVICES=2 python run_job.py --experiment=berson_combos --train=berson_010 --val=berson_010 --model=seung_unet_per_pixel_instance --no_db
CUDA_VISIBLE_DEVICES=2 python run_job.py --experiment=berson_combos --train=berson_100 --val=berson_100 --model=seung_unet_per_pixel_instance --no_db
CUDA_VISIBLE_DEVICES=2 python run_job.py --experiment=star_combos --train=berson_100 --val=berson_100 --model=seung_unet_per_pixel_instance --no_db




