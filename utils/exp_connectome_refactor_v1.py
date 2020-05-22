

## Instance
# SNEMI Pretrained
cd /media/data_cifs/cluster_projects/refactor_gammanet
CUDA_VISIBLE_DEVICES=2 python run_job.py --experiment=snemi_combos --train=snemi_001 --val=snemi_001  --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/refactored_v6_star_combos_2019_05_07_22_01_35_756630/model_6800.ckpt-6800 --model=refactored_v6 --no_db

CUDA_VISIBLE_DEVICES=2 python run_job.py --experiment=snemi_combos --train=snemi_010 --val=snemi_010  --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/refactored_v6_star_combos_2019_05_07_22_01_35_756630/model_6800.ckpt-6800 --model=refactored_v6 --no_db

CUDA_VISIBLE_DEVICES=2 python run_job.py --experiment=snemi_combos --train=snemi_100 --val=snemi_100  --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/refactored_v6_star_combos_2019_05_07_22_01_35_756630/model_6800.ckpt-6800 --model=refactored_v6 --no_db

CUDA_VISIBLE_DEVICES=2 python run_job.py --experiment=star_combos --train=snemi_100 --val=snemi_100  --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/refactored_v6_star_combos_2019_05_07_22_01_35_756630/model_6800.ckpt-6800 --model=refactored_v6 --no_db

# SNEMI scratch
CUDA_VISIBLE_DEVICES=2 python run_job.py --experiment=snemi_combos --train=snemi_001 --val=snemi_001 --model=refactored_v6 --no_db
CUDA_VISIBLE_DEVICES=2 python run_job.py --experiment=snemi_combos --train=snemi_010 --val=snemi_010 --model=refactored_v6 --no_db
CUDA_VISIBLE_DEVICES=2 python run_job.py --experiment=snemi_combos --train=snemi_100 --val=snemi_100 --model=refactored_v6 --no_db
CUDA_VISIBLE_DEVICES=2 python run_job.py --experiment=star_combos --train=snemi_100 --val=snemi_100 --model=refactored_v6 --no_db

