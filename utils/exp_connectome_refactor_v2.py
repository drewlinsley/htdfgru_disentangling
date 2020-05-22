
# Berson Pretrained
CUDA_VISIBLE_DEVICES=3 python run_job.py --experiment=berson_combos --train=berson_001 --val=berson_001  --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/refactored_v6_star_combos_2019_05_07_22_01_35_756630/model_6800.ckpt-6800 --model=refactored_v6 --no_db

CUDA_VISIBLE_DEVICES=3 python run_job.py --experiment=berson_combos --train=berson_010 --val=berson_010  --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/refactored_v6_star_combos_2019_05_07_22_01_35_756630/model_6800.ckpt-6800 --model=refactored_v6 --no_db

CUDA_VISIBLE_DEVICES=3 python run_job.py --experiment=berson_combos --train=berson_100 --val=berson_100  --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/refactored_v6_star_combos_2019_05_07_22_01_35_756630/model_6800.ckpt-6800 --model=refactored_v6 --no_db

CUDA_VISIBLE_DEVICES=3 python run_job.py --experiment=star_combos --train=berson_100 --val=berson_100  --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/refactored_v6_star_combos_2019_05_07_22_01_35_756630/model_6800.ckpt-6800 --model=refactored_v6 --no_db


# Berson scratch
CUDA_VISIBLE_DEVICES=3 python run_job.py --experiment=berson_combos --train=berson_001 --val=berson_001 --model=refactored_v6 --no_db
CUDA_VISIBLE_DEVICES=3 python run_job.py --experiment=berson_combos --train=berson_010 --val=berson_010 --model=refactored_v6 --no_db
CUDA_VISIBLE_DEVICES=3 python run_job.py --experiment=berson_combos --train=berson_100 --val=berson_100 --model=refactored_v6 --no_db
CUDA_VISIBLE_DEVICES=3 python run_job.py --experiment=star_combos --train=berson_100 --val=berson_100 --model=refactored_v6 --no_db




