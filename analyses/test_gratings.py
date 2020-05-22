CUDA_VISIBLE_DEVICES=5 python run_job.py --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_gratings_simple_gratings_2019_05_20_12_13_01_973409/model_1900.ckpt-1900 --model=BSDS_vgg_gratings_simple --experiment=gratings_test --test --placeholders --out_dir=gratings --no_db


CUDA_VISIBLE_DEVICES=2 python run_job.py --experiment=BSDS500_test --model=BSDS_vgg_cheap_deepest_final_simple_per_timestep --no_db --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_cheap_deepest_final_simple_BSDS500_combos_100_no_aux_augs_2019_05_20_18_21_17_393001/model_2520.ckpt-2520 --placeholders --test --out_dir=bsds_landscape --train=BSDS500_test_landscape --val=BSDS500_test_landscape


CUDA_VISIBLE_DEVICES=1 python run_job.py --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_gratings_simple_gratings_2019_05_20_12_13_01_973409/model_1900.ckpt-1900 --model=BSDS_vgg_gratings_simple --train=BSDS500_test_landscape --val=BSDS500_test_landscape --experiment=gratings_test --test --placeholders --out_dir=gratings --no_db


CUDA_VISIBLE_DEVICES=5 python run_job.py --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_gratings_simple_gratings_2019_05_20_12_13_01_973409/model_1900.ckpt-1900 --model=BSDS_vgg_gratings_simple --train=BSDS500_test_landscape --val=BSDS500_test_landscape --experiment=gratings_test_bsds --test --placeholders --out_dir=gratings --no_db

