#!/usr/bin/env bash
# Autogen script for running models

CUDA_VISIBLE_DEVICES=5 python run_job.py --experiment=BSDS500_test_orientation_viz --model=BSDS_fixed_perturb --no_db --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_cheap_deepest_final_simple_BSDS500_combos_100_no_aux_2019_05_20_00_56_41_386546/model_1240.ckpt-1240 --placeholders --out_dir=orientation_probe_contrast_12 --train=circuit_contrast_circuit_exc_12_1 --val=circuit_contrast_circuit_exc_12_1
CUDA_VISIBLE_DEVICES=5 python run_job.py --experiment=BSDS500_test_orientation_viz --model=BSDS_fixed_perturb --no_db --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_cheap_deepest_final_simple_BSDS500_combos_100_no_aux_2019_05_20_00_56_41_386546/model_1240.ckpt-1240 --placeholders --out_dir=orientation_probe_contrast_12 --train=circuit_contrast_circuit_exc_12_31 --val=circuit_contrast_circuit_exc_12_31
CUDA_VISIBLE_DEVICES=5 python run_job.py --experiment=BSDS500_test_orientation_viz --model=BSDS_fixed_perturb --no_db --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_cheap_deepest_final_simple_BSDS500_combos_100_no_aux_2019_05_20_00_56_41_386546/model_1240.ckpt-1240 --placeholders --out_dir=orientation_probe_contrast_12 --train=circuit_contrast_circuit_exc_12_61 --val=circuit_contrast_circuit_exc_12_61
CUDA_VISIBLE_DEVICES=5 python run_job.py --experiment=BSDS500_test_orientation_viz --model=BSDS_fixed_perturb --no_db --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_cheap_deepest_final_simple_BSDS500_combos_100_no_aux_2019_05_20_00_56_41_386546/model_1240.ckpt-1240 --placeholders --out_dir=orientation_probe_contrast_12 --train=circuit_contrast_circuit_exc_12_91 --val=circuit_contrast_circuit_exc_12_91
CUDA_VISIBLE_DEVICES=5 python run_job.py --experiment=BSDS500_test_orientation_viz --model=BSDS_fixed_perturb --no_db --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_cheap_deepest_final_simple_BSDS500_combos_100_no_aux_2019_05_20_00_56_41_386546/model_1240.ckpt-1240 --placeholders --out_dir=orientation_probe_contrast_12 --train=circuit_contrast_circuit_exc_12_121 --val=circuit_contrast_circuit_exc_12_121
CUDA_VISIBLE_DEVICES=5 python run_job.py --experiment=BSDS500_test_orientation_viz --model=BSDS_fixed_perturb --no_db --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_cheap_deepest_final_simple_BSDS500_combos_100_no_aux_2019_05_20_00_56_41_386546/model_1240.ckpt-1240 --placeholders --out_dir=orientation_probe_contrast_12 --train=circuit_contrast_circuit_exc_12_151 --val=circuit_contrast_circuit_exc_12_151