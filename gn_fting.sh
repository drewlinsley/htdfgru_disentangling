CUDA_VISIBLE_DEVICES=0 python run_job.py --no_db --experiment=nist_3_ix2v2_50k_frozen --model=BSDS_vgg_pf_unfrozen_less_nl --train=cluttered_nist_3_ix2v2_50k --val=cluttered_nist_3_ix2v2_50k
CUDA_VISIBLE_DEVICES=2 python run_job.py --no_db --experiment=nist_3_ix2v2_50k_frozen --model=BSDS_vgg_cabc_unfrozen --train=cluttered_nist_3_ix2v2_50k --val=cluttered_nist_3_ix2v2_50k
CUDA_VISIBLE_DEVICES=1 python run_job.py --no_db --experiment=pathfinder_14_frozen --model=BSDS_vgg_cabc --train=curv_contour_length_14_50k --val=curv_contour_length_14_50k
CUDA_VISIBLE_DEVICES=3 python run_job.py --no_db --experiment=pathfinder_14_frozen --model=BSDS_vgg_cabc_unfrozen --train=curv_contour_length_14_50k --val=curv_contour_length_14_50k
CUDA_VISIBLE_DEVICES=5 python run_job.py --no_db --experiment=pathfinder_14_frozen --model=BSDS_vgg_pf_unfrozen_less_nl --train=curv_contour_length_14_50k --val=curv_contour_length_14_50k



CUDA_VISIBLE_DEVICES=0 python run_job.py --no_db --experiment=pathfinder_14_frozen_224 --model=BSDS_vgg_gestalt --train=curv_contour_length_14_50k_224 --val=curv_contour_length_14_50k_224 --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_gratings_simple_gratings_2019_05_20_12_13_01_973409/model_1900.ckpt-1900


CUDA_VISIBLE_DEVICES=1 python run_job.py --no_db --experiment=pathfinder_14_frozen_224 --model=BSDS_vgg_gestalt_lowlevel --train=curv_contour_length_14_50k_224 --val=curv_contour_length_14_50k_224 --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_gratings_simple_gratings_2019_05_20_12_13_01_973409/model_1900.ckpt-1900



CUDA_VISIBLE_DEVICES=2 python run_job.py --no_db --experiment=pathfinder_14_frozen_224_trainable --model=BSDS_vgg_gestalt_lowlevel_trainable --train=curv_contour_length_14_50k_224 --val=curv_contour_length_14_50k_224 --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_gratings_simple_gratings_2019_05_20_12_13_01_973409/model_1900.ckpt-1900


CUDA_VISIBLE_DEVICES=1 python run_job.py --no_db --experiment=pathfinder_14_frozen_224_trainable --model=BSDS_vgg_gestalt_lowlevel_trainable --train=curv_contour_length_14_50k --val=curv_contour_length_14_50k --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_cheap_deepest_final_simple_BSDS500_combos_100_no_aux_2019_05_20_00_56_41_386546/model_1240.ckpt-1240 --no_db

CUDA_VISIBLE_DEVICES=0 python run_job.py --no_db --experiment=pathfinder_14_frozen_224_trainable --model=BSDS_vgg_gestalt_lowlevel_trainable_batch --train=curv_contour_length_14_50k --val=curv_contour_length_14_50k --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_cheap_deepest_final_simple_BSDS500_combos_100_no_aux_2019_05_20_00_56_41_386546/model_1240.ckpt-1240 --no_db

CUDA_VISIBLE_DEVICES=0 python run_job.py --no_db --experiment=pathfinder_14_frozen_224_trainable_faster --model=BSDS_vgg_gestalt_lowlevel_trainable_v2 --train=curv_contour_length_14_5k_224 --val=curv_contour_length_14_5k_224 --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_cheap_deepest_final_simple_BSDS500_combos_100_no_aux_2019_05_20_00_56_41_386546/model_1240.ckpt-1240 --no_db

CUDA_VISIBLE_DEVICES=1 python run_job.py --no_db --experiment=pathfinder_14_frozen_224_trainable_faster --model=BSDS_vgg_gestalt_lowlevel_trainable_v2 --train=curv_contour_length_14_50k_224 --val=curv_contour_length_14_50k_224 --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_cheap_deepest_final_simple_BSDS500_combos_100_no_aux_2019_05_20_00_56_41_386546/model_1240.ckpt-1240 --no_db


---

python run_job.py --no_db --experiment=seg_gestalt --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_cheap_deepest_final_simple_BSDS500_combos_100_no_aux_2019_05_20_00_56_41_386546/model_1240.ckpt-1240 --model=BSDS_vgg_cabc_v2
python run_job.py --no_db --experiment=seg_gestalt --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_cabc_v2_seg_gestalt_2020_06_25_13_38_20_160191/model_2400.ckpt-2400 --model=BSDS_vgg_cabc_v2 --test

python run_job.py --no_db --experiment=seg_gestalt --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_cabc_v2_seg_gestalt_2020_06_25_16_35_19_051649/model_35000.ckpt-35000 --model=BSDS_vgg_cabc_v2 --test


---

# Train for saliency
CUDA_VISIBLE_DEVICES=7 python run_job.py --no_db --experiment=gilbert --model=BSDS_vgg_gilbert --train=gilbert_length21_shear0_0 --val=gilbert_length21_shear0_0 --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_gratings_simple_gratings_2019_05_20_12_13_01_973409/model_1900.ckpt-1900

# Test for saliency
CUDA_VISIBLE_DEVICES=7 python run_job.py --no_db --experiment=gilbert --model=BSDS_vgg_gilbert --train=gilbert_length17_shearp6 --val=gilbert_length17_shear0_6 --ckpt=/media/data_cifs_lrs/projects/prj_neural_circuits/gammanet/checkpoints/BSDS_vgg_gilbert_gilbert_2020_08_01_16_47_02_940774/model_1000.ckpt-1000 --test --out_dir=gilbert_test


---

# # Visualization
# First get target activities
CUDA_VISIBLE_DEVICES=5 python run_job.py --experiment=BSDS500_test --model=BSDS_vgg_cheap_deepest_final_simple --no_db --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_cheap_deepest_final_simple_BSDS500_combos_100_no_aux_2019_05_20_00_56_41_386546/model_1240.ckpt-1240 --placeholders --test --out_dir=perturb_viz --train=BSDS500_test_landscape --val=BSDS500_test_landscape

CUDA_VISIBLE_DEVICES=5 python run_job.py --experiment=BSDS500_test_orientation --model=BSDS_vgg_gratings_simple --no_db --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_cheap_deepest_final_simple_BSDS500_combos_100_no_aux_2019_05_20_00_56_41_386546/model_1240.ckpt-1240 --placeholders --test --out_dir=perturb_viz --train=orientation_tilt --val=orientation_tilt

# Then perform optimization BSDS
# Exc
CUDA_VISIBLE_DEVICES=6 python run_job.py --experiment=BSDS500_test_viz --model=BSDS_exc_perturb --no_db --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_cheap_deepest_final_simple_BSDS500_combos_100_no_aux_2019_05_20_00_56_41_386546/model_1240.ckpt-1240 --placeholders --out_dir=perturb_viz --train=BSDS500_test_portrait_viz --val=BSDS500_test_portrait_viz

# Inh
CUDA_VISIBLE_DEVICES=5 python run_job.py --experiment=BSDS500_test_viz --model=BSDS_inh_perturb --no_db --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_cheap_deepest_final_simple_BSDS500_combos_100_no_aux_2019_05_20_00_56_41_386546/model_1240.ckpt-1240 --placeholders --out_dir=perturb_viz --train=BSDS500_test_portrait_viz --val=BSDS500_test_portrait_viz

# Then perform optimization Orientation-tilt
# Exc
CUDA_VISIBLE_DEVICES=6 python run_job.py --experiment=BSDS500_test_orientation_viz --model=BSDS_exc_perturb_viz --no_db --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_cheap_deepest_final_simple_BSDS500_combos_100_no_aux_2019_05_20_00_56_41_386546/model_1240.ckpt-1240 --placeholders --out_dir=perturb_viz --train=orientation_tilt_viz --val=orientation_tilt_viz

# Inh
CUDA_VISIBLE_DEVICES=5 python run_job.py --experiment=BSDS500_test_orientation_viz --model=BSDS_inh_perturb_viz --no_db --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_cheap_deepest_final_simple_BSDS500_combos_100_no_aux_2019_05_20_00_56_41_386546/model_1240.ckpt-1240 --placeholders --out_dir=perturb_viz --train=orientation_tilt_viz --val=orientation_tilt_viz
