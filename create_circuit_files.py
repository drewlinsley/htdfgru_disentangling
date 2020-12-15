import os
import numpy as np
from copy import deepcopy
from glob2 import glob


data_dir = "/media/data_cifs_lrs/projects/prj_neural_circuits/"
# template = "datasets/orientation_tilt_viz.py"
template = "datasets/circuit_contrast_circuit_inh_001_full_field_1BAK.py"
# template = "datasets/orientation_tilt_viz_phase_180.py"
# template = "datasets/orientation_tilt_viz_phase_90.py"
# template = "datasets/orientation_tilt_viz_phase_0.py"
with open(template, "r+") as f:
    template_script = f.readlines()

tf_lines = [
    "#!/usr/bin/env bash\n# Autogen script for creating gilbert tfrecords\n\n",
    # "python encode_dataset.py --dataset=gilbert_length17_shearp6",  # Rebuild the template
]
rng = np.arange(1, 181, 30)
# rng = np.arange(1, 181, 15)
# Exp name/Dataset/init-responses/target-responses/<depreciated perturb strength>/thetas
experiments = [
    # ["circuit_exc_400", "orientation_probe", "../refactor_gammanet/orientation_probe_outputs/BSDS_vgg_gratings_simple_orientation_test_2020_09_27_10_14_47_500413.npz", "gammanet_full_orientation_probe_outputs_data.npy", 4.00, rng, "BSDS500_test_orientation_viz", None],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc
    # ["circuit_exc_200", "orientation_probe", "../refactor_gammanet/orientation_probe_outputs/BSDS_vgg_gratings_simple_orientation_test_2020_09_27_10_14_47_500413.npz", "gammanet_full_orientation_probe_outputs_data.npy", 2.00, rng, "BSDS500_test_orientation_viz", None],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc
    # ["circuit_exc_150", "orientation_probe", "../refactor_gammanet/orientation_probe_outputs/BSDS_vgg_gratings_simple_orientation_test_2020_09_27_10_14_47_500413.npz", "gammanet_full_orientation_probe_outputs_data.npy", 1.50, rng, "BSDS500_test_orientation_viz", None],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc


    ["circuit_exc_150_full_field", "orientation_probe", "../refactor_gammanet/orientation_probe_full_field_outputs/BSDS_vgg_gratings_simple_orientation_test_full_field_2020_12_10_16_46_24_554110.npz", "gammanet_full_orientation_probe_full_field_outputs_data.npy", 1.50, rng, "BSDS500_test_orientation_viz_inh", None],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc
    ["circuit_exc_140_full_field", "orientation_probe", "../refactor_gammanet/orientation_probe_full_field_outputs/BSDS_vgg_gratings_simple_orientation_test_full_field_2020_12_10_16_46_24_554110.npz", "gammanet_full_orientation_probe_full_field_outputs_data.npy", 1.40, rng, "BSDS500_test_orientation_viz_inh", None],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc
    ["circuit_exc_130_full_field", "orientation_probe", "../refactor_gammanet/orientation_probe_full_field_outputs/BSDS_vgg_gratings_simple_orientation_test_full_field_2020_12_10_16_46_24_554110.npz", "gammanet_full_orientation_probe_full_field_outputs_data.npy", 1.30, rng, "BSDS500_test_orientation_viz_inh", None],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc
    ["circuit_exc_120_full_field", "orientation_probe", "../refactor_gammanet/orientation_probe_full_field_outputs/BSDS_vgg_gratings_simple_orientation_test_full_field_2020_12_10_16_46_24_554110.npz", "gammanet_full_orientation_probe_full_field_outputs_data.npy", 1.20, rng, "BSDS500_test_orientation_viz_inh", None],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc
    ["circuit_exc_110_full_field", "orientation_probe", "../refactor_gammanet/orientation_probe_full_field_outputs/BSDS_vgg_gratings_simple_orientation_test_full_field_2020_12_10_16_46_24_554110.npz", "gammanet_full_orientation_probe_full_field_outputs_data.npy", 1.10, rng, "BSDS500_test_orientation_viz_inh", None],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc


    ["circuit_exc_105_full_field", "orientation_probe", "../refactor_gammanet/orientation_probe_full_field_outputs/BSDS_vgg_gratings_simple_orientation_test_full_field_2020_12_10_16_46_24_554110.npz", "gammanet_full_orientation_probe_full_field_outputs_data.npy", 1.05, rng, "BSDS500_test_orientation_viz_inh", None],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc
    ["circuit_inh_090_full_field", "orientation_probe", "../refactor_gammanet/orientation_probe_full_field_outputs/BSDS_vgg_gratings_simple_orientation_test_full_field_2020_12_10_16_46_24_554110.npz", "gammanet_full_orientation_probe_full_field_outputs_data.npy", 0.95, rng, "BSDS500_test_orientation_viz_inh", None],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc


    ["circuit_inh_090_full_field", "orientation_probe", "../refactor_gammanet/orientation_probe_full_field_outputs/BSDS_vgg_gratings_simple_orientation_test_full_field_2020_12_10_16_46_24_554110.npz", "gammanet_full_orientation_probe_full_field_outputs_data.npy", 0.90, rng, "BSDS500_test_orientation_viz_inh", None],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc
    ["circuit_inh_080_full_field", "orientation_probe", "../refactor_gammanet/orientation_probe_full_field_outputs/BSDS_vgg_gratings_simple_orientation_test_full_field_2020_12_10_16_46_24_554110.npz", "gammanet_full_orientation_probe_full_field_outputs_data.npy", 0.80, rng, "BSDS500_test_orientation_viz_inh", None],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc
    ["circuit_inh_070_full_field", "orientation_probe", "../refactor_gammanet/orientation_probe_full_field_outputs/BSDS_vgg_gratings_simple_orientation_test_full_field_2020_12_10_16_46_24_554110.npz", "gammanet_full_orientation_probe_full_field_outputs_data.npy", 0.70, rng, "BSDS500_test_orientation_viz_inh", None],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc
    ["circuit_inh_060_full_field", "orientation_probe", "../refactor_gammanet/orientation_probe_full_field_outputs/BSDS_vgg_gratings_simple_orientation_test_full_field_2020_12_10_16_46_24_554110.npz", "gammanet_full_orientation_probe_full_field_outputs_data.npy", 0.60, rng, "BSDS500_test_orientation_viz_inh", None],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc
    ["circuit_inh_050_full_field", "orientation_probe", "../refactor_gammanet/orientation_probe_full_field_outputs/BSDS_vgg_gratings_simple_orientation_test_full_field_2020_12_10_16_46_24_554110.npz", "gammanet_full_orientation_probe_full_field_outputs_data.npy", 0.50, rng, "BSDS500_test_orientation_viz_inh", None],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc

    # ["circuit_exc_050", "orientation_probe", "../refactor_gammanet/orientation_probe_outputs/BSDS_vgg_gratings_simple_orientation_test_2020_09_27_10_14_47_500413.npz", "gammanet_full_orientation_probe_outputs_data.npy", 0.50, rng, "BSDS500_test_orientation_viz", None],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc

    # ["circuit_inh_001", "orientation_probe", "../refactor_gammanet/orientation_probe_outputs/BSDS_vgg_gratings_simple_orientation_test_2020_09_27_10_14_47_500413.npz", "gammanet_full_orientation_probe_outputs_data.npy", 0.01, rng, "BSDS500_test_orientation_viz_inh", None],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc
    # ["circuit_inh_0001", "orientation_probe", "../refactor_gammanet/orientation_probe_outputs/BSDS_vgg_gratings_simple_orientation_test_2020_09_27_10_14_47_500413.npz", "gammanet_full_orientation_probe_outputs_data.npy", 0.001, rng, "BSDS500_test_orientation_viz_inh", None],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc
    # ["circuit_inh_00001", "orientation_probe", "../refactor_gammanet/orientation_probe_outputs/BSDS_vgg_gratings_simple_orientation_test_2020_09_27_10_14_47_500413.npz", "gammanet_full_orientation_probe_outputs_data.npy", 0.0001, rng, "BSDS500_test_orientation_viz_inh", None],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc
    # ["circuit_inh_000001", "orientation_probe", "../refactor_gammanet/orientation_probe_outputs/BSDS_vgg_gratings_simple_orientation_test_2020_09_27_10_14_47_500413.npz", "gammanet_full_orientation_probe_outputs_data.npy", 0.00001, rng, "BSDS500_test_orientation_viz_inh", None],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc
    # ["circuit_inh_0000001", "orientation_probe", "../refactor_gammanet/orientation_probe_outputs/BSDS_vgg_gratings_simple_orientation_test_2020_09_27_10_14_47_500413.npz", "gammanet_full_orientation_probe_outputs_data.npy", 0.000001, rng, "BSDS500_test_orientation_viz_inh", None],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc

    # ["circuit_inh", "orientation_probe", "../refactor_gammanet/orientation_probe_outputs/BSDS_vgg_gratings_simple_orientation_test_2020_09_27_10_14_47_500413.npz", "gammanet_full_orientation_probe_outputs_data.npy", 0.01, rng],  # Demonstrate excitation/inh on T&B single-component stimuli  Inh

    # ["circuit_exc_06", "orientation_probe", "../refactor_gammanet/orientation_probe_outputs/BSDS_vgg_gratings_simple_orientation_test_2020_09_27_10_14_47_500413.npz", "gammanet_full_orientation_probe_contrast_06_outputs_data.npy", 1.3, rng, "BSDS500_test_orientation_viz", 0.06],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc
    # ["circuit_exc_12", "orientation_probe", "../refactor_gammanet/orientation_probe_outputs/BSDS_vgg_gratings_simple_orientation_test_2020_09_27_10_14_47_500413.npz", "gammanet_full_orientation_probe_contrast_12_outputs_data.npy", 1.3, rng, "BSDS500_test_orientation_viz", 0.12],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc
    # ["circuit_exc_25", "orientation_probe", "../refactor_gammanet/orientation_probe_outputs/BSDS_vgg_gratings_simple_orientation_test_2020_09_27_10_14_47_500413.npz", "gammanet_full_orientation_probe_contrast_25_outputs_data.npy", 1.3, rng, "BSDS500_test_orientation_viz", 0.25],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc
    # ["circuit_exc_50", "orientation_probe", "../refactor_gammanet/orientation_probe_outputs/BSDS_vgg_gratings_simple_orientation_test_2020_09_27_10_14_47_500413.npz", "gammanet_full_orientation_probe_contrast_50_outputs_data.npy", 1.3, rng, "BSDS500_test_orientation_viz", 0.50],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc
    # ["circuit_exc_75", "orientation_probe", "../refactor_gammanet/orientation_probe_outputs/BSDS_vgg_gratings_simple_orientation_test_2020_09_27_10_14_47_500413.npz", "gammanet_full_orientation_probe_contrast_75_outputs_data.npy", 1.3, rng, "BSDS500_test_orientation_viz", 0.75],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc
    # ["circuit_exc", "orientation_probe", "../refactor_gammanet/orientation_probe_outputs/BSDS_vgg_gratings_simple_orientation_test_2020_09_27_10_14_47_500413.npz", "gammanet_full_orientation_probe_outputs_data.npy", 1.3, rng, "BSDS500_test_orientation_viz", None],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc

    ### Flipped -- fix target, change the inputs
    # ["circuit_exc_06", "orientation_probe", "../refactor_gammanet/orientation_probe_contrast_06_outputs/BSDS_vgg_gratings_simple_orientation_test_contrast_0_06_hack_2020_11_06_23_22_30_102494.npz", "gammanet_full_orientation_probe_outputs_data.npy", 1., rng, "BSDS500_test_orientation_viz", 0.06],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc
    # ["circuit_exc_12", "orientation_probe", "../refactor_gammanet/orientation_probe_contrast_12_outputs/BSDS_vgg_gratings_simple_orientation_test_contrast_0_12_hack_2020_11_06_23_24_24_550974.npz", "gammanet_full_orientation_probe_outputs_data.npy", 1., rng, "BSDS500_test_orientation_viz", 0.12],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc
    # ["circuit_exc_25", "orientation_probe", "../refactor_gammanet/orientation_probe_contrast_25_outputs/BSDS_vgg_gratings_simple_orientation_test_contrast_0_25_hack_2020_11_06_23_26_18_279541.npz", "gammanet_full_orientation_probe_outputs_data.npy", 1., rng, "BSDS500_test_orientation_viz", 0.25],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc
    # ["circuit_exc_50", "orientation_probe", "../refactor_gammanet/orientation_probe_contrast_50_outputs/BSDS_vgg_gratings_simple_orientation_test_contrast_0_50_hack_2020_11_06_23_28_10_945117.npz", "gammanet_full_orientation_probe_outputs_data.npy", 1., rng, "BSDS500_test_orientation_viz", 0.50],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc
    # ["circuit_exc_75", "orientation_probe", "../refactor_gammanet/orientation_probe_contrast_75_outputs/BSDS_vgg_gratings_simple_orientation_test_contrast_0_75_hack_2020_11_06_23_30_04_371668.npz", "gammanet_full_orientation_probe_outputs_data.npy", 1., rng, "BSDS500_test_orientation_viz", 0.75],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc

    # ["circuit_exc_full_field", "orientation_probe", "../refactor_gammanet/orientation_probe_full_field/BSDS_vgg_gratings_simple_orientation_test_full_field_2020_10_21_15_20_15_758731.npz", "gammanet_full_orientation_probe_full_field_data.npy", 1.3, rng, "BSDS500_test_orientation_viz", None],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc
    # ["circuit_inh_full_field", "orientation_probe", "../refactor_gammanet/orientation_probe_full_field/BSDS_vgg_gratings_simple_orientation_test_full_field_2020_10_21_15_20_15_758731.npz", "gammanet_full_orientation_probe_full_field_data.npy", 0.01, rng, "BSDS500_test_orientation_viz", None],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc

    # ["circuit_plaid_exc", "plaid_surround", "../refactor_gammanet/plaid_no_surround_outputs/BSDS_vgg_gratings_simple_plaid_no_surround_2020_09_27_10_12_52_983631.npz", "gammanet_full_plaid_surround_outputs_data.npy", 1.3, rng, "BSDS500_test_orientation_viz", None],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc
    # ["circuit_plaid_inh", "plaid_surround", "../refactor_gammanet/plaid_no_surround_outputs/BSDS_vgg_gratings_simple_plaid_no_surround_2020_09_27_10_12_52_983631.npz", "gammanet_full_plaid_surround_outputs_data.npy", 0.01, rng, "BSDS500_test_orientation_viz", None],  # Demonstrate excitation/inh on T&B single-component stimuli  Inh
    # ["circuit_bwc_inh", "contrast_modulated", "../refactor_gammanet/INSILICO_BSDS_vgg_gratings_simple_contrast_test/BSDS_vgg_gratings_simple_contrast_test_2020_10_05_23_29_08_737162.npz", "gammanet_full_contrast_modulated_outputs_data.npy", 0.01, rng, "BSDS500_test_orientation_viz", None],  # Center-surround BWC stimuli. Prediction.
    # ["circuit_bwc_exc", "contrast_modulated", "../refactor_gammanet/INSILICO_BSDS_vgg_gratings_simple_contrast_test/BSDS_vgg_gratings_simple_contrast_test_2020_10_05_23_29_08_737162.npz", "gammanet_full_contrast_modulated_outputs_data.npy", 1.3, rng, "BSDS500_test_orientation_viz", None],  # Center-surround BWC stimuli. Prediction.


    # ["circuit_FS", "plaid_surround", "../refactor_gammanet/plaid_no_surround_outputs/BSDS_vgg_gratings_simple_plaid_no_surround_2020_09_27_10_12_52_983631.npz", "gammanet_full_plaid_surround_outputs_data.npy", 1., rng],  # Use T&B center-only plaid stimuli as perturbation for plaid center-surround stimuli


    # ["circuit_roll_1", "orientation_probe", "../refactor_gammanet/orientation_probe_outputs/BSDS_vgg_gratings_simple_orientation_test_2020_09_27_10_14_47_500413.npz", "gammanet_full_orientation_probe_outputs_data.npy", 1, rng],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc
    # ["circuit_roll_-1", "orientation_probe", "../refactor_gammanet/orientation_probe_outputs/BSDS_vgg_gratings_simple_orientation_test_2020_09_27_10_14_47_500413.npz", "gammanet_full_orientation_probe_outputs_data.npy", -1, rng],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc

    # ["circuit_roll_2", "orientation_probe", "../refactor_gammanet/orientation_probe_outputs/BSDS_vgg_gratings_simple_orientation_test_2020_09_27_10_14_47_500413.npz", "gammanet_full_orientation_probe_outputs_data.npy", 2, rng],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc
    # ["circuit_roll_-2", "orientation_probe", "../refactor_gammanet/orientation_probe_outputs/BSDS_vgg_gratings_simple_orientation_test_2020_09_27_10_14_47_500413.npz", "gammanet_full_orientation_probe_outputs_data.npy", -2, rng],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc

    # ["circuit_roll_3", "orientation_probe", "../refactor_gammanet/orientation_probe_outputs/BSDS_vgg_gratings_simple_orientation_test_2020_09_27_10_14_47_500413.npz", "gammanet_full_orientation_probe_outputs_data.npy", 3, rng],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc
    # ["circuit_roll_-3", "orientation_probe", "../refactor_gammanet/orientation_probe_outputs/BSDS_vgg_gratings_simple_orientation_test_2020_09_27_10_14_47_500413.npz", "gammanet_full_orientation_probe_outputs_data.npy", -3, rng],  # Demonstrate excitation/inh on T&B single-component stimuli  Exc

    ## Old bad
    # ["circuit_native_inh", "orientation_probe", "../refactor_gammanet/orientation_probe_no_surround_outputs/BSDS_vgg_gratings_simple_orientation_test_no_surround_2020_09_27_10_15_45_604237.npz", "gammanet_full_orientation_probe_outputs_data.npy", 1., rng],  # Demonstrate excitation/inh on T&B single-component stimuli  Inh
    # ["circuit_FS", "plaid_surround", "../refactor_gammanet/plaid_no_surround_outputs/BSDS_vgg_gratings_simple_plaid_no_surround_2020_09_27_10_12_52_983631.npz", "gammanet_full_plaid_surround_outputs_data.npy", 1., rng],  # Use T&B center-only plaid stimuli as perturbation for plaid center-surround stimuli
    # ["circuit_native_exc", "orientation_probe_no_surround", "../refactor_gammanet/orientation_probe_outputs/BSDS_vgg_gratings_simple_orientation_test_2020_09_27_10_14_47_500413.npz", "gammanet_full_orientation_probe_no_surround_outputs_data.npy", 1., rng],  # Demonstrate excitation/inh on T&B single-component stimuli  Inh
    # ["circuit_ff_exc", "orientation_probe", "../refactor_gammanet/orientation_probe_no_surround_outputs/BSDS_vgg_gratings_simple_orientation_test_no_surround_2020_09_27_10_15_45_604237.npz", "gammanet_full_orientation_probe_no_surround_outputs_data.npy", 4., rng],  # Exc on full-field stimuli
    # ["circuit_ff_inh", "orientation_probe", "../refactor_gammanet/orientation_probe_no_surround_outputs/BSDS_vgg_gratings_simple_orientation_test_no_surround_2020_09_27_10_15_45_604237.npz", "gammanet_full_orientation_probe_no_surround_outputs_data.npy", 0.5, rng],  # Inh on full-field stimuli

    ## Older badder
    # ["circuit_FS", "plaid_surround", "gammanet_full_plaid_no_surround_outputs_data.npy", "gammanet_full_plaid_surround_outputs_data.npy", 1., rng],  # Use T&B center-only plaid stimuli as perturbation for plaid center-surround stimuli
    # ["circuit_tuned", "orientation_tilt", "gammanet_full_orientation_probe_no_surround_outputs_data.npy", "gammanet_full_orientation_tilt_outputs_data.npy", 1., rng],  # Use T&B center-only SO stimuli as perturbation for SO center-surround stimuli
    # ["circuit_wta", "contrast_modulated_no_surround", "gammanet_full_contrast_modulated_no_surround_outputs_data_flipped.npy", "gammanet_full_contrast_modulated_no_surround_outputs_data.npy", None, [12, 15]],  # Use the transpose position as target for WTA stimuli. I.e. contrast 0.5x0.25 with 0.25x0.5
]

for idx, experiment in enumerate(experiments):
    gpu = (idx + 1) % 8  # idx
    exp_name = experiment[6]
    mod = experiment[7]
    cmd = "CUDA_VISIBLE_DEVICES={} python run_job.py --experiment={} --model=BSDS_exc_perturb --no_db --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_cheap_deepest_final_simple_BSDS500_combos_100_no_aux_2019_05_20_00_56_41_386546/model_1240.ckpt-1240 --placeholders --out_dir=perturb_viz --train=orientation_tilt_viz --val=orientation_tilt_viz\n".format(gpu, exp_name)  # noqa

    cmds = [
        '#!/usr/bin/env bash\n# Autogen script for running models\n\n',
        # cmd,
    ]
    exp = experiment[0]
    dataset = experiment[1]
    perturb_data = experiment[2]
    target_data = experiment[3]
    perturbation = experiment[4]
    thetas = experiment[5]
    for theta in thetas:
        # TFRecords
        target_file = deepcopy(template_script)

        # Change dataset name
        target_file[11] = target_file[11].replace("orientation_tilt", dataset)

        # Change tfrecord output name
        target_file[13] = target_file[13].replace("orientation_tilt_viz", "{}_{}_viz".format(dataset, theta))

        # Change perturbation
        # target_file[17] = target_file[17].replace("gammanet_full_plaid_surround_outputs_data.npy", perturb_data)
        target_file[18] = target_file[18].replace(
            "../refactor_gammanet/plaid_surround_outputs/BSDS_vgg_gratings_simple_plaid_surround_2020_09_27_10_13_50_541685.npz",
            perturb_data)

        # Change target
        target_file[19] = target_file[19].replace("gammanet_full_plaid_surround_outputs_data.npy", target_data)

        # Change thetas
        if 0:  # "orientation_probe" in perturb_data:
            target_file[152] = target_file[152].replace("[0]", "[{}]".format(theta))
        else:
            target_file[157] = target_file[157].replace("[0]", "[{}]".format(theta - 1))

        if 0:  # "orientation_probe" in target_data:
            target_file[155] = target_file[155].replace("[0]", "[{}]".format(theta))
        else:
            target_file[160] = target_file[160].replace("[0]", "[{}]".format(theta - 1))

        # Mod hacks
        if mod is not None:
            target_file[109] = target_file[109].replace("10", "10 / ({} / 2 + 0.6)".format(mod))
            it_dataset = "{}_contrast_{}".format(dataset, str(mod).split(".")[-1])  # Rewrite the outdir too
        else:
            it_dataset = "{}_{}".format(exp, dataset)

        if "full_field" in exp:
            target_file[84] = target_file[84].replace("test", "train")
            target_file[88] = target_file[88].replace("test", "train")
            target_file[98] = target_file[98].replace("test", "train")
            target_file[102] = target_file[102].replace("test", "train")

        # Write the file
        if "optim_contrast" in dataset:
            data_name = "circuit_contrast_{}_{}_{}".format(dataset.split(".")[1].split("_")[0], exp, theta)
        else:
            data_name = "circuit_contrast_{}_{}".format(exp, theta)
        out_file = os.path.join("datasets", "{}.py".format(data_name))
        with open(out_file, "w") as f:
            f.writelines(target_file)

        # CMDs
        # if perturbation == 1:  # These arent working correctly so we will punt
        #     model = "BSDS_data_perturb_viz"
        """
        if perturbation == 1:
            model = "BSDS_roll_1_perturb"
        elif perturbation == 2:
            model = "BSDS_roll_2_perturb"
        elif perturbation == 3:
            model = "BSDS_roll_3_perturb"
        elif perturbation == -1:
            model = "BSDS_roll_-1_perturb"
        elif perturbation == -2:
            model = "BSDS_roll_-2_perturb"
        elif perturbation == -3:
            model = "BSDS_roll_-3_perturb"
        """
        if perturbation == 1.:
            model = "BSDS_fixed_perturb"
        elif perturbation == 4.0:
            model = "BSDS_exc_400_perturb"
        elif perturbation == 2.0:
            model = "BSDS_exc_200_perturb"
        elif perturbation == 1.5:
            model = "BSDS_exc_150_perturb"
        elif perturbation == 1.25:
            model = "BSDS_exc_125_perturb"
        elif perturbation == 1.50:
            model = "BSDS_exc_150_perturb"
        elif perturbation == 1.40:
            model = "BSDS_exc_140_perturb"
        elif perturbation == 1.30:
            model = "BSDS_exc_130_perturb"
        elif perturbation == 1.20:
            model = "BSDS_exc_120_perturb"
        elif perturbation == 1.10:
            model = "BSDS_exc_110_perturb"
        elif perturbation == 1.05:
            model = "BSDS_exc_105_perturb"

        elif perturbation == 0.95:
           model = "BSDS_inh_095_perturb"
        elif perturbation == 0.90:
           model = "BSDS_inh_090_perturb"
        elif perturbation == 0.80:
           model = "BSDS_inh_080_perturb"
        elif perturbation == 0.70:
           model = "BSDS_inh_070_perturb"
        elif perturbation == 0.60:
           model = "BSDS_inh_060_perturb"
        elif perturbation == 0.50:
           model = "BSDS_inh_050_perturb"

        elif perturbation == 0.75:
           model = "BSDS_inh_075_perturb"
        elif perturbation == 0.50:
           model = "BSDS_inh_050_perturb"
        elif perturbation == 0.25:
           model = "BSDS_inh_025_perturb"
        elif perturbation == 0.01:
           model = "BSDS_inh_001_perturb"
        elif perturbation == 0.001:
           model = "BSDS_inh_0001_perturb"
        elif perturbation == 0.0001:
           model = "BSDS_inh_00001_perturb"
        elif perturbation == 0.00001:
           model = "BSDS_inh_000001_perturb"
        elif perturbation == 0.000001:
           model = "BSDS_inh_0000001_perturb"
        elif perturbation == 0.1:
           model = "BSDS_inh_01_perturb"
        else:
           raise NotImplementedError("Perturbation {} not implemented.".format(perturbation))

        # elif perturbation > 1:
        #     model = "BSDS_exc_perturb"
        # elif perturbation < 1:
        #     model = "BSDS_inh_perturb"
        it_cmd = deepcopy(cmd)
        it_cmd = it_cmd.replace("--model=BSDS_exc_perturb", "--model={}".format(model))
        it_cmd = it_cmd.replace("--out_dir=perturb_viz", "--out_dir={}".format(it_dataset))  # exp
        it_cmd = it_cmd.replace("--train=orientation_tilt_viz", "--train={}".format(data_name))
        it_cmd = it_cmd.replace("--val=orientation_tilt_viz", "--val={}".format(data_name))
        cmds.append(it_cmd)

    # Create the model eval script
    with open("run_circuit_exps_{}.sh".format(idx), "w") as f:
        f.writelines(cmds)

