#!/bin/bash

# # Inputs:
# 0. GPU ID
# 1. CSV with model results
# 2. Test experiment name
# 3. Output name

cd /media/data_cifs/cluster_projects/cluttered_nist_experiments 
GPU="1"
CKPTCSV="data_to_process_for_jk/inv_main_synth_experiment_data/inv_main_synth_experiment_data_ckpts_only.csv"
MODELCSV="data_to_process_for_jk/inv_main_synth_experiment_data/inv_main_synth_experiment_data_models_only.csv"
TRAINDSCSV="data_to_process_for_jk/inv_main_synth_experiment_data/inv_main_synth_experiment_data_trainds_only.csv"
declare -a EXPERIMENTS=("v2_synth_connectomics_test_baseline" "v2_synth_connectomics_test_size_1" "v2_synth_connectomics_test_size_2" "v2_synth_connectomics_test_size_3" "v2_synth_connectomics_test_lum_1" "v2_synth_connectomics_test_lum_2" "v2_synth_connectomics_test_lum_3")
# declare -a EXPERIMENTS=("v2_synth_connectomics_test_lum_1" "v2_synth_connectomics_test_lum_2" "v2_synth_connectomics_test_lum_3")
OUTPUT_PREFIX="synth_invariance_"

# Load a results CSV then begin parsing its models
readarray -t array1 < <(cut -d, -f2 $CKPTCSV | awk '{if(NR>1)print}')
readarray -t array2 < <(cut -d, -f2 $MODELCSV | awk '{if(NR>1)print}')
readarray -t array3 < <(cut -d, -f2 $TRAINDSCSV | awk '{if(NR>1)print}')
echo $array1
echo $array2
echo $array3
for experiment in "${EXPERIMENTS[@]}"
do
    OUTPUT=$OUTPUT_PREFIX$experiment
    echo "Saving data in $OUTPUT"
    for (( i=0; i<${#array1[@]}; i++ ));
    do
        echo "Beginning $i, command: CUDA_VISIBLE_DEVICES=$GPU python run_job.py --experiment=$experiment --ckpt=${array1[$i]} --model=${array2[$i]} --add_config=${array3[$i]} --out_dir=$OUTPUT --placeholders --test --no_db --no_npz"
        CUDA_VISIBLE_DEVICES=$GPU python run_job.py --experiment=$experiment --ckpt=${array1[$i]} --model=${array2[$i]} --add_config=${array3[$i]} --out_dir=$OUTPUT --placeholders --test --no_db --no_npz
    done
done

