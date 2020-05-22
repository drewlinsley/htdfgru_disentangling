#!/bin/bash

# # Inputs:
# 0. GPU ID
# 1. CSV with model results
# 2. Test experiment name
# 3. Output name

cd /media/data_cifs/cluster_projects/cluttered_nist_experiments 
GPU="7"
CKPTCSV="data_to_process_for_jk/snemi_experiment_data/snemi_experiment_data_ckpts_only.csv"
MODELCSV="data_to_process_for_jk/snemi_experiment_data/snemi_experiment_data_models_only.csv"
TRAINDSCSV="data_to_process_for_jk/snemi_experiment_data/snemi_experiment_data_trainds_only.csv"
declare -a EXPERIMENTS=("cremi_combos") 
OUTPUT_PREFIX="connectomics_generalization_"

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

