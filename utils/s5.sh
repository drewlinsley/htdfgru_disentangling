#!/bin/bash

COUNTER=0
while [ $COUNTER -lt 10 ]; do
	echo Step $COUNTER
	CUDA_VISIBLE_DEVICES=5 python run_job.py --experiment=snemi_combos --train=snemi_200 --val=snemi_200 --model=seung_unet_per_pixel_adabn --no_db
	let COUNTER=COUNTER+1
done

