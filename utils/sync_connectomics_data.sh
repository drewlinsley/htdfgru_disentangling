#!/bin/bash

echo Synching mAPs with CCV
rsync --progress dlinsley@transfer.ccv.brown.edu:/users/dlinsley/cluttered_nist_experiments/maps/* maps/

echo Synching checkpoints with CCV
rsync -r --progress dlinsley@transfer.ccv.brown.edu:/gpfs/data/tserre/data/drew_cnist/checkpoints/* /media/data_cifs/cluttered_nist_experiments/checkpoints

