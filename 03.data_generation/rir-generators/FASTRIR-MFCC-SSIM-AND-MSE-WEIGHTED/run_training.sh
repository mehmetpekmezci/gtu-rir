#!/bin/bash

CURRENT_DIR=$(pwd)

DEFAULT_MAIN_DATA_DIR=$HOME/RIR_DATA

if [ "$FAST_RIR_TRAINING_DATA" = "" ]
then
	echo "FAST_RIR_TRAINING_DATA env. var. is not defined"
        export FAST_RIR_TRAINING_DATA=$DEFAULT_MAIN_DATA_DIR/FAST_RIR_TRAINING_DATA
        echo "FAST_RIR_TRAINING_DATA=$FAST_RIR_TRAINING_DATA"
fi

../TRAINING_DATA-PREPARE-FASTRIR/prepare_data.sh

cd $CURRENT_DIR/code_new 

python3 main.py --cfg cfg/RIR_s1.yml --gpu 0 --data_dir $FAST_RIR_TRAINING_DATA/data/Medium_Room
