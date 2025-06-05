#!/bin/bash

CURRENT_DIR=$(pwd)

DEFAULT_MAIN_DATA_DIR=$HOME/RIR_DATA

if [ "$MESH2IR_EVALUATION_DATA" = "" ]
then
	echo "MESH2IR_EVALUATION_DATA env. var. is not defined"
        export MESH2IR_EVALUATION_DATA=$DEFAULT_MAIN_DATA_DIR/MESH2IR_TRAINING_DATA
        echo "MESH2IR_EVALUATION_DATA=$MESH2IR_EVALUATION_DATA"
fi


if [ ! -f "$MESH2IR_EVALUATION_DATA/3D-FRONT/outputs/validation.embeddings.pickle" ]
then
        cd $CURRENT_DIR/train/
	python3 embed_generator.py $MESH2IR_TRAINING_DATA/3D-FRONT/outputs
else
	echo "$MESH2IR_TRAINING_DATA/3D-FRONT/outputs/validation.embeddings.pickle file is already prepared ..."
fi

cd $CURRENT_DIR/evaluate

python3 evaluate.py  $MESH2IR_EVALUATION_DATA/3D-FRONT/outputs   validation.embeddings.pickle


