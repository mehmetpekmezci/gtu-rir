#!/bin/bash

CURRENT_DIR=$(pwd)

DEFAULT_MAIN_DATA_DIR=$HOME/RIR_DATA

if [ "$MESH2IR_TRAINING_DATA" = "" ]
then
	echo "MESH2IR_TRAINING_DATA env. var. is not defined"
        export MESH2IR_TRAINING_DATA=$DEFAULT_MAIN_DATA_DIR/MESH2IR_TRAINING_DATA
        echo "MESH2IR_TRAINING_DATA=$MESH2IR_TRAINING_DATA"
fi

if [ ! -d "$MESH2IR_TRAINING_DATA/3D-FRONT/outputs" ]
then
	cd ../TRAINING_DATA-PREPARE-MESH2IR/
	./prepare_dataset.sh
else
	echo "$MESH2IR_TRAINING_DATA/3D-FRONT/outputs training data is already prepared ..."
fi

if [ ! -f "$MESH2IR_TRAINING_DATA/3D-FRONT/outputs/embeddings.pickle" ]
then
        cd $CURRENT_DIR/train/
	python3 embed_generator.py $MESH2IR_TRAINING_DATA/3D-FRONT/outputs
else
	echo "$MESH2IR_TRAINING_DATA/3D-FRONT/outputs/embeddings.pickle file is already prepared ..."
fi

cd $CURRENT_DIR/train/MESH2IR 

python3 main.py --cfg cfg/RIR_s1.yml --gpu 0 --data_dir $MESH2IR_TRAINING_DATA/3D-FRONT/outputs