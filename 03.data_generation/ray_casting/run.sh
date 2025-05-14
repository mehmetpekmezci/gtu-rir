#!/bin/bash


if [ $# -lt 3 ]
then
	echo "Usage: $0 <FORCE/SKIP_IF_ALREADY_DONE> RESOLUTION_QUOTIENT MAX_DISTANCE_IN_A_ROOM"
	### RES_QUOT=30 ise 30 derecede bir ray var demektir.
	echo "Example: $0 FORCE 30 15"
	echo "the final ray_casting image will of shape 360/30 by 180/30 = 12 x 6"
	exit 1
fi

echo
echo "THIS SCRIPT NEEDS BLENDER 4.0, if the blender version is later, ray_casting_blender.py script may fail !!!!"
echo
echo
sleep 3

DEFAULT_MAIN_DATA_DIR=$HOME/RIR_DATA


MODE=$1
RESOLUTION_QUOTIENT=$2
MAX_DISTANCE_IN_A_ROOM=$3

if [ "$MESH2IR_TRAINING_DATA" = "" ]
then
        echo "MESH2IR_TRAINING_DATA env. var. is not defined"
        export MESH2IR_TRAINING_DATA=$DEFAULT_MAIN_DATA_DIR/MESH2IR_TRAINING_DATA
        echo "MESH2IR_TRAINING_DATA=$MESH2IR_TRAINING_DATA"
fi



if [ "$MODE" == "FORCE" ]
then
	rm -f $MESH2IR_TRAINING_DATA/3D-FRONT/outputs/.ray_casting_generated
fi

if [ ! -f "$MESH2IR_TRAINING_DATA/3D-FRONT/outputs/.ray_casting_generated" ]
then
        python3 main.py $MESH2IR_TRAINING_DATA/3D-FRONT/outputs $RESOLUTION_QUOTIENT $MAX_DISTANCE_IN_A_ROOM 
	touch "$MESH2IR_TRAINING_DATA/3D-FRONT/outputs/.ray_casting_generated"
else
        echo "$MESH2IR_TRAINING_DATA/3D-FRONT/outputs/ ray_casting files are already prepared ..."
fi

