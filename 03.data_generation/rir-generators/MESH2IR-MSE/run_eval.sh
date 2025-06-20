#!/bin/bash

#./run_eval.sh  $HOME/RIR_DATA/GTU-RIR-1.0/data/single-speaker $HOME/RIR_REPORT/GTURIR/MESH2IR-MSE GTURIR
if [ $# -lt 2 ]
then
	echo "Usage $0 <MESH2IR_INPUT_DATA_DIR>  <GENERATED_RIRS_DIR>  <GTURIR|BUTReverbDB>"
	exit 1
fi

MESH2IR_INPUT_DATA_DIR=$1
GENERATED_RIRS_DIR=$2
DATASET_NAME=$3 # GTURIR | BUTReverbDB
METADATA_DIR_NAME="METADATA_$DATASET_NAME"

if [ ! -d "$MESH2IR_INPUT_DATA_DIR" ]
then
        echo "Data directory $MESH2IR_INPUT_DATA_DIR does not exist ..."
        exit 1
fi

mkdir -p $GENERATED_RIRS_DIR/$METADATA_DIR_NAME/{Paths,Meshes,Embeddings}

cp ../EVALUATION_DATA-PREPARE-MESH2IR/ROOM_MESHES/$DATASET_NAME/*.obj $GENERATED_RIRS_DIR/$METADATA_DIR_NAME/Meshes


cd evaluate

if [ ! -f $GENERATED_RIRS_DIR/$METADATA_DIR_NAME/.mesh_metadata_is_generated ]
then
     python3 mesh_simplification.py  $GENERATED_RIRS_DIR $METADATA_DIR_NAME
     if [ $? = 0 ]
     then
         touch $GENERATED_RIRS_DIR/$METADATA_DIR_NAME/.mesh_metadata_is_generated
     else
	 echo "ERROR: Could Not Generate Simplified Meshes"
	 exit 1
     fi
fi

if [ ! -f $GENERATED_RIRS_DIR/$METADATA_DIR_NAME/.graph_metadata_is_generated ]
then
     python3 graph_generator.py  $GENERATED_RIRS_DIR $METADATA_DIR_NAME
     if [ $? = 0 ]
     then
         touch $GENERATED_RIRS_DIR/$METADATA_DIR_NAME/.graph_metadata_is_generated
     else
	 echo "ERROR: Could Not Generate Graphs"
	 exit 1
     fi
fi

if [ ! -f $GENERATED_RIRS_DIR/$METADATA_DIR_NAME/.embedding_pickle_files_are_generated_for_$DATASET_NAME ]
then
     python3 embed_generator.py $MESH2IR_INPUT_DATA_DIR/RIR.pickle.dat $GENERATED_RIRS_DIR $DATASET_NAME $METADATA_DIR_NAME
     if [ $? = 0 ]
     then
         touch $GENERATED_RIRS_DIR/$METADATA_DIR_NAME/.embedding_pickle_files_are_generated_for_$DATASET_NAME
     else
	 echo "ERROR: Could Not Generate Emdeddings"
	 exit 1
     fi
fi


CURRENT_DIR=$(pwd)

cd Models

ls *pth* > /dev/null
if [ $? != 0 ]
then
        echo "There is no *.pth file in the generate directory, copy the netG_epoch_*.pth and meshnet_*.pth file you trained."
        echo "This file may be found as output/*/Model_RT/*.pth after training the model "
	 exit 1
fi

./merge.sh

cd $CURRENT_DIR

if [ ! -f $GENERATED_RIRS_DIR/$METADATA_DIR_NAME/.rirs_are_generated_for_$DATASET_NAME ]
then
     python3 evaluate.py $GENERATED_RIRS_DIR $METADATA_DIR_NAME
     if [ $? = 0 ]
     then
         touch  $GENERATED_RIRS_DIR/$METADATA_DIR_NAME/.rirs_are_generated_for_$DATASET_NAME
     else
	 echo "ERROR: Could Not Generate RIRs"
	 exit 1
     fi
fi


