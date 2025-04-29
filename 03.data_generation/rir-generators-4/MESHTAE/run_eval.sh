#!/bin/bash

#./run_eval.sh  $HOME/RIR_DATA/GTU-RIR-1.0/data/single-speaker $HOME/RIR_REPORT/GTURIR/MESH2IR-MSE GTURIR
#./run_eval.sh /home/mpekmezci/RIR_DATA/GTU-RIR-1.0/data/single-speaker/ /home/mpekmezci/RIR_REPORT/GTURIR//MESHTAE GTURIR

if [ $# -lt 2 ]
then
	echo "Usage $0 <MESH2IR_INPUT_DATA_DIR>  <GENERATED_RIRS_DIR>  <GTURIR|BUTReverbDB> "
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


if [ ! -f $GENERATED_RIRS_DIR/$METADATA_DIR_NAME/.embedding_pickle_files_are_generated_for_$DATASET_NAME ]
then
     echo python3 embed_generator.py $MESH2IR_INPUT_DATA_DIR/RIR.pickle.dat $GENERATED_RIRS_DIR $DATASET_NAME $METADATA_DIR_NAME
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

#echo "1. Change the netG.pth file"
#echo "2. Change the mesh_net.pth file"
#echo "3. Change the number of nodes in the RIR_s1.yml file" 
#echo "4. Change the number of heads h in the mesh_model.py file" 
#echo "Press Enter When Done"
#read 

echo " Set the MAX_FACE_COUNT and NUMBER_OF_TRANSFORMER_HEADS in RIR_s1.yml corresponding your model file !"

if [ ! -f $GENERATED_RIRS_DIR/$METADATA_DIR_NAME/.rirs_are_generated_for_$DATASET_NAME ]
then
     echo python3 evaluate.py $GENERATED_RIRS_DIR $METADATA_DIR_NAME 
     python3 evaluate.py $GENERATED_RIRS_DIR $METADATA_DIR_NAME 
     if [ $? = 0 ]
     then
         touch  $GENERATED_RIRS_DIR/$METADATA_DIR_NAME/.rirs_are_generated_for_$DATASET_NAME
     else
	 echo "ERROR: Could Not Generate RIRs"
	 exit 1
     fi
fi


