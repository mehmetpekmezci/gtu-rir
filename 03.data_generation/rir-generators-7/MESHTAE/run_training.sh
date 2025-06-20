#!/bin/bash


echo rm train/pre-trained-models/mesh_embeddings.pickle
ls -alrt   train/pre-trained-models/
echo "###"
echo "###"
echo "###"
echo "###"
echo "ATTENTION PLEASE :"
echo 
echo "If you changed the geometric model , please delete the file train/pre-trained-models/mesh_embeddings.pickle"
echo 
echo "###"
echo "###"
echo "###"
sleep 15
#rm train/pre-trained-models/mesh_embeddings.pickle



CURRENT_DIR=$(pwd)

DEFAULT_MAIN_DATA_DIR=$HOME/RIR_DATA

if [ "$MESH2IR_TRAINING_DATA" = "" ]
then
	echo "MESH2IR_TRAINING_DATA env. var. is not defined"
        export MESH2IR_TRAINING_DATA=$DEFAULT_MAIN_DATA_DIR/MESH2IR_TRAINING_DATA
        echo "MESH2IR_TRAINING_DATA=$MESH2IR_TRAINING_DATA"
fi

if [ ! -f "train/pre-trained-models/gae_mesh_net_trained_model.pth" ]
then
	echo "Mesh model is not trained, first mesh model will be trained, then RIR GAN model will be trained"
fi

if [ ! -f "$DEFAULT_MAIN_DATA_DIR/APARTMENT_MESH/synthetic_geometric_embeddings.pickle" ]
then
        cd $CURRENT_DIR/train/
	python3 synthetic_geometric_embed_generator.py $DEFAULT_MAIN_DATA_DIR/APARTMENT_MESH
else
	echo "$DEFAULT_MAIN_DATA_DIR/APARTMENT_MESH/synthetic_geometric_embeddings.pickle file is already prepared ..."
fi

if [ ! -d "$MESH2IR_TRAINING_DATA/3D-FRONT/outputs" ]
then
	cd ../TRAINING_DATA-PREPARE-MESH2IR/
	./prepare_dataset.sh
else
	echo "$MESH2IR_TRAINING_DATA/3D-FRONT/outputs training data is already prepared ..."
fi

if [ ! -f "$MESH2IR_TRAINING_DATA/3D-FRONT/outputs/training.embeddings.pickle" ]
then
        cd $CURRENT_DIR/train/
	echo  python3 embed_generator.py $MESH2IR_TRAINING_DATA/3D-FRONT/outputs
	python3 embed_generator.py $MESH2IR_TRAINING_DATA/3D-FRONT/outputs
else
	echo "$MESH2IR_TRAINING_DATA/3D-FRONT/outputs/training.embeddings.pickle file  is already prepared ..."
fi

if [ ! -f $MESH2IR_TRAINING_DATA/3D-FRONT/outputs/cache/.caching_is_done ]
then
        cd $CURRENT_DIR/train/
        mkdir -p $MESH2IR_TRAINING_DATA/3D-FRONT/outputs/cache
	echo python3 wav_pickle_cache_generator.py $MESH2IR_TRAINING_DATA/3D-FRONT/outputs
        python3 wav_pickle_cache_generator.py $MESH2IR_TRAINING_DATA/3D-FRONT/outputs
	if [ $? = 0 ]
	then
             touch $MESH2IR_TRAINING_DATA/3D-FRONT/outputs/cache/.caching_is_done
	fi
else
	echo "$MESH2IR_TRAINING_DATA/3D-FRONT/outputs/cache directories are already prepared ..."
fi

cd $CURRENT_DIR/train/MESH2IR 

#GPU_NO="0,1"
GPU_NO="1"

nvidia-smi | grep "NVIDIA GeForce 840M"
if [ $? = 0 ]
then
	GPU_NO=0
fi

mkdir -p /fastdisk/mpekmezci/temp
echo python3 main.py --cfg cfg/RIR_s1.yml --gpu $GPU_NO --data_dir $MESH2IR_TRAINING_DATA/3D-FRONT/outputs --synthetic_geometric_data_dir $DEFAULT_MAIN_DATA_DIR/APARTMENT_MESH

python3 main.py --cfg cfg/RIR_s1.yml --gpu $GPU_NO --data_dir $MESH2IR_TRAINING_DATA/3D-FRONT/outputs --synthetic_geometric_data_dir $DEFAULT_MAIN_DATA_DIR/APARTMENT_MESH
