#!/bin/bash



CURRENT_DIR=$(pwd)
TRAINING_DATA_PREPARATION_SCRIPTS_DIR=$CURRENT_DIR

DEFAULT_MAIN_DATA_DIR=$HOME/RIR_DATA

if [ "$MESH2IR_TRAINING_DATA" = "" ]
then
        #echo "MESH2IR_TRAINING_DATA env. var. is not defined"
        export MESH2IR_TRAINING_DATA=$DEFAULT_MAIN_DATA_DIR/MESH2IR_TRAINING_DATA
        echo "MESH2IR_TRAINING_DATA=$MESH2IR_TRAINING_DATA"
fi

if [ ! -d $MESH2IR_TRAINING_DATA/GWA ]
then
    mkdir -p $MESH2IR_TRAINING_DATA/GWA
    git clone https://github.com/GAMMA-UMD/GWA $MESH2IR_TRAINING_DATA/GWA
    git clone https://github.com/libigl/libigl.git $MESH2IR_TRAINING_DATA/libigl
    ## Build libigl :
    cd $MESH2IR_TRAINING_DATA/libigl
    mkdir build
    cd build
    cmake  -DCMAKE_BUILD_TYPE=Release  ..
    make -j8
    sudo make install
    cd $MESH2IR_TRAINING_DATA
    git clone https://github.com/libigl/libigl-python-bindings
    cd libigl-python-bindings
    sudo python3 setup.py develop
    # run GWA/tools/json2obj.py
else
   echo "$MESH2IR_TRAINING_DATA/GWA directory already exists ..."
fi


if [ ! -d $MESH2IR_TRAINING_DATA/3D-FRONT ]
then
	echo "Download MANUALLY , 3D-FRONT data from https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset and extract it in the direcotry $MESH2IR_TRAINING_DATA/3D-FRONT "
	echo ""
	echo "This download requires registration and login, find 'TC Lab: 3D-FRONT Dataset' in the page"
	echo "The download links get available after they accept your reistration request and login ( Which may take 2 weeks)"
	echo "The extracted directory will contain 3 sub directories : 3D-FRONT  3D-FRONT-texture   3D-FUTURE-model"
	exit 1
fi

if [ ! -d $MESH2IR_TRAINING_DATA/GWA/GWA_Dataset_full ]
then
	echo "Downloading GWA_Dataset_full.zip  : THIS FILE IS 500GB !!!"
	echo "When we extract zip file ,  GWA_Dataset_full directory contains stats.csv and hash coded directories like 99fbb361-8436-4e2c-9ba9-f9293a0062e4"
	cd $MESH2IR_TRAINING_DATA/GWA
	echo wget https://obj.umiacs.umd.edu/gamma-datasets/GWA_Dataset.zip
	wget https://obj.umiacs.umd.edu/gamma-datasets/GWA_Dataset.zip
        echo unzip GWA_Dataset.zip
        unzip GWA_Dataset.zip
        if [ ! -d $MESH2IR_TRAINING_DATA/GWA/GWA_Dataset_full ]
        then
		echo "Problem downloading or extracting GWA_Dataset.zip"
		exit 1
	fi
fi

if [ ! -d $MESH2IR_TRAINING_DATA/3D-FRONT/outputs ]
then
    mkdir $MESH2IR_TRAINING_DATA/3D-FRONT/outputs
    cd $MESH2IR_TRAINING_DATA/GWA/tools
    ln -s $MESH2IR_TRAINING_DATA/3D-FRONT/outputs
    ln -s $MESH2IR_TRAINING_DATA/3D-FRONT/3D-FRONT
    ln -s $MESH2IR_TRAINING_DATA/3D-FRONT/3D-FRONT-texture
    ln -s $MESH2IR_TRAINING_DATA/3D-FRONT/3D-FUTURE-model
    python3 assign_mats.py
    # run python3 assign_mats.py
    pip3 install sentence-transformers
    ln -s ../files/acoustic_absorptions.json
    python3 assign_mats.py --obj_path ./outputs

else
    echo "$MESH2IR_TRAINING_DATA/3D-FRONT/outputs directory already exists ..."
fi

if [ ! -f $MESH2IR_TRAINING_DATA/3D-FRONT/outputs/.sim_config.json.is.generated ]
then
    python3 generate_sim_config_json_from_data_stats_csv.py $MESH2IR_TRAINING_DATA/GWA/GWA_Dataset_full/ $MESH2IR_TRAINING_DATA/3D-FRONT/outputs/
    touch $MESH2IR_TRAINING_DATA/3D-FRONT/outputs/.sim_config.json.is.generated
else
    echo "python3 generate_sim_config_json_from_data_stats_csv.py $MESH2IR_TRAINING_DATA/GWA/GWA_Dataset_full/ $MESH2IR_TRAINING_DATA/3D-FRONT/outputs/ command already executed, no need to execute again"
fi


if [ ! -f $MESH2IR_TRAINING_DATA/3D-FRONT/outputs/.hybrid.dirs.created ]
then
    cd $MESH2IR_TRAINING_DATA/3D-FRONT/outputs
    for i in *-*
    do
            mkdir $i/hybrid
            mv    $i/* $i/hybrid 2>/dev/null
            if [ -d $MESH2IR_TRAINING_DATA/GWA/GWA_Dataset_full/$i ]
            then
                 mv    $MESH2IR_TRAINING_DATA/GWA/GWA_Dataset_full/$i/*  $i/hybrid
            fi
    done
    touch $MESH2IR_TRAINING_DATA/3D-FRONT/outputs/.hybrid.dirs.created
else
    echo "Hybrid directories are already ccreated."
fi



if [ ! -f $MESH2IR_TRAINING_DATA/3D-FRONT/outputs/.mesh_simplify ]
then
    cd $MESH2IR_TRAINING_DATA/3D-FRONT/outputs
    cp $TRAINING_DATA_PREPARATION_SCRIPTS_DIR/mesh_simplification.py .
    cp $TRAINING_DATA_PREPARATION_SCRIPTS_DIR/graph_generator.py .
    cp $TRAINING_DATA_PREPARATION_SCRIPTS_DIR/sample.mtl .

    for i in *-*
    do
            echo "#############  $i #################"
            python3 mesh_simplification.py $i/hybrid/house.obj $i/$i.obj
            cp sample.mtl $i/$i.obj.mtl
            python3 graph_generator.py $i/$i.obj

    done
    touch $MESH2IR_TRAINING_DATA/3D-FRONT/outputs/.mesh_simplify
else
    echo "Meshes are already simplified ."
fi

#if [ ! -f $MESH2IR_TRAINING_DATA/3D-FRONT/outputs/.extract_room_dims ]
#then
#    cd $MESH2IR_TRAINING_DATA/3D-FRONT/outputs
#    cp $TRAINING_DATA_PREPARATION_SCRIPTS_DIR/room_dimension_extractor.py .
#    for i in *-*
#    do
#            echo "#############  $i #################"
#            python3 room_dimension_extractor.py $i/hybrid/house.obj $i/$i.room_dim.txt
#    done
#    touch $MESH2IR_TRAINING_DATA/3D-FRONT/outputs/.extract_room_dims
#else
#    echo "Room Dimensions are already extracted"
#fi
