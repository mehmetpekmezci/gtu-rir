
if [ ! -d $HOME/workspace-python/GWA ]
then
    mkdir -p $HOME/workspace-python/GWA
    git clone https://github.com/GAMMA-UMD/GWA $HOME/workspace-python/GWA
    git clone https://github.com/libigl/libigl.git $HOME/workspace-python/libigl
    ## Build libigl :
    cd $HOME/workspace-python/libigl
    mkdir build
    cd build
    cmake  -DCMAKE_BUILD_TYPE=Release  ..
    make -j8
    sudo make install
    cd $HOME/workspace-python/
    git clone https://github.com/libigl/libigl-python-bindings
    cd libigl-python-bindings
    sudo python3 setup.py develop
    # run GWA/tools/json2obj.py
fi

if [ ! -d /data.ext4/mpekmezci/3D-FRONT/outputs ]
then
    mkdir /data.ext4/mpekmezci/3D-FRONT/outputs
    cd $HOME/workspace-python/GWA/tools
    ln -s /data.ext4/mpekmezci/3D-FRONT/outputs
    ln -s /data.ext4/mpekmezci/3D-FRONT/3D-FRONT
    ln -s /data.ext4/mpekmezci/3D-FRONT/3D-FRONT-texture
    ln -s /data.ext4/mpekmezci/3D-FRONT/3D-FUTURE-model
    python3 assign_mats.py
    # run python3 assign_mats.py
    pip3 install sentence-transformers
    ln -s ../files/acoustic_absorptions.json
    python3 assign_mats.py --obj_path ./outputs
fi

if [ ! -f /data.ext4/mpekmezci/3D-FRONT/outputs/.sim_config.json.is.generated ]
then
    python3 generate_sim_config_json_from_data_stats_csv.py /data.ext4/mpekmezci/GWA/GWA_Dataset_full/ /data.ext4/mpekmezci/3D-FRONT/outputs/
    touch /data.ext4/mpekmezci/3D-FRONT/outputs/.sim_config.json.is.generated
fi

if [ ! -f /data.ext4/mpekmezci/3D-FRONT/outputs/.hybrid.dirs.created ]
then
    cd /data.ext4/mpekmezci/3D-FRONT/outputs
    for i in *-*
    do
	    mkdir $i/hybrid
	    mv    $i/* $i/hybrid 2>/dev/null
	    if [ -d /data.ext4/mpekmezci/GWA/GWA_Dataset_full/$i ]
	    then
	         mv    /data.ext4/mpekmezci/GWA/GWA_Dataset_full/$i/*  $i/hybrid
	    fi
    done
    touch /data.ext4/mpekmezci/3D-FRONT/outputs/.hybrid.dirs.created
fi

if [ ! -f /data.ext4/mpekmezci/3D-FRONT/outputs/.mesh_simplify ]
then
    cd /data.ext4/mpekmezci/3D-FRONT/outputs
    cp /home/mpekmezci/workspace-python/room_impulse_response_phd_thessis/src/rir-measurement/09.mesh2ir-cross-check/MESH2IR-DATASET-PREPARE-FOR-TRAINING/mesh_simplification.py .
    cp /home/mpekmezci/workspace-python/room_impulse_response_phd_thessis/src/rir-measurement/09.mesh2ir-cross-check/MESH2IR-DATASET-PREPARE-FOR-TRAINING/graph_generator.py .
    cp /home/mpekmezci/workspace-python/room_impulse_response_phd_thessis/src/rir-measurement/09.mesh2ir-cross-check/MESH2IR-DATASET-PREPARE-FOR-TRAINING/sample.mtl .

    for i in *-*
    do
	    echo "#############  $i #################"
	    python3 mesh_simplification.py $i/hybrid/house.obj $i/$i.obj
	    cp sample.mtl $i/$i.obj.mtl
	    python3 graph_generator.py $i/$i.obj

    done
    touch /data.ext4/mpekmezci/3D-FRONT/outputs/.mesh_simplify
fi 
