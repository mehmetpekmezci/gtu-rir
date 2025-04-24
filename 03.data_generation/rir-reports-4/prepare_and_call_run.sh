#!/bin/bash

if [ $# -lt 1 ]
then
	echo "Usage : $0 <gtu_rir_roomid>"
	echo "Example : $0 208"
	#echo "Example : $0 207"
	#echo "Example : $0 conferrence01"
	#echo "Example : $0 z04"
	exit 1
fi

roomno=$1

rm -f ~/RIR_DATA/GTURIR-ROOM-MESH/*

cd ~/workspace-python/gtu-rir/03.data_generation/mesh-generators/gtu-room-generator
./run.sample.sh

cd  ~/workspace-python/gtu-rir/03.data_generation/rir-reports-4

rm -rf ~/RIR_REPORT
     
for((i=0;i<5;i++))
do
  echo
  echo
  echo "#######################################################################################"
  echo "#######i=$i"
  echo "#######################################################################################"
  echo
  echo

  rm -f ~/RIR_REPORT
  rm -rf ~/RIR_REPORT.$i
  mkdir -p ~/RIR_REPORT.$i/GTURIR
  ln -s ~/RIR_REPORT.$i ~/RIR_REPORT
  rm ~/workspace-python/gtu-rir/03.data_generation/rir-generators-4/EVALUATION_DATA-PREPARE-MESH2IR/ROOM_MESHES/GTURIR/*.obj

  if [ $i == 4 ]
  then
      for defaultobjfile in ~/workspace-python/gtu-rir/03.data_generation/rir-generators-4/EVALUATION_DATA-PREPARE-MESH2IR/ROOM_MESHES/GTURIR/*$roomno*.default
      do
	    obj=$(echo $defaultobjfile | sed -e 's/.default//')
	    ln -s $defaultobjfile $obj
      done
  else
      for furnituredobjfile in ~/RIR_DATA/GTURIR-ROOM-MESH/gtu-cs-room-$roomno.mesh.$i.obj   
      do
	   #roomno=$(echo $furnituredobjfile| sed -e 's#.*/##'|cut -d. -f1|cut -d- -f4)
	   ln -s $furnituredobjfile ~/workspace-python/gtu-rir/03.data_generation/rir-generators-4/EVALUATION_DATA-PREPARE-MESH2IR/ROOM_MESHES/GTURIR/room-$roomno-freecad-mesh-Body.obj
      done
  fi

  echo "ls -l ~/workspace-python/gtu-rir/03.data_generation/rir-generators-4/EVALUATION_DATA-PREPARE-MESH2IR/ROOM_MESHES/GTURIR/"  
  ls -l ~/workspace-python/gtu-rir/03.data_generation/rir-generators-4/EVALUATION_DATA-PREPARE-MESH2IR/ROOM_MESHES/GTURIR/


  for node in 10 50 500 1500 2000 2500 3000
  do
	for head in 8 21 51 71
	do
		if [ -f stop ]
		then
			echo "STOPPING"
			rm -f stop
			exit 1
		fi
		if [ ! -f /fastdisk/mpekmezci/models/netG_GAN_${node}_nodes_${head}_heads.pth ]
		then
			continue
		fi
		rm -f  ~/workspace-python/gtu-rir/03.data_generation/rir-generators-4/MESHTAE/train/pre-trained-models/mesh_embeddings.pickle
		ls -l ~/workspace-python/gtu-rir/03.data_generation/rir-generators-4/MESHTAE/train/pre-trained-models/mesh_embeddings.pickle
		perl -pi -e "s/NUMBER_OF_TRANSFORMER_HEADS:.*/NUMBER_OF_TRANSFORMER_HEADS: $head/"  ~/workspace-python/gtu-rir/03.data_generation/rir-generators-4/MESHTAE/evaluate/cfg/RIR_s1.yml
		perl -pi -e "s/MAX_FACE_COUNT:.*/MAX_FACE_COUNT: $node/"  ~/workspace-python/gtu-rir/03.data_generation/rir-generators-4/MESHTAE/evaluate/cfg/RIR_s1.yml
		rm -Rf ~/RIR_REPORT/GTURIR/MESHTAE
		rm -Rf ~/RIR_REPORT/summary

		mkdir ~/RIR_REPORT/GTURIR/MESHTAE.node.$node.head.$head
		mkdir ~/RIR_REPORT/summary.node.$node.head.$head

		ln -s ~/RIR_REPORT/GTURIR/MESHTAE.node.$node.head.$head ~/RIR_REPORT/GTURIR/MESHTAE
		ln -s ~/RIR_REPORT/summary.node.$node.head.$head ~/RIR_REPORT/summary

                pwd
		./run.sh
               
		echo " ####### NODE:$node HEAD:$head is DONE "
	done
  done
done
