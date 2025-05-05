#!/bin/bash

#echo /home/mpekmezci/workspace-python/gtu-rir/03.data_generation/rir-generators-4/MESHTAE/evaluate/evaluate.py icinde cakinti 3000 71 meshnet verdim


mkdir -p ~/RIR_REPORT/GTURIR/
mkdir -p ~/RIR_REPORT/BUTReverbDB
ln -sf ~/RIR_REPORT.backup.default/GTURIR/MESH2IR ~/RIR_REPORT/GTURIR/MESH2IR
ln -sf ~/RIR_REPORT.backup.default/BUTReverbDB/MESH2IR ~/RIR_REPORT/BUTReverbDB/MESH2IR

#for node in 3000 2500 2000 1500 500 50 10
for node in 1000
do
	#for head in 8 21 51 71
	for head in 4
	do
		if [ -f stop ]
		then
			echo "STOPPING"
			rm -f stop
			exit 1
		fi
                if [ ! -f  ~/workspace-python/gtu-rir/03.data_generation/rir-generators-4/MESHTAE/evaluate/Models/netG_GAN_${node}_nodes_${head}_heads*pth ]
                then
                        continue
                fi

		rm -f  ~/workspace-python/gtu-rir/03.data_generation/rir-generators-4/MESHTAE/train/pre-trained-models/mesh_embeddings.pickle
		perl -pi -e "s/NUMBER_OF_TRANSFORMER_HEADS:.*/NUMBER_OF_TRANSFORMER_HEADS: $head/"  ~/workspace-python/gtu-rir/03.data_generation/rir-generators-4/MESHTAE/evaluate/cfg/RIR_s1.yml
		perl -pi -e "s/MAX_FACE_COUNT:.*/MAX_FACE_COUNT: $node/"  ~/workspace-python/gtu-rir/03.data_generation/rir-generators-4/MESHTAE/evaluate/cfg/RIR_s1.yml

		rm -f  ~/workspace-python/gtu-rir/03.data_generation/rir-generators-4/RIRT/train/pre-trained-models/mesh_embeddings.pickle
		perl -pi -e "s/NUMBER_OF_TRANSFORMER_HEADS:.*/NUMBER_OF_TRANSFORMER_HEADS: $head/"  ~/workspace-python/gtu-rir/03.data_generation/rir-generators-4/RIRT/evaluate/cfg/RIR_s1.yml
		perl -pi -e "s/MAX_FACE_COUNT:.*/MAX_FACE_COUNT: $node/"  ~/workspace-python/gtu-rir/03.data_generation/rir-generators-4/RIRT/evaluate/cfg/RIR_s1.yml

		rm -Rf ~/RIR_REPORT/GTURIR/MESHTAE
		rm -Rf ~/RIR_REPORT/GTURIR/RIRT
		rm -Rf ~/RIR_REPORT/BUTReverbDB/MESHTAE
		rm -Rf ~/RIR_REPORT/BUTReverbDB/RIRT
		rm -Rf ~/RIR_REPORT/summary

		mkdir ~/RIR_REPORT/GTURIR/MESHTAE.node.$node.head.$head
		mkdir ~/RIR_REPORT/GTURIR/RIRT.node.$node.head.$head
		mkdir ~/RIR_REPORT/BUTReverbDB/MESHTAE.node.$node.head.$head
		mkdir ~/RIR_REPORT/BUTReverbDB/RIRT.node.$node.head.$head
		mkdir ~/RIR_REPORT/summary.node.$node.head.$head

		ln -s ~/RIR_REPORT/GTURIR/MESHTAE.node.$node.head.$head ~/RIR_REPORT/GTURIR/MESHTAE
		ln -s ~/RIR_REPORT/GTURIR/RIRT.node.$node.head.$head ~/RIR_REPORT/GTURIR/RIRT
		ln -s ~/RIR_REPORT/BUTReverbDB/MESHTAE.node.$node.head.$head ~/RIR_REPORT/BUTReverbDB/MESHTAE
		ln -s ~/RIR_REPORT/BUTReverbDB/RIRT.node.$node.head.$head ~/RIR_REPORT/BUTReverbDB/RIRT
		ln -s ~/RIR_REPORT/summary.node.$node.head.$head ~/RIR_REPORT/summary


		./run.sh
               
		echo " ####### NODE:$node HEAD:$head is DONE "
	done
done

