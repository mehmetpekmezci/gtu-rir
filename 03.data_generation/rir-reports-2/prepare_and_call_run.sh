#!/bin/bash

#for node in 1500 2000 2500
for node in 2000 2500
do
	for head in 8 21 51
	do
		cd ~/workspace-python/gtu-rir/03.data_generation/rir-generators-4/MESHTAE/evaluate
		perl -pi -e "s/h=..?,/h=$head,/" model_mesh.py
		perl -pi -e "s/MAX_FACE_COUNT:.*/MAX_FACE_COUNT: $node/"  cfg/RIR_s1.yml
                cd Models
		rm -f mesh_net.pth
		ln -s /fastdisk/mpekmezci/models/mesh_net_transformer_${node}_nodes_${head}_heads.pth mesh_net.pth
		rm -f netG.pth
		ln -s /fastdisk/mpekmezci/models/netG_GAN_${node}_nodes_${head}_heads.pth netG.pth
                cd /home/mpekmezci/workspace-python/gtu-rir/03.data_generation/rir-reports-2
		./run.sh
                cd /home/mpekmezci/RIR_REPORT
		mv summary /fastdisk/mpekmezci/rir-reports/summary.node.$node.head.$head
		mv GTURIR/MESHTAE /fastdisk/mpekmezci/rir-reports/GTU-RIR.MESHTAE.node.$node.head.$head
		mv BUTReverbDB/MESHTAE /fastdisk/mpekmezci/rir-reports/BUTReverbDB.MESHTAE.node.$node.head.$head
	done
done

