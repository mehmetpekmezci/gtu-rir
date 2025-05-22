import os
import json
import numpy as np
import random
import argparse
import pickle
import math
import sys
#embeddings = [mesh_path,RIR_path,source,receiver]

embedding_list=[]

#path = "dataset/"
path = str(sys.argv[1]).strip()
mesh_folders = os.listdir(path)
print(f"len(mesh_folders)={len(mesh_folders)}")
# num_counter = 9
# temp_counter = 0 
# print("len folders ",len(mesh_folders))



def get_graph(full_graph_path):

        with open(full_graph_path, 'rb') as f:
            graph = pickle.load(f)

        return graph #edge_index, vertex_position

total_mech_folders_count=len(mesh_folders)
counter=0

for folder in mesh_folders:
        counter+=1
	# mesh_path = folder +"/" + folder +".obj"
        mesh_path = folder +"/" + folder +".pickle"
        RIR_folder  = path + '/'+ folder +"/hybrid"

        if counter % 100 == 0 :
           print(f'{counter}/{total_mech_folders_count}')

        print(f"if(os.path.exists({RIR_folder}) and os.path.exists({path+'/'+mesh_path}))")
        if(os.path.exists(RIR_folder) and os.path.exists(path+'/'+mesh_path) and os.path.exists(RIR_folder+"/sim_config.json")):
                print(f"generating embeddings for {RIR_folder}") 
                full_graph_path = os.path.join(path,mesh_path)
                #graph = get_graph(full_graph_path);
                json_path = RIR_folder +"/sim_config.json"
                json_file = open(json_path)
                data = json.load(json_file)
                # receivers = len(data['receivers'])

                # if(receivers<(num_counter+temp_counter)):
                #         num_receivers =receivers #len(data['receivers'])
                #         temp_counter = temp_counter + (num_counter - receivers)
                # else:
                #         num_receivers = num_counter+temp_counter
                #         temp_counter = 0

                num_receivers = len(data['receivers'])
                num_sources = len(data['sources'])

                # print("num_receivers  ", num_receivers,"   num_sources  ", num_sources)
                for s in range(num_sources):

                        source = data['sources'][s]['xyz']
                        #temp=source[1]
                        #source[1]=str(-float(source[2]))
                        #source[2]=temp

                        #print(f'{s}}')
                        for n in range(num_receivers):
                                receiver = data['receivers'][n]['xyz'] # MP: xyz diye yazmislar ama icinde xzy olarak bulunyor
                # coordinate lar xzy ama zaten mesh de xzy olark geliyor. onun icin bununla oynama :
                # diye yazmistim ama meshi export edip bakinca asagidaki donusumu yapmak mantilki geldi.
                                #temp=receiver[1]
                                #receiver[1]=str(-float(receiver[2]))
                                #receiver[2]=temp

                                RIR_name = "L"+str(data['sources'][s]['name'][1:]) + "_R"  + str(data['receivers'][n]['name'][1:]).zfill(4)+".wav"
                                RIR_path = folder +"/hybrid/" + RIR_name
                                full_RIR_path = path+'/'+ RIR_path
                                if(os.path.exists(full_RIR_path)):
                                        #embeddings = [mesh_path,RIR_path,source,receiver,cos_theta,graph]
                                        embeddings = [mesh_path,RIR_path,source,receiver]
                                        embedding_list.append(embeddings)



print("embdiing_list:", len(embedding_list))
filler = 128  - (len(embedding_list) % 128)
len_embed_list = len(embedding_list) -1
if(filler < 128):
	for i in range(filler):
		embedding_list.append(embedding_list[len_embed_list-filler+i])

# embed_count = 128*2
# embedding_list = embedding_list[0:embed_count]
# print("embdiing_list12345", len(embedding_list))

embeddings_pickle =path+"/embeddings.pickle"
with open(embeddings_pickle, 'wb') as f:
	pickle.dump(embedding_list, f, protocol=2)




