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
# num_counter = 9
# temp_counter = 0 
# print("len folders ",len(mesh_folders))

graph_edge_unit_vectors={}
source_unit_vectors={}


def get_graph(full_graph_path):

        with open(full_graph_path, 'rb') as f:
            graph = pickle.load(f)

        return graph #edge_index, vertex_position

def computeEdgeUnitVectors(graph):
        edge_vectors=graph['pos'][graph['edge_index'][0]]-graph['pos'][graph['edge_index'][1]]
        #print('edge_vectors')
        #print(edge_vectors)
        edge_vector_magnitudes=np.sqrt(np.dot(edge_vectors,edge_vectors.T).diagonal())
        #print('edge_vector_magnitudes')
        #print(edge_vector_magnitudes)
        edge_unit_vectors=edge_vectors*np.reshape(1/edge_vector_magnitudes,(edge_vector_magnitudes.shape[0],1))
        #print('edge_unit_vectors')
        #print(edge_unit_vectors)
        return edge_unit_vectors



def computeSourceUnitVector(source_location):
         
        source_location=np.array(source_location).astype(np.float64)
        source_unit_vector=source_location/np.sqrt(np.dot(source_location,source_location))

        return source_unit_vector

def computeCosThetaBetweenEdgeAndSource(edge_unit_vectors,source_unit_vector):
         
        cos_theta=np.inner(edge_unit_vectors,source_unit_vector)

        return cos_theta


total_mech_folders_count=len(mesh_folders)
counter=0

for folder in mesh_folders:
        counter+=1
	# mesh_path = folder +"/" + folder +".obj"
        mesh_path = folder +"/" + folder +".pickle"
        RIR_folder  = path + '/'+ folder +"/hybrid"

        if counter % 100 == 0 :
           print(f'{counter}/{total_mech_folders_count}')

        if(os.path.exists(RIR_folder) and os.path.exists(path+'/'+mesh_path)):
                 
                full_graph_path = os.path.join(path,mesh_path)
                graph = get_graph(full_graph_path);
                edge_unit_vectors=computeEdgeUnitVectors(graph)
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
                        source_unit_vector=computeSourceUnitVector(source)
                        cos_theta=computeCosThetaBetweenEdgeAndSource(edge_unit_vectors,source_unit_vector)

                        #print(f'{s}}')
                        for n in range(num_receivers):
                                receiver = data['receivers'][n]['xyz']
                                RIR_name = "L"+str(data['sources'][s]['name'][1:]) + "_R"  + str(data['receivers'][n]['name'][1:]).zfill(4)+".wav"
                                RIR_path = folder +"/hybrid/" + RIR_name
                                full_RIR_path = path+'/'+ RIR_path
                                if(os.path.exists(full_RIR_path)):
                                        embeddings = [mesh_path,RIR_path,source,receiver,cos_theta,graph]
                                        embedding_list.append(embeddings)



print("embdiing_list", len(embedding_list))
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




