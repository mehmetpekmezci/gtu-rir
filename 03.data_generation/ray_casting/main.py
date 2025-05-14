import os
import json
import numpy as np
import random
import argparse
import pickle
import math
import sys
from ray_direction_generator import generate_ray_directions
from ray_casting_blender import ray_cast


main_path = str(sys.argv[1]).strip()
resolution_quotient = int(str(sys.argv[2]).strip()) # 180%RESOLUTION_QUOTIENT == 0
max_distance_in_a_room = int(str(sys.argv[3]).strip()) # max_distance_in_a_room

ray_directions=generate_ray_directions(RESOLUTION_QUOTIENT=resolution_quotient) # 180%RESOLUTION_QUOTIENT == 0
mesh_folders = os.listdir(main_path)


total=len(mesh_folders)
i=0
for folder in mesh_folders:
 mesh_path = folder +"/" + folder +".obj"
 RIR_folder  = main_path +'/'+ folder +"/hybrid"
 
 print(f"######2 RIR_folder={RIR_folder}")
 if(os.path.exists(RIR_folder)):
   print(f"######2 folder={folder} {i}/{total}")
   json_path = RIR_folder +"/sim_config.json"
   json_file = open(json_path)
   data = json.load(json_file)
   num_receivers = len(data['receivers'])
   num_sources = len(data['sources'])
   for n in range(num_receivers):
     receiver = data['receivers'][n]['xyz']
     receiver_position_name="R"+str(n)
     ray_cast(main_path+'/'+mesh_path,receiver,ray_directions,receiver_position_name,max_distance_in_a_room)
     for s in range(num_sources):
       source = data['sources'][s]['xyz']
       loudspeaker_position_name="L"+str(n)
       ray_cast(main_path+'/'+mesh_path,source,ray_directions,loudspeaker_position_name,max_distance_in_a_room)
#       RIR_name = "L"+str(data['sources'][s]['name'][1:]) + "_R"  + str(data['receivers'][n]['name'][1:]).zfill(4)+".wav"
#       RIR_path = folder +"/hybrid/" + RIR_name
#       full_RIR_path = main_path+ RIR_path
#       if(os.path.exists(full_RIR_path)):
#         embeddings = [mesh_path,RIR_path,source,receiver]
 i=i+1
 #if i%10 == 0 : 
 if i%1 == 0 : 
   print(f"###### folder={folder} {i}/{total}")
 if i%10 == 0 : 
   sys.exit(0)


print("validation_embdding_list", len(validation_embedding_list))
print("training_embdding_list", len(training_embedding_list))

filler = 128  - (len(validation_embedding_list) % 128)
len_embed_list = len(validation_embedding_list) -1
if(filler < 128):
	for i in range(filler):
		validation_embedding_list.append(validation_embedding_list[len_embed_list-filler+i])

filler = 128  - (len(training_embedding_list) % 128)
len_embed_list = len(training_embedding_list) -1
if(filler < 128):
	for i in range(filler):
		training_embedding_list.append(training_embedding_list[len_embed_list-filler+i])

# embed_count = 128*2
# embedding_list = embedding_list[0:embed_count]
# print("embdiing_list12345", len(embedding_list))

training_embeddings_pickle =main_path+"/training.embeddings.pickle"
with open(training_embeddings_pickle, 'wb') as f:
	pickle.dump(training_embedding_list, f, protocol=2)

validation_embeddings_pickle =main_path+"/validation.embeddings.pickle"
with open(validation_embeddings_pickle, 'wb') as f:
	pickle.dump(validation_embedding_list, f, protocol=2)




