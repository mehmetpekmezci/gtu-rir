import os
import json
import numpy as np
import random
import argparse
import pickle
import math
import sys
from ray_direction_generator import generate_ray_directions
from ray_casting_blender import ray_cast ,clear_scene
import time
import bpy
import bmesh

main_path = str(sys.argv[1]).strip()
resolution_quotient = int(str(sys.argv[2]).strip()) # 180%RESOLUTION_QUOTIENT == 0
max_distance_in_a_room = int(str(sys.argv[3]).strip()) # max_distance_in_a_room

ray_directions=generate_ray_directions(RESOLUTION_QUOTIENT=resolution_quotient) # 180%RESOLUTION_QUOTIENT == 0
mesh_folders = os.listdir(main_path)


total=len(mesh_folders)
i=0



context = bpy.context
#vl = context.view_layer
scene = context.scene
depsgraph=context.evaluated_depsgraph_get()

for folder in mesh_folders:
 t1=time.time()
 mesh_path = folder +"/" + folder +".obj"
 RIR_folder  = main_path +'/'+ folder +"/hybrid"
 clear_scene()
 bmesh_object=bpy.ops.wm.obj_import(filepath=main_path+'/'+mesh_path)

 l=[]

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
     ray_cast_image=ray_cast(bmesh_object,scene,depsgraph,receiver,ray_directions,receiver_position_name,max_distance_in_a_room)
     l.append([ray_cast_image,n,0])
     #print(ray_cast_image)
     ## save image
   for s in range(num_sources):
       source = data['sources'][s]['xyz']
       loudspeaker_position_name="L"+str(s)
       ray_cast_image=ray_cast(bmesh_object,scene,depsgraph,source,ray_directions,loudspeaker_position_name,max_distance_in_a_room)
       #print(ray_cast_image)
       l.append([ray_cast_image,s,1])

   identical_l=[]
   for r1 in l:
      for r2 in l:
           if np.array_equal(r1[0],r2[0]) and (r1[1]!=r2[1] and r1[2]!=r1[2]):
           #if np.array_equal(r1[0],r2[0]) and (r1[1]!=r2[1] or r1[2]!=r1[2]):
           #if np.array_equal(r1[0],r2[0]) :
              identical_l.append([r1,r2])

   print(f"n_identical={len(identical_l)} / {len(l)*len(l)}  len(l)={len(l)} num_receivers={num_receivers} num_sources={num_sources} ")

       ### CHECK IF THE PREVIOUS VALUE ARE THE SAME
       ## save ray cast image as pickle
#       RIR_name = "L"+str(data['sources'][s]['name'][1:]) + "_R"  + str(data['receivers'][n]['name'][1:]).zfill(4)+".wav"
#       RIR_path = folder +"/hybrid/" + RIR_name
#       full_RIR_path = main_path+ RIR_path
#       if(os.path.exists(full_RIR_path)):
#         embeddings = [mesh_path,RIR_path,source,receiver]
 t2=time.time()
 i=i+1
 #if i%10 == 0 : 
 if i%1 == 0 : 
     print(f"###### folder={folder} {i}/{total} time:{t2-t1}")
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




