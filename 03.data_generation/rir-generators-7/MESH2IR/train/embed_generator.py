import os
import json
import numpy as np
import random
import argparse
import pickle
import math
import sys
#embeddings = [mesh_path,RIR_path,source,receiver]

training_embedding_list=[]
validation_embedding_list=[]

#path = "dataset/"
path = str(sys.argv[1]).strip()
mesh_folders = os.listdir(path)
# num_counter = 9
# temp_counter = 0 
# print("len folders ",len(mesh_folders))

print(f"searching for folders in {path}")

for folder in mesh_folders:
	# mesh_path = folder +"/" + folder +".obj"
	mesh_path = folder + "/" + folder +".pickle"
	RIR_folder  = path + "/" + folder +"/hybrid"

	if(os.path.exists(RIR_folder)):
		json_path = RIR_folder +"/sim_config.json"
		json_file = open(json_path)
		data = json.load(json_file)
		# receivers = len(data['receivers'])

		# if(receivers<(num_counter+temp_counter)):
		# 	num_receivers =receivers #len(data['receivers'])
		# 	temp_counter = temp_counter + (num_counter - receivers)
		# else:
		# 	num_receivers = num_counter+temp_counter
		# 	temp_counter = 0

		num_receivers = len(data['receivers'])
		num_sources = len(data['sources'])

		# print("num_receivers  ", num_receivers,"   num_sources  ", num_sources)
		for n in range(num_receivers):
			for s in range(num_sources):
				source = data['sources'][s]['xyz']
				receiver = data['receivers'][n]['xyz']
				RIR_name = "L"+str(data['sources'][s]['name'][1:]) + "_R"  + str(data['receivers'][n]['name'][1:]).zfill(4)+".wav"
				RIR_path = folder +"/hybrid/" + RIR_name
				full_RIR_path = path+"/"+ RIR_path
				if(os.path.exists(full_RIR_path)):
                                  print(full_RIR_path)
                                  embeddings = [mesh_path,RIR_path,source,receiver]
                                  if folder.startswith('f') or folder.startswith('e') or folder.startswith('d') : # each having 300 , 900/5000 makes 18 percent of data will be validation data.
                                     validation_embedding_list.append(embeddings)
                                  else:
                                     training_embedding_list.append(embeddings)


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

training_embeddings_pickle =path+"/training.embeddings.pickle"
with open(training_embeddings_pickle, 'wb') as f:
	pickle.dump(training_embedding_list, f, protocol=2)

validation_embeddings_pickle =path+"/validation.embeddings.pickle"
with open(validation_embeddings_pickle, 'wb') as f:
	pickle.dump(validation_embedding_list, f, protocol=2)




