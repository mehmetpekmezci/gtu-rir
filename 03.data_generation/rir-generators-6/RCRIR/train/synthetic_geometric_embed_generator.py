import os
import json
import numpy as np
import random
import argparse
import pickle
import math
import sys
import glob
#embeddings = [mesh_path,RIR_path,source,receiver]

embedding_list=[]

#path = "dataset/"
path = str(sys.argv[1]).strip()

#embedding_list = glob.glob(path+"/*/floor_plan-*.pickle", recursive=True)
embedding_list = glob.glob(path+"/*/floor_plan-*.obj", recursive=True)

print("embedding_list", len(embedding_list))
filler = 128  - (len(embedding_list) % 128)
print(f"filler={filler}")
len_embed_list = len(embedding_list) -1

if filler > len(embedding_list) :
   while len(embedding_list) < 128 :    
         embedding_list.append(embedding_list[random.randint(0, len(embedding_list) -1)])

else :
   if(filler < 128):
      for i in range(filler):
         embedding_list.append(embedding_list[len_embed_list-filler+i])


embeddings_pickle =path+"/synthetic_geometric_embeddings.pickle"
with open(embeddings_pickle, 'wb') as f:
	pickle.dump(embedding_list, f, protocol=2)




