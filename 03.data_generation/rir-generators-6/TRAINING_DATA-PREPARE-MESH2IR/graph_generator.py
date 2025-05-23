import os
import json
import numpy as np
import random
import argparse
import pickle
import sys
import torch
import torch_geometric
from torch_geometric.io import read_ply

if len(sys.argv) < 2 :
   print("Usage : python3 graph_generator.py <SOURCE_OBJ_FILE> ")
   exit(1)
else:
#        if(file.endswith(".obj")):
		full_mesh_path = sys.argv[1]
		graph_path = full_mesh_path[0:len(full_mesh_path)-4] +".pickle"
		print("graph_path ",graph_path)
		if(os.path.exists(full_mesh_path)):
			print("came here ")
			mesh = read_ply(full_mesh_path);
			pre_transform = torch_geometric.transforms.FaceToEdge();
			graph =pre_transform(mesh);

			with open(graph_path, 'wb') as f:
				pickle.dump(graph, f, protocol=2)
		
