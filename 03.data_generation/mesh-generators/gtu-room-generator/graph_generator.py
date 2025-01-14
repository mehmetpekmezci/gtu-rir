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
import glob


if len(sys.argv) < 2 :
   print("Usage : python3 graph_generator.py <MESH_DATA_DIR> ")
   exit(1)
else:
     MESH_DATA_DIR=sys.argv[1]
     pre_transform = torch_geometric.transforms.FaceToEdge();
     for full_mesh_path in glob.glob(MESH_DATA_DIR+'/*.obj'):
        graph_path = full_mesh_path[0:len(full_mesh_path)-4] +".pickle"
        if(os.path.exists(full_mesh_path) and not os.path.exists(graph_path)):
            print("graph_path ",graph_path)
            mesh = read_ply(full_mesh_path);
            graph =pre_transform(mesh);
            with open(graph_path, 'wb') as f:
              pickle.dump(graph, f, protocol=2)
