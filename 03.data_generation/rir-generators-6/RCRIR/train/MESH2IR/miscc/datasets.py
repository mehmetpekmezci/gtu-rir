from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pymeshlab as ml
import time
import torch.utils.data as data
# from PIL import Image
import soundfile as sf
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import pandas as pd
from scipy import signal

import torch
import torch_geometric
from torch_geometric.io import read_ply
import librosa

import io
import sys
import random
from miscc.config import cfg
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_dense_batch, dropout_adj
import scipy.sparse as sp
import traceback


from miscc.utils import save_mesh_as_obj,save_pos_face_as_obj,load_pickle, write_pickle


# using the modelnet40 as the dataset, and using the processed feature matrixes
import json
import random
from pathlib import Path
import numpy as np
import os
import torch
import torch.utils.data as data
import trimesh
from scipy.spatial.transform import Rotation
import pygem
from pygem import FFD
import copy
import csv

import math
import pyglet


#embeddings = [mesh_path,RIR_path,source,receiver]
class MeshDataset(data.Dataset):
    def __init__(self, data_dir,mesh_paths, train=True,augment=None): 
        self.data_dir = data_dir       
        self.mesh_paths = mesh_paths

        self.augments = []
        self.feats = ['area', 'face_angles', 'curvs', 'normal']
         
        if train and augment:
            self.augments = augment

    def __getitem__(self, index):

        label = 0
        full_mesh_path=os.path.join(self.data_dir,self.mesh_paths[index]).replace('.pickle','.obj')
        try:
          tirangle_coordinates,normals,centers,areas = load_mesh2(full_mesh_path, augments=self.augments,request=self.feats)
          tirangle_coordinates,normals,centers,areas = normalize_mesh_values(tirangle_coordinates,normals,centers,areas)
            #write_pickle(mesh_pickle_file_path,(tirangle_coordinates,normals,centers,areas))
        except:
          tirangle_coordinates,normals,centers,areas,full_mesh_path =  np.zeros((cfg.MAX_FACE_COUNT,9)),np.zeros((cfg.MAX_FACE_COUNT,3)),np.zeros((cfg.MAX_FACE_COUNT,3)),np.zeros((cfg.MAX_FACE_COUNT,1)),"ERRONOUS_MESH"

        return   tirangle_coordinates, normals,centers,areas, full_mesh_path

        
    def __len__(self):
        return len(self.mesh_paths)
            
            

class RIRDataset(data.Dataset):
    def __init__(self,data_dir,embeddings,split='train',rirsize=4096): 

        self.rirsize = rirsize
        self.data = []
        self.data_dir = data_dir       
        self.bbox = None
        
  
        self.embeddings = embeddings
        self.mesh_embeddings = mesh_embeddings

    def get_RIR(self, full_RIR_path):
        # wav,fs = sf.read(full_RIR_path) 
        wav,fs = librosa.load(full_RIR_path)
 
        # wav_resample = librosa.resample(wav,16000,fs)
        wav_resample = librosa.resample(wav,orig_sr=fs,target_sr=16000)

        length = wav_resample.size

        crop_length = 3968 #int(16384)
        if(length<crop_length):
            zeros = np.zeros(crop_length-length)
            std_value = np.std(wav_resample) * 10
            std_array = np.repeat(std_value,128)
            wav_resample_new = np.concatenate([wav_resample,zeros])/std_value
            RIR_original = np.concatenate([wav_resample_new,std_array])
        else:
            wav_resample_new = wav_resample[0:crop_length]
            std_value = np.std(wav_resample_new)  * 10
            std_array = np.repeat(std_value,128)
            wav_resample_new =wav_resample_new/std_value
            RIR_original = np.concatenate([wav_resample_new,std_array])

        resample_length = int(self.rirsize)
        
        RIR = RIR_original

        RIR = np.array([RIR]).astype('float32')

        return RIR

    def __getitem__(self, index):

        graph_path,RIR_path,source_location,receiver_location= self.embeddings[index]

        data_dir = self.data_dir

        full_graph_path = os.path.join(data_dir,graph_path)
        full_RIR_path  = os.path.join(data_dir,RIR_path)
        source_receiver = source_location+receiver_location

        RIR = self.get_RIR(full_RIR_path)

        data = {}
        
        data["RIR"] = RIR
        data["embeddings"] =  np.array(source_receiver).astype('float32')
        data["mesh_embeddings"] = self.mesh_embeddings[graph_path]

        return data
        
    def __len__(self):
        return len(self.embeddings)



def build_mesh_embeddings(data_dir,embeddings):
    for i in range(len(embeddings)):
        if i%100000 == 0 :
            print(f"{i}/{len(embeddings)}")
        graph_path,RIR_path,source_location,receiver_location= embeddings[i]
        full_graph_path = os.path.join(data_dir,graph_path)
        if graph_path not in  mesh_embeddings:
           full_mesh_path = full_graph_path.replace('.pickle','.obj')
           #triangle_coordinates,normals,centers,areas = load_mesh(full_mesh_path)
           triangle_coordinates,normals,centers,areas = load_mesh2(full_mesh_path)
           real_triangle_coordinates,real_normals,real_centers,real_areas = triangle_coordinates,normals,centers,areas
           triangle_coordinates,normals,centers,areas = normalize_mesh_values(triangle_coordinates,normals,centers,areas)
           triangle_coordinates=torch.autograd.Variable(torch.from_numpy(triangle_coordinates)).float()
           normals=torch.autograd.Variable(torch.from_numpy(normals)).float()
           centers=torch.autograd.Variable(torch.from_numpy(centers)).float()
           areas=torch.autograd.Variable(torch.from_numpy(areas)).float()
           faceDataDim=triangle_coordinates.shape[1]+centers.shape[1]+normals.shape[1]+areas.shape[1]
           faceData=torch.cat((triangle_coordinates,normals,centers,areas),1)
           faceData=faceData.unsqueeze(0).detach().cuda()
           faceData_predicted , latent_vector =  gae_mesh_net(faceData)
           mesh_embeddings[graph_path]=latent_vector.squeeze().detach().cpu()
 
    return mesh_embeddings

