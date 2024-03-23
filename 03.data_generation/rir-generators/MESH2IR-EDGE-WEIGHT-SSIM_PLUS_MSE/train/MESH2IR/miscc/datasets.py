from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


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

from miscc.config import cfg

#embeddings = [mesh_path,RIR_path,source,receiver]
class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train',rirsize=4096): #, transform=None, target_transform=None):

        self.rirsize = rirsize
        self.data = []
        self.data_dir = data_dir       
        self.bbox = None
        
  
        self.embeddings = self.load_embedding(data_dir)

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


    # def get_graph(self, full_mesh_path):
    #     mesh = read_ply(full_mesh_path);
    #     pre_transform = torch_geometric.transforms.FaceToEdge();
    #     graph =pre_transform(mesh);
    #     # edge_index = graph['edge_index']
    #     # vertex_position = graph['pos']
        
    #     return graph #edge_index, vertex_position


    def get_graph(self, full_graph_path):
        
        with open(full_graph_path, 'rb') as f:
            graph = pickle.load(f)
        
        return graph #edge_index, vertex_position

    def load_embedding(self, data_dir):
        embedding_directory   = self.data_dir+'/embeddings.pickle'  
        with open(embedding_directory, 'rb') as f:
            embeddings = pickle.load(f)
        return embeddings


    def __getitem__(self, index):
      
        #print("__getitem__:index:"+str(index))

        graph_path,RIR_path,source_location,receiver_location,cos_theta,graph = self.embeddings[index]

        data_dir = self.data_dir

        full_graph_path = os.path.join(data_dir,graph_path)
        full_RIR_path  = os.path.join(data_dir,RIR_path)
        rir_dir_name=os.path.dirname(os.path.dirname(RIR_path))
        rir_basename_name=os.path.basename(RIR_path)
        source_receiver = source_location+receiver_location

        RIR=None

        #print('../cache/'+rir_dir_name+'/'+rir_basename_name+'.pickle')
        if os.path.exists('../cache/'+rir_dir_name+'/'+rir_basename_name+'.pickle'):
           with open('../cache/'+rir_dir_name+'/'+rir_basename_name+'.pickle', 'rb') as f:
            RIR = pickle.load(f)
            #print('loaded from : ../cache/'+rir_dir_name+'/'+rir_basename_name+'.pickle')
        else :
            full_RIR_path  = os.path.join(data_dir,RIR_path)
            RIR = self.get_RIR(full_RIR_path)


        embedding = np.array(source_receiver).astype('float32')

        #graph = self.get_graph(full_graph_path);
        
        graph.RIR = RIR
        graph.embeddings = embedding

        ## cos = -cos beacuse we are free of direcction. The other normalization -1,1 to 0,1  lke given below gives the same result as if we did not do anything.
        graph['edge_weights']=torch.from_numpy(np.abs(np.array(cos_theta)))

        ### graph['edge_weights']=torch.from_numpy(np.array(cos_theta))
        ### graph['edge_weights']=(graph['edge_weights']+1)/2 ## MP: for numerical stability, normalley cos_theta [-1,1] has negative values, but we cannot have negative values in edge_weights, so we add +1 and divide by 2.
                                                    ## as tihs is a linear operation (x+1/2)  , this will not effect the results.
 
        #print("__getitem__:graph:"+str(graph))

        # print("shape ", transpose_edge_index.shape)
        return graph
        
    def __len__(self):
        return len(self.embeddings)
