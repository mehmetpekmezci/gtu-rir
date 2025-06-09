import time
import torch.utils.data as data
import os
import numpy as np
import torch
import librosa
import sys
import bpy
import bmesh
from miscc.config import cfg
from PIL import Image
from contextlib import redirect_stdout
import io
import pickle
import trimesh
import pygame


class RIRDataset(data.Dataset):
    def __init__(self,data_dir,embeddings,split='train',rirsize=4096): 
        self.rirsize = rirsize
        self.data = []
        self.data_dir = data_dir       
        self.embeddings = embeddings

    def get_RIR(self, full_RIR_path):

        picklePath=full_RIR_path.replace(".wav",".pickle")
        if os.path.exists(picklePath):
            with open(picklePath, "rb") as f:
                  x = pickle.load(f)
            return x

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

        with open(picklePath, 'wb') as f:
            pickle.dump(RIR, f, protocol=2)

        return RIR

    def __getitem__(self, index):

        graph_path,RIR_path,source_location,receiver_location= self.embeddings[index]
        room_dims= self.mesh_embeddings(os.path.join(self.data_dir,graph_path))
        data = {}
        data["RIR"] =  self.get_RIR(os.path.join(self.data_dir,RIR_path))
        data["source_and_receiver_and_roomdims"] =  np.concatenate((np.array(source_location).astype('float32'),np.array(receiver_location).astype('float32'),np.array( room_dims).astype('float32')))

        return data
        
    def __len__(self):
        return len(self.embeddings)

    def loadMesh(self,full_graph_path):
        mesh = trimesh.load_mesh(full_graph_path)
        v=np.array(mesh.vertices)
        max_x=np.max(v[:,0])
        min_x=np.min(v[:,0])
        max_y=np.max(v[:,1])
        min_y=np.min(v[:,1])
        max_z=np.max(v[:,2])
        min_z=np.min(v[:,2])
        DEPTH=(max_x-min_x)
        WIDTH=(max_z-min_z)
        HEIGHT=(max_y-min_y)
        return DEPTH,WIDTH,HEIGHT
    

            
    def mesh_embeddings(self,full_graph_path):
        full_graph_path=full_graph_path.replace(".pickle",".obj")
        ROOM_DEPTH,ROOM_WIDTH,ROOM_HEIGHT=self.loadMesh(full_graph_path)
        return  [ROOM_DEPTH,ROOM_WIDTH,ROOM_HEIGHT]

 

