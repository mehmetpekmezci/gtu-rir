from __future__ import print_function
from six.moves import range
from PIL import Image


import torch.backends.cudnn as cudnn
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch_geometric.loader import DataLoader

import numpy as np
import cupy as cp
from cupyx.scipy.signal import fftconvolve
import torchfile
import pickle


import soundfile as sf
import re
import math
from wavefile import WaveWriter, Format


import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil
import dateutil.tz

import librosa

from miscc.config import cfg, cfg_from_file
from miscc.datasets import load_mesh, normalize_mesh_values


generated_rirs_dir=str(sys.argv[1]).strip()
metadata_dirname=str(sys.argv[2]).strip()
metadata_dir = generated_rirs_dir+"/"+metadata_dirname




def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def load_network_stageI(netG_path,mesh_net_path,batch_size):
        from model import STAGE1_G, STAGE1_D
        from model_mesh import MESH_TRANSFORMER_AE
        netG = STAGE1_G()
        netG.apply(weights_init)

        print(netG)
       
        mesh_net = MESH_TRANSFORMER_AE()



        if netG_path!= '':
            state_dict = \
                torch.load(netG_path,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load from: ', netG_path)
       
        if mesh_net_path != '':
            state_dict = \
                torch.load(mesh_net_path,
                           map_location=lambda storage, loc: storage)
            mesh_net.load_state_dict(state_dict)
            print('Load from: ', mesh_net_path)


        
        #netG.cuda()
        #mesh_net.cuda()
        return netG, mesh_net

def get_mesh(full_mesh_path):
        
        tirangle_coordinates,normals,centers,areas = load_mesh(full_mesh_path)
        
        mesh={}
        
        tirangle_coordinates,normals,centers,areas = normalize_mesh_values(tirangle_coordinates,normals,centers,areas)
        mesh={}
        mesh["triangle_coordinates"]=tirangle_coordinates
        mesh["normals"]=normals
        mesh["centers"]=centers
        mesh["areas"]=areas
   
        
        return mesh

def load_embedding(data_dir):
    # embedding_filename   = '/embeddings.pickle'  
    with open(data_dir, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings


def evaluate():
    print(f"metadata_dir={metadata_dir}") 

    SCRIPT_DIR=os.path.dirname(os.path.realpath(__file__))
    
    print("This file is found in "+SCRIPT_DIR)
    
    cfg_from_file(SCRIPT_DIR+'/cfg/RIR_s1.yml')
    
    embedding_directory =metadata_dir+"/Embeddings/"
    mesh_directory = metadata_dir+"/Meshes"
    output_directory = generated_rirs_dir

    #netG_path = "Models/MESH2IR/netG_epoch_175.pth"
    #mesh_net_path = "Models/MESH2IR/mesh_net_epoch_175.pth"
    netG_path = "Models/netG.pth"
    mesh_net_path = "Models/gae_mesh_net_trained_model.pth"
    #gpus =[0,1]
    #gpus =[0]

    #batch_size = 256
    batch_size = 2
    fs = 16000


    if(not os.path.exists(output_directory)):
        os.mkdir(output_directory)

    netG, mesh_net = load_network_stageI(netG_path,mesh_net_path,batch_size)
    netG.eval()
    mesh_net.eval()


    #netG.to(device='cuda')
    #mesh_net.to(device='cuda')

    embedding_list = os.listdir(embedding_directory)
    
    for embed in embedding_list:
        embed_path = embedding_directory + "/"+embed
        embeddings = load_embedding(embed_path)
        embed_name = embed[0:len(embed)-7]
        output_embed  = output_directory+'/'+embed_name
        if(not os.path.exists(output_embed)):
            os.mkdir(output_embed)

        print("embed_name   ",output_embed)

        mesh_obj,folder_name,wave_name,source_location,receiver_location = embeddings[0]

        data=get_mesh(mesh_directory+"/"+mesh_obj)
        triangle_coordinates=Variable(torch.from_numpy(data['triangle_coordinates'])).float()
        normals=Variable(torch.from_numpy(data['normals'])).float()
        centers=Variable(torch.from_numpy(data['centers'])).float()
        areas=Variable(torch.from_numpy(data['areas'])).float()
        #print(f"triangle_coordinates.shape = {triangle_coordinates.shape} normals.shape={normals.shape} centers.shape={centers.shape} areas.shape={areas.shape} ")
        faceDataDim=triangle_coordinates.shape[1]+centers.shape[1]+normals.shape[1]+areas.shape[1]

        faceData=torch.cat((triangle_coordinates,normals,centers,areas),1).unsqueeze(0).detach()

        faceData_predicted , latent_vector =  mesh_net(faceData)
        mesh_embed=latent_vector            

                       


    
        embed_sets = len(embeddings) /batch_size
        embed_sets = int(embed_sets)

        for i in range(embed_sets):
            txt_embedding_list = []
            folder_name_list =[]
            wave_name_list = []
            for j in range(batch_size):
                mesh_obj,folder_name,wave_name,source_location,receiver_location = embeddings[((i*batch_size)+j)]

                source_receiver = source_location+receiver_location
                txt_embedding_single = np.array(source_receiver).astype('float32')

                txt_embedding_list.append(txt_embedding_single)
                folder_name_list.append(folder_name)
                wave_name_list.append(wave_name)

            txt_embedding =torch.from_numpy(np.array(txt_embedding_list))
            txt_embedding = Variable(txt_embedding)
            print(f"txt_embedding.shape={txt_embedding.shape} mesh_embed.shape={mesh_embed.shape}")
            lr_fake, fake, _  = netG(txt_embedding,mesh_embed.repeat(2,1))

            for i in range(len(fake)):
                if(not os.path.exists(output_embed+"/"+folder_name_list[i])):
                    os.mkdir(output_embed+"/"+folder_name_list[i])

                fake_RIR_path = output_embed+"/"+folder_name_list[i]+"/"+wave_name_list[i]
                fake_IR = np.array(fake[i].to("cpu").detach())
                fake_IR_only = fake_IR[:,0:(4096-128)]
                fake_energy = np.median(fake_IR[:,(4096-128):4096])*10
                fake_IR = fake_IR_only*fake_energy
                f = WaveWriter(fake_RIR_path, channels=1, samplerate=fs)
                f.write(np.array(fake_IR))
                f.close()


evaluate()
