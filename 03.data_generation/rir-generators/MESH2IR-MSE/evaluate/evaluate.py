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

def load_network_stageI(netG_path,mesh_net_path):
        from model import STAGE1_G, STAGE1_D, MESH_NET
        netG = STAGE1_G()
        netG.apply(weights_init)

        print(netG)
       

        mesh_net =MESH_NET() 



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


        
        netG.cuda()
        mesh_net.cuda()
        return netG, mesh_net

def get_graph(full_graph_path):
        
    with open(full_graph_path, 'rb') as f:
        graph = pickle.load(f)
        
    return graph #edge_index, vertex_position

def load_embedding(data_dir):
    # embedding_filename   = '/embeddings.pickle'  
    with open(data_dir, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings


def evaluate():
    print(f"metadata_dir={metadata_dir}") 
    embedding_directory =metadata_dir+"/Embeddings/"
    graph_directory = metadata_dir+"/Mesh_Graphs/"
    output_directory = generated_rirs_dir

    #netG_path = "Models/MESH2IR/netG_epoch_175.pth"
    #mesh_net_path = "Models/MESH2IR/mesh_net_epoch_175.pth"
    netG_path = "Models/netG_epoch_175.pth"
    mesh_net_path = "Models/mesh_net_epoch_175.pth"
    #gpus =[0,1]
    gpus =[0]

    #batch_size = 256
    batch_size = 2
    fs = 16000


    if(not os.path.exists(output_directory)):
        os.mkdir(output_directory)

    netG, mesh_net = load_network_stageI(netG_path,mesh_net_path)
    netG.eval()
    mesh_net.eval()


    netG.to(device='cuda')
    mesh_net.to(device='cuda')

    embedding_list = os.listdir(embedding_directory)
    
    for embed in embedding_list:
        embed_path = embedding_directory + "/"+embed
        embeddings = load_embedding(embed_path)
        embed_name = embed[0:len(embed)-7]
        output_embed  = output_directory+'/'+embed_name
        if(not os.path.exists(output_embed)):
            os.mkdir(output_embed)

        print("embed_name   ",output_embed)

        graph_path,folder_name,wave_name,source_location,receiver_location = embeddings[0]

        full_graph_path = graph_directory + graph_path

        data_single = get_graph(full_graph_path)
        data_list=[data_single]*batch_size
        loader = DataLoader(data_list, batch_size=batch_size)

        data = next(iter(loader))
        data['edge_index'] = Variable(data['edge_index'])
        data['pos'] = Variable(data['pos'])
        data = data.cuda()
        
        mesh_embed = nn.parallel.data_parallel(mesh_net, data,  [gpus[0]])
    
        embed_sets = len(embeddings) /batch_size
        embed_sets = int(embed_sets)

        for i in range(embed_sets):
            txt_embedding_list = []
            folder_name_list =[]
            wave_name_list = []
            for j in range(batch_size):
                graph_path,folder_name,wave_name,source_location,receiver_location = embeddings[((i*batch_size)+j)]

                source_receiver = source_location+receiver_location
                txt_embedding_single = np.array(source_receiver).astype('float32')

                txt_embedding_list.append(txt_embedding_single)
                folder_name_list.append(folder_name)
                wave_name_list.append(wave_name)


            txt_embedding =torch.from_numpy(np.array(txt_embedding_list))
            txt_embedding = Variable(txt_embedding)
            txt_embedding = txt_embedding.cuda()

         
            inputs = (txt_embedding,mesh_embed)
            lr_fake, fake, _ = nn.parallel.data_parallel(netG, inputs, gpus)

            for i in range(len(fake)):
                if(not os.path.exists(output_embed+"/"+folder_name_list[i])):
                    os.mkdir(output_embed+"/"+folder_name_list[i])

                fake_RIR_path = output_embed+"/"+folder_name_list[i]+"/"+wave_name_list[i]
                fake_IR = np.array(fake[i].to("cpu").detach())
                       
                fake_IR_only = fake_IR[:,0:(4096-128)]
                fake_energy = np.median(fake_IR[:,(4096-128):4096])*10
                fake_IR = fake_IR_only*fake_energy
                #print(f"fake_RIR_path={fake_RIR_path}")
                f = WaveWriter(fake_RIR_path, channels=1, samplerate=fs)
                f.write(np.array(fake_IR))
                f.close()


evaluate()




