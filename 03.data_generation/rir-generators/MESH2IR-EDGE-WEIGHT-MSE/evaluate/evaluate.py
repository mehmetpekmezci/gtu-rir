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

    SCRIPT_DIR=os.path.dirname(os.path.realpath(__file__))
    
    print("This file is found in "+SCRIPT_DIR)
    
    cfg_from_file(SCRIPT_DIR+'/cfg/RIR_s1.yml')
    
    embedding_directory =metadata_dir+"/Embeddings/"
    graph_directory = metadata_dir+"/Mesh_Graphs/"
    output_directory = generated_rirs_dir

    #netG_path = "Models/MESH2IR/netG_epoch_175.pth"
    #mesh_net_path = "Models/MESH2IR/mesh_net_epoch_175.pth"
    netG_path = "Models/netG.pth"
    mesh_net_path = "Models/mesh_net.pth"
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

        graph_path,folder_name,wave_name,source_location,receiver_location,cos_theta_edge_weight,graph = embeddings[0]

        #cos_theta_edge_weight=(cos_theta_edge_weight+1)/2 
        cos_theta_edge_weight=np.abs(cos_theta_edge_weight)
        #### ## MP: for numerical stability, cos_theta [-1,1] has negative values, but we cannot have negative values in edge_weights, so we add +1 and divide by 2.
        #### ## as this is a linear operation (x+1/2)  , this will not effect the results.
        #### ## we do the same operation in miscc/datasets.py while training


        data_single = graph
        data_single['edge_weights'] = torch.Tensor(cos_theta_edge_weight)

        data_list=[data_single]*batch_size
        loader = DataLoader(data_list, batch_size=batch_size)

        data = next(iter(loader))

        data['edge_index'] = Variable(data['edge_index'])
        data['pos'] = Variable(data['pos'])
        data['edge_weights'] = Variable(data['edge_weights'])
        data = data.cuda()
       
        print("data['edge_weights']:")
        print(data['edge_weights'])
        
        mesh_embed = nn.parallel.data_parallel(mesh_net, data,  [gpus[0]])

    
        embed_sets = len(embeddings) /batch_size
        embed_sets = int(embed_sets)

        for i in range(embed_sets):
            txt_embedding_list = []
            folder_name_list =[]
            wave_name_list = []
            for j in range(batch_size):
                graph_path,folder_name,wave_name,source_location,receiver_location,cos_theta,graph = embeddings[((i*batch_size)+j)]

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
                f = WaveWriter(fake_RIR_path, channels=1, samplerate=fs)
                f.write(np.array(fake_IR))
                f.close()


evaluate()




   #      time_counter = 0
   #      # start_time = time.time()
   #      counter_ir=0
   #      for i, data in enumerate(data_loader, 0):
   #          # real_RIR_cpu = torch.from_numpy(np.array(data['RIR']))
   #          txt_embedding = torch.from_numpy(np.array(data['embeddings']))
   #          path = data['path']

   #          txt_embedding_cpu = np.array(data['embeddings'])
   #          wavename = np.array(data['wavename'])
   #          foldername = np.array(data['foldername'])
   #          # data.pop('RIR')
   #          data.pop('embeddings')
   #          data.pop('path')
   #          data.pop('wavename')
   #          data.pop('foldername')

   #          txt_embedding = Variable(txt_embedding)
   #          data['edge_index'] = Variable(data['edge_index'])
   #          data['pos'] = Variable(data['pos'])

   #          if cfg.CUDA:
   #              txt_embedding = txt_embedding.cuda()
   #              data = data.cuda()

   #          print("data is   ", data)
   #          mesh_embed = nn.parallel.data_parallel(mesh_net, data,  [self.gpus[0]])

   #          inputs = (txt_embedding,mesh_embed)
   #          print("txt_embedding    ",txt_embedding)
   #          print("txt_embedding  shape   ",txt_embedding.shape)



   #          lr_fake, fake, _ = nn.parallel.data_parallel(netG, inputs, self.gpus)

   #          # save_RIR_results_eval(foldername,wavename,fake,txt_embedding_cpu,path, self.eval_dir,counter_ir)
   #          counter_ir = counter_ir + 1
   #

