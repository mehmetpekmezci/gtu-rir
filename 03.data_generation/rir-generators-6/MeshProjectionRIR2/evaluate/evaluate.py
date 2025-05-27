from __future__ import print_function
from six.moves import range
from PIL import Image
from diffusers.models import AutoencoderKL

from miscc.datasets import RIRDataset

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


import requests
from huggingface_hub import configure_http_backend

def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session
configure_http_backend(backend_factory=backend_factory)



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

def load_network_stageI(netG_path):
        from model import STAGE1_G, STAGE1_D
        netG = STAGE1_G()
        netG.apply(weights_init)

        print(netG)
       



        if netG_path!= '':
            state_dict = \
                torch.load(netG_path,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load from: ', netG_path)
       
        #netG.to(device='cuda:2')
        netG.cuda()

        image_vae = AutoencoderKL.from_pretrained("zelaki/eq-vae")
        image_vae.eval()
        image_vae.cuda()

        return netG,image_vae


def load_embedding(data_dir):
    # embedding_filename   = '/embeddings.pickle'  
    with open(data_dir, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings


def evaluate():
    print(f"metadata_dir={metadata_dir}") 
    gpus = cfg.GPU_ID.split(',')
    gpus = [int(ix) for ix in gpus]
        #self.gpus=[0]
    num_gpus = len(gpus)

    torch.cuda.set_device(gpus[0]) 
    
    SCRIPT_DIR=os.path.dirname(os.path.realpath(__file__))
    
    print("This file is found in "+SCRIPT_DIR)
    
    cfg_from_file(SCRIPT_DIR+'/cfg/RIR_s1.yml')
   
    cfg.TRAIN.FLAG = False

    embedding_directory =metadata_dir+"/Embeddings/"
    mesh_directory = metadata_dir+"/Meshes"
    output_directory = generated_rirs_dir

    #netG_path = "Models/netG_GAN_"+str(cfg.MAX_FACE_COUNT)+"_nodes_"+str(cfg.NUMBER_OF_TRANSFORMER_HEADS)+"_heads.pth"
    netG_path = "Models/netG.pth"
    #gpus =[0,1]
    #gpus =[0]

    #batch_size = 256
    batch_size = 1
    fs = 16000


    if(not os.path.exists(output_directory)):
        os.mkdir(output_directory)

    netG,image_vae = load_network_stageI(netG_path)
    netG.eval()

    embedding_list = os.listdir(embedding_directory)

    for embedding_pickle_per_room in embedding_list:
      print("embedding_pickle_per_room:",embedding_pickle_per_room)
      if not "mesh_embeddings" in embedding_pickle_per_room:
        embed_path = embedding_directory + "/"+embedding_pickle_per_room
        embeddings = load_embedding(embed_path)
        embed_name = embedding_pickle_per_room[0:len(embedding_pickle_per_room)-7]
        output_embed  = output_directory+'/'+embed_name
        if(not os.path.exists(output_embed)):
            os.mkdir(output_embed)

        print("embed_name   ",output_embed)
        print("len(embeddings)   ",len(embeddings))

        mesh_obj,folder_name,wave_name,source_location,receiver_location = embeddings[0]
        print("mesh_obj:",mesh_obj)
        print("folder_name:",folder_name)
        print("wave_name:",wave_name)
        print("source_location:",source_location)
        print("receiver_location:",receiver_location)

        rir_dataset = RIRDataset(cfg.DATA_DIR, embeddings, rirsize=cfg.RIRSIZE)



        embed_sets = len(embeddings) /batch_size
        embed_sets = int(embed_sets)

        for i in range(embed_sets):
            txt_embedding_list = []
            folder_name_list =[]
            wave_name_list = []
            for j in range(batch_size):
                mesh_obj,folder_name,wave_name,source_location,receiver_location = embeddings[((i*batch_size)+j)]
                mesh_embed,room_dims=rir_dataset.mesh_embeddings(os.path.join(mesh_directory,mesh_obj))#,source_location,receiver_location)

                source_receiver = source_location+receiver_location+room_dims
                txt_embedding_single = np.array(source_receiver).astype('float32')

                txt_embedding_list.append(txt_embedding_single)
                folder_name_list.append(folder_name)
                wave_name_list.append(wave_name)

            print(mesh_embed)
            print(mesh_embed.shape)
            txt_embedding =torch.from_numpy(np.array(txt_embedding_list))
            mesh_embed=torch.from_numpy(np.array(mesh_embed)).unsqueeze(0)
            #txt_embedding = Variable(txt_embedding).detach().to(device='cuda:2')
            txt_embedding = Variable(txt_embedding).detach().cuda()
            mesh_embed = Variable(mesh_embed).detach().cuda()
            print(txt_embedding)
            print(txt_embedding.shape)
            #print(f"txt_embedding.shape={txt_embedding.shape} mesh_embed.shape={mesh_embed.shape}")
            #lr_fake, fake, _  = netG(txt_embedding,mesh_embed.repeat(2,1))
            #print(txt_embedding.shape)
            #print(mesh_embed.unsqueeze(0).shape)
            #lr_fake, fake, _  = netG.forward(txt_embedding,mesh_embed.unsqueeze(0))
            lr_fake, fake, _  = netG.forward(txt_embedding,mesh_embed)

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

