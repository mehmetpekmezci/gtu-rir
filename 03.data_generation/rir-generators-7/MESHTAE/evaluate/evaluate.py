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

from miscc.config import cfg
from miscc.datasets import TextDataset
from miscc.config import cfg, cfg_from_file

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import torchaudio

import gc


SSIM_DATA_RANGE=4 # 1 means lower SSIM, 2,4,8... means greater ssim, but will stay parralel 

def pure_ssim(real_data,generated_data):
         generated_data_tiled=torch.tile(generated_data, (2, 1)) ## duplicate 1d data to 2d
         real_data_tiled=torch.tile(real_data, (2, 1)) ## duplicate 1d data to 2d
         generated_data_tiled=torch.reshape(generated_data_tiled,(1,generated_data_tiled.shape[0],generated_data_tiled.shape[1],generated_data_tiled.shape[2]))
         real_data_tiled=torch.reshape(real_data_tiled,(1,real_data_tiled.shape[0],real_data_tiled.shape[1],real_data_tiled.shape[2]))

         SSIM=ssim(generated_data_tiled,real_data_tiled, data_range=SSIM_DATA_RANGE, size_average=True).item()
         return SSIM


def load_embedding(data_dir,embedding_file_name):
        print("Loading embeddings ...")
        embedding_directory   = data_dir+'/'+embedding_file_name
        with open(embedding_directory, 'rb') as f:
            embeddings = pickle.load(f)
        print(embedding_file_name+" embeddings are loaded ...")
        return embeddings


def load_network_stageI(netG_path,mesh_net_path):
        from model import STAGE1_G, STAGE1_D
        from model_mesh import MESH_TRANSFORMER_AE

        netG = STAGE1_G()     
        mesh_net = MESH_TRANSFORMER_AE()


        if True:
#        try:
         if netG_path!= '':
            state_dict = \
                torch.load(netG_path,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load from: ', netG_path)
         else:
            print("CAN NOT FIND NETG, EXITING")
            sys.exit(1)
            
         if mesh_net_path != '':
            state_dict = \
                torch.load(mesh_net_path,
                           map_location=lambda storage, loc: storage)
            mesh_net.load_state_dict(state_dict)
            print('Load from: ', mesh_net_path)
         else:
            print("CAN NOT FIND MESH_NET, EXITING")
            sys.exit(1)
#        except:
#            print("CAN NOT LOAD TRAINED NETWORKS, EXITING")
#            sys.exit(1)


        
        netG.cuda()
        mesh_net.cuda()
        return netG, mesh_net

def evaluate(main_dir,validation_pickle_file):
#    torch.set_default_device('cuda:0')
    cfg_from_file("cfg/RIR_s1.yml")    
    netG_path = "Models/netG.pth"
    mesh_net_path = "Models/mesh_net.pth"
    batch_size = 1000 # 1600
    fs = 16000
    netG, mesh_net = load_network_stageI(netG_path,mesh_net_path)
    netG.eval()
    mesh_net.eval()
    embeddings = load_embedding(main_dir,'validation.embeddings.pickle')
    dataset = TextDataset(main_dir,embeddings,None, split='train',rirsize=4096, gae_mesh_net=mesh_net)
    dataloader = DataLoader(dataset, batch_size=batch_size , num_workers=0,)
    mses=[]
    ssims=[]
    mse_loss=nn.MSELoss()     
    for i, data in enumerate(dataloader):
        with torch.no_grad():
                real_RIRs = torch.from_numpy(np.array(data['RIR']))
                txt_embedding = torch.from_numpy(np.array(data['embeddings']))
                mesh_embed = torch.from_numpy(np.array(data['mesh_embeddings']))

                #data.pop('RIR')
                #data.pop('embeddings')

                #real_RIRs = Variable(real_RIR_cpu)
                #txt_embedding = Variable(txt_embedding)


                real_RIRs = real_RIRs[:,:,0:(4096-128)]
                real_RIRs = real_RIRs.cuda()
                txt_embedding = txt_embedding.cuda()
                mesh_embed = mesh_embed.cuda()
                #print(f"txt_embedding.shape={txt_embedding.shape}")
                #print(f"mesh_embed.shape={mesh_embed.shape}")
                #data = data.cuda()

                GPU_NOs=[0]

                _, fake_RIRs,c_code = netG.forward(txt_embedding,mesh_embed)
                fake_RIRs.detach().cpu()
                real_RIRs.detach().cpu()
                #data.detach().cpu()
                txt_embedding.detach().cpu()
                mesh_embed.detach().cpu()
                fake_RIR_only = fake_RIRs[:,:,0:(4096-128)]
                fake_energy = torch.median(fake_RIRs[:,:,(4096-128):4096])*10
                fake_RIRs = fake_RIR_only*fake_energy
                mse=mse_loss(real_RIRs,fake_RIRs).item()
                ssim=pure_ssim(real_RIRs,fake_RIRs)
                print(f"{i}/{len(dataloader)} MSE={mse} SSIM={ssim}",flush=True)
                mses.append(mse)
                ssims.append(ssim)
                torch.cuda.empty_cache()
                del mesh_embed
                del fake_RIRs
                del real_RIRs
                del data
                gc.collect()
#               for j in range(len(fake_RIRs_computed)):
#                   fake_RIR_path=full_real_RIR_paths+'.MESH2IR.wav'
#                   fake_IR = np.array(fake_RIRs_computed[j].to("cpu").detach())
#                   f = WaveWriter(fake_RIR_path, channels=1, samplerate=fs)
#                   f.write(np.array(fake_IR))


    MEAN_MSE=np.array(mses).mean()
    MEAN_SSIM=np.array(ssims).mean()
    
    print(f"MEAN_MSE={MEAN_MSE} MEAN_SSIM={MEAN_SSIM}",flush=True)
 

evaluate(str(sys.argv[1]).strip(),str(sys.argv[2]).strip())











   # 
