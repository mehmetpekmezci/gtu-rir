from __future__ import print_function
from six.moves import range
from PIL import Image


import torch.backends.cudnn as cudnn
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
#from torch_geometric.loader import DataLoader
from  torch.utils.data import DataLoader

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
from miscc.datasets import RIRDataset
from miscc.config import cfg, cfg_from_file

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import torchaudio

import gc

from diffusers.models import AutoencoderKL

import requests
from huggingface_hub import configure_http_backend

def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session
configure_http_backend(backend_factory=backend_factory)



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


def load_network_stageI(netG_path):
        from model import STAGE1_G, STAGE1_D
        netG = STAGE1_G()     

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
            
#        except:
#            print("CAN NOT LOAD TRAINED NETWORKS, EXITING")
#            sys.exit(1)




        netG.eval()
        netG.cuda()
        image_vae = AutoencoderKL.from_pretrained("zelaki/eq-vae")
        image_vae.eval()
        image_vae.cuda()
        return netG, image_vae

def evaluate(main_dir,validation_pickle_file):
#    torch.set_default_device('cuda:0')
    cfg_from_file("cfg/RIR_s1.yml")    
    netG_path = "Models/netG.pth"
    batch_size = 32 # 500
    fs = 16000
    netG, image_vae = load_network_stageI(netG_path)
    embeddings = load_embedding(main_dir,'validation.embeddings.pickle')
    #dataset = TextDataset(main_dir,embeddings,None, split='train',rirsize=4096, gae_mesh_net=mesh_net)
    dataset = RIRDataset(main_dir, embeddings, rirsize=cfg.RIRSIZE)
    dataloader = DataLoader(dataset, batch_size=batch_size , num_workers=8,)
    mses=[]
    ssims=[]
    mse_loss=nn.MSELoss()     
    for i, data in enumerate(dataloader):
        with torch.no_grad():
                real_RIR_cpu = torch.from_numpy(np.array(data['RIR']))
                txt_embedding = data['source_and_receiver']
                mesh_embedding_source_image = data['mesh_embeddings_source_image']
                mesh_embedding_receiver_image = data['mesh_embeddings_receiver_image']

                real_RIRs = Variable(real_RIR_cpu)
                txt_embedding = Variable(txt_embedding)
                mesh_embedding_source_image = Variable(mesh_embedding_source_image)
                mesh_embedding_receiver_image = Variable(mesh_embedding_receiver_image)

                if True:
                    real_RIRs = real_RIRs.cuda()
                    txt_embedding = txt_embedding.cuda()
                    mesh_embedding_source_image = mesh_embedding_source_image.cuda()
                    mesh_embedding_receiver_image = mesh_embedding_receiver_image.cuda()

                if True:
                     mesh_embedding_source_image_latents=image_vae.encode(mesh_embedding_source_image).latent_dist.sample().reshape(batch_size,int(4*cfg.RAY_CASTING_IMAGE_RESOLUTION/8*cfg.RAY_CASTING_IMAGE_RESOLUTION/8))
                     mesh_embedding_receiver_image_latents=image_vae.encode(mesh_embedding_receiver_image).latent_dist.sample().reshape(batch_size,int(4*cfg.RAY_CASTING_IMAGE_RESOLUTION/8*cfg.RAY_CASTING_IMAGE_RESOLUTION/8))
                     mesh_embed=torch.concatenate((mesh_embedding_source_image_latents,mesh_embedding_receiver_image_latents),axis=1)

                inputs = (txt_embedding,mesh_embed)

                # _, fake_RIRs, mu, logvar = \
                #     nn.parallel.data_parallel(netG, inputs, self.gpus)

                # print("self.gpus ", [self.gpus[0]])
                _, fake_RIRs,c_code = nn.parallel.data_parallel(netG, inputs,  [0])


                #_, fake_RIRs,c_code = netG.forward(txt_embedding,mesh_embed)

                real_RIRs = real_RIRs[:,:,0:(4096-128)]
                real_RIRs = real_RIRs.cuda()
                txt_embedding = txt_embedding.cuda()
                mesh_embed = mesh_embed.cuda()


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
 
if __name__ == '__main__':
   evaluate(str(sys.argv[1]).strip(),str(sys.argv[2]).strip())









