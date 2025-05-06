from __future__ import print_function
from six.moves import range
from PIL import Image

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import sys
import time

import numpy as np
import cupy as cp
from cupyx.scipy.signal import fftconvolve
import torchfile
import pickle

import soundfile as sf
import re
import math
from wavefile import WaveWriter, Format

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import weights_init
from miscc.utils import save_RIR_results, save_model 
from miscc.utils import KL_loss
from miscc.utils import compute_discriminator_loss, compute_generator_loss
from miscc.utils import convert_IR2EC,convert_IR2EC_batch, generate_complementary_filterbank
import time
from torchinfo import summary
# from torch.utils.tensorboard import summary
# from torch.utils.tensorboard import FileWriter
import torchaudio
from torch_geometric.utils import to_dense_adj, to_dense_batch
import traceback
from model_mesh import MESH_TRANSFORMER_AE 

class GANTrainer(object):
    def __init__(self, output_dir):


        print(f"###################### GANTrainer : cfg.TRAIN.FLAG={cfg.TRAIN.FLAG}  ############################")
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.model_dir_RT = os.path.join(output_dir, 'Model_RT')
            self.RIR_dir = os.path.join(output_dir, 'RIR')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.model_dir_RT)
            mkdir_p(self.RIR_dir)
            mkdir_p(self.log_dir)
            # self.summary_writer = FileWriter(self.log_dir)
        print(f"###################### GANTrainer : self.model_dir={self.model_dir}  ############################")

        self.cfg=cfg
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        #self.batch_size = cfg.TRAIN.BATCH_SIZE 
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True



    # ############# For training stageI GAN #############
    def load_network(self):
        print("from model import RIR_TRANSFORMER")
        from model import RIR_TRANSFORMER

        netG = RIR_TRANSFORMER()
        netG.apply(weights_init)

        if cfg.NET_G != '':
            state_dict = \
                torch.load(cfg.NET_G,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load NETG from: ', cfg.NET_G)
        
        #print("Printing netG:")
        #print(netG)
        
        #print("printing netG arch summary:")
        SOURCE_RECEIVER_XYZ_DIM=6    
        summary(netG,[(self.batch_size,SOURCE_RECEIVER_XYZ_DIM),(self.batch_size,cfg.LATENT_VECTOR_SIZE)] )

        if cfg.CUDA:
            netG.cuda()
        return netG


    def train(self, data_loader, stage=1):
        netG= self.load_network()
        netG.to(device='cuda')
        batch_size = self.batch_size

        generator_lr = cfg.TRAIN.GENERATOR_LR
        netG_para = []
        for p in netG.parameters():
            if p.requires_grad:
                netG_para.append(p)
        optimizerG = optim.RMSprop(netG_para,
                                lr=cfg.TRAIN.GENERATOR_LR)

        for epoch in range(self.max_epoch):
            t1=time.time()
            for i, data in enumerate(data_loader, 0):
              try: 
                if  len(data['RIR']) < self.batch_size :
                    print("len(data['RIR']):",len(data))
                    print("self.batch_size:",self.batch_size)
                    continue

                real_RIR_cpu = torch.from_numpy(np.array(data['RIR']))
                txt_embedding = torch.from_numpy(np.array(data['embeddings']))
                mesh_embed = torch.from_numpy(np.array(data['mesh_embeddings'])) 
                
                real_RIRs = Variable(real_RIR_cpu)
                txt_embedding = Variable(txt_embedding) 
                mesh_embed = Variable(mesh_embed) 

                
                if cfg.CUDA:
                    real_RIRs = real_RIRs.cuda()
                    txt_embedding = txt_embedding.cuda()
                    mesh_embed = mesh_embed.cuda()

                inputs = (txt_embedding,mesh_embed)
                
                fake_RIRs = nn.parallel.data_parallel(netG, inputs,  self.gpus)

                netG.zero_grad()

                loss = netG.loss(fake_RIRs,real_RIRs)

                loss.backward()
               
                optimizerG.step()
               

                if i % 1000 == 0:
                    print("saving model ...")                    
                    save_model(netG, epoch, self.model_dir)

                if generator_lr > 0.00000005 and i>0 and ((i%500==0 and i<1100) or (i%1000==0 and i<11000)  or  i%10000==0 ) :
                    rate=0.5
                    print(f"decreasing lr by 0.5 old generator_lr={generator_lr} ")
                    generator_lr *= rate
                    for param_group in optimizerG.param_groups:
                        param_group['lr'] = generator_lr
                    print(f"new generator_lr={generator_lr} ")

                
                if i % 10 == 0:
                    t2 = time.time()
                    print('''[%d/%d][%d/%d] Loss_G: %.8f Total Time: %.2fsec''' % (epoch, self.max_epoch, i, len(data_loader),loss,(t2 - t1)),flush=True) 
                    t1=time.time()
                    save_model(netG, epoch, self.model_dir)
              except:
                print(f"we had an exception  i={i} ")
                traceback.print_exc() 
                sys.exit(1)
                continue 



        


