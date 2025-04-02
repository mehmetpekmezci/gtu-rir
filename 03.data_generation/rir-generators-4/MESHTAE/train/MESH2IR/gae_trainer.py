from __future__ import print_function
#from six.moves import range
from PIL import Image

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
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
from miscc.utils import save_RIR_results, save_model, save_mesh_model,save_mesh_final_model, convert_to_trimesh,plot_mesh , plot_points,edge_index_to_face
from miscc.datasets import save_face_normal_center_area_as_obj,denormalize_mesh_values
import time
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_dense_batch, dropout_adj, unbatch_edge_index,unbatch
import scipy.sparse as sp
import traceback
from torchinfo import summary
#from torchstat import stat

class GAETrainer(object):
    def __init__(self, output_dir):
        print(f"###################### GAETrainer : cfg.TRAIN.FLAG={cfg.TRAIN.FLAG}  ############################")
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.model_dir_RT = os.path.join(output_dir, 'Model_RT')
            self.RIR_dir = os.path.join(output_dir, 'RIR')
            self.MESH_dir = os.path.join(output_dir, 'MESH')
            self.log_dir = os.path.join(output_dir, 'Log')
            
            mkdir_p(self.model_dir)
            mkdir_p(self.model_dir_RT)
            mkdir_p(self.RIR_dir)
            mkdir_p(self.MESH_dir)
            mkdir_p(self.log_dir)
            # self.summary_writer = FileWriter(self.log_dir)

        self.max_epoch = int(cfg.TRAIN.MAX_MESHNET_GAE_EPOCH)
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        #self.gpus=[0]
        self.num_gpus = len(self.gpus)
        #print(f"self.gpus={self.gpus}")
        #print(f"self.num_gpus={self.num_gpus}")
        #self.batch_size = cfg.TRAIN.GAE_BATCH_SIZE * self.num_gpus
        self.batch_size = cfg.TRAIN.GAE_BATCH_SIZE 
        print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True
        self.mesh_net = self.load_network()
    # ############# For training stageI GAN #############
    def load_network(self):
        from model_mesh import MESH_TRANSFORMER_AE

        mesh_net =MESH_TRANSFORMER_AE() 
        
        
#            mesh_net = Mesh_mae(
#                   masking_ratio=args.mask_ratio,# 0.75
#                   channels=args.channels, # 13
#                   num_heads=args.heads, # 12
#                   encoder_depth=args.encoder_depth, # 12
#                   embed_dim=args.dim, # 768
#                   decoder_num_heads=args.decoder_num_heads, # 16
#                   decoder_depth=args.decoder_depth, # 6
#                   decoder_embed_dim=args.decoder_dim, # 512
#                   patch_size=args.patch_size, # 64
#                   weight=args.weight # 0.2
#                  )


        if cfg.MESH_NET != '':
            state_dict = \
                torch.load(cfg.MESH_NET,
                           map_location=lambda storage, loc: storage)
            mesh_net.load_state_dict(state_dict)
            print('Load from: ', cfg.MESH_NET)

        #faces_summary=torch.rand(cfg.TRAIN.GAE_BATCH_SIZE,cfg.MAX_FACE_COUNT,16).cuda()
        #faces_summary=torch.rand(cfg.TRAIN.GAE_BATCH_SIZE,cfg.MAX_FACE_COUNT,cfg.MESH_FACE_DATA_SIZE).cuda()
        faces_summary=torch.rand(cfg.TRAIN.GAE_BATCH_SIZE,cfg.MAX_FACE_COUNT,cfg.MESH_FACE_DATA_SIZE)
        summary(mesh_net,input_data=[faces_summary] )

        return mesh_net


    def train(self, data_loader, stage=1):
        self.mesh_net=self.mesh_net.cuda()
        MAX_DIM=cfg.MAX_DIM
        self.mesh_net.to(device='cuda')
        mesh_lr = cfg.TRAIN.MESH_LR
        lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH

        optimizerM = optim.RMSprop(self.mesh_net.parameters(),lr=cfg.TRAIN.MESH_LR)

        least_RT=10
        loss=0
        self.mesh_net.train()

#        torch.set_printoptions(profile="full")

        for epoch in range(self.max_epoch):

            start_t = time.time()
            t1=start_t
#            if epoch % lr_decay_step == 0 and epoch > 0:
#                mesh_lr *= 0.5#0.5  ### HER EPCOHTA learning rate 0.8 azaliyor.
#                for param_group in optimizerM.param_groups:
#                    param_group['lr'] = mesh_lr
#            

            for i, data in enumerate(data_loader, 0):
               # optimizerM.zero_grad() https://discuss.pytorch.org/t/model-zero-grad-or-optimizer-zero-grad/28426/7
               self.mesh_net.zero_grad()   
               (triangle_coordinates,normals,centers,areas, full_mesh_path)=data
               if triangle_coordinates is None:
                   continue
               #print(f"full_mesh_path[0]={full_mesh_path[0]}")
               triangle_coordinates=Variable(triangle_coordinates).float()
               normals=Variable(normals).float()
               centers=Variable(centers).float()
               areas=Variable(areas).float()
               #print(f"triangle_coordinates.shape = {triangle_coordinates.shape} normals.shape={normals.shape} centers.shape={centers.shape} areas.shape={areas.shape} ")
               faceDataDim=triangle_coordinates.shape[2]+centers.shape[2]+normals.shape[2]+areas.shape[2]
               faceData=torch.cat((triangle_coordinates,normals,centers,areas),2)


               #print(f"1 faceData.shape={faceData.shape}")
               
               #faceData=faceData.reshape(faceData.shape[0]*faceData.shape[1],faceData.shape[2])


               
               if cfg.CUDA:
                 faceData = faceData.to(torch.float32).cuda()


               faceData_predicted,latent_vector=nn.parallel.data_parallel(self.mesh_net, (faceData, ), self.gpus)

#               print(f"faceData_predicted.shape={faceData_predicted.shape}")
#               print(f"faceData.shape={faceData.shape}")
               
               loss = self.mesh_net.loss(faceData_predicted,faceData)

               loss.backward()
               optimizerM.step()



               if i % 10 == 0 :
                  end_t = time.time()
                  print('''[%d/%d][%d/%d] loss: %.8f Total Time: %.8fsec'''% (epoch, self.max_epoch, i, len(data_loader),loss,(end_t - start_t)),flush=True)
               if i>0 and (i==100 or i==500 or i==1000 or i % 10000 == 0):
                    save_mesh_model(self.mesh_net, epoch, self.model_dir)
                    print(f"saved_model : {i}")
                    if mesh_lr > 0.00000001:
                     mesh_lr *= 0.5#0.5  ### HER EPCOHTA learning rate 0.8 azaliyor.
                     for param_group in optimizerM.param_groups:
                        print(param_group['lr'])
                        param_group['lr'] = mesh_lr
                        print(param_group['lr'])
               if  (epoch>0 and epoch%10 == 0 and i==0) or (i>0 and i % 10000 == 0) :
#                 try :
                    path=full_mesh_path[0]
                    print(f"Generating example mesh STARTED : {path}.face.regenerated.1.obj")
                    faceData_pred=faceData_predicted[0].cpu().detach().numpy()
                    faceData_pred=faceData_pred.reshape(cfg.MAX_FACE_COUNT,faceDataDim)
                    triangle_coordinates=faceData_pred[:,0:9]
                    normals=faceData_pred[:,9:12]
                    centers=faceData_pred[:,12:15]
                    areas=faceData_pred[:,15:16]
                    areas=abs(areas.squeeze()+0.000001)
                    triangle_coordinates,normals,centers,areas=denormalize_mesh_values(triangle_coordinates,normals,centers,areas)
                    save_face_normal_center_area_as_obj(normals,centers,areas,path+".face.regenerated.1.obj")
                    print(f"Generating example mesh ENDED: {path}.face.regenerated.1.obj")
#                 except:   

                  
            end_t = time.time()
            print('''[%d/%d][%d/%d] loss: %.8f Total Time: %.8fsec'''% (epoch, self.max_epoch, i, len(data_loader),loss,(end_t - start_t)))

            
            if epoch % self.snapshot_interval == 0:
                save_mesh_model(self.mesh_net, epoch, self.model_dir)
        #
        save_mesh_model(self.mesh_net, self.max_epoch, self.model_dir)
        save_mesh_final_model(self.mesh_net)


