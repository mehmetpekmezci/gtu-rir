from __future__ import print_function
#from six.moves import range
from PIL import Image

#import subprocess
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
from miscc.utils import save_RIR_results, save_model, save_mesh_encoder_model,save_mesh_encoder_final_model, save_mesh_decoder_model,save_mesh_decoder_final_model, convert_to_trimesh,plot_mesh , plot_points,edge_index_to_face
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
        self.mesh_net_encoder,self.mesh_net_decoder = self.load_network()
       
    # ############# For training stageI GAN #############
    def load_network(self):
        from model_mesh import MESH_TRANSFORMER_ENCODER,MESH_TRANSFORMER_DECODER

        mesh_net_encoder =MESH_TRANSFORMER_ENCODER() 
        mesh_net_decoder =MESH_TRANSFORMER_DECODER() 
        
        
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

        if os.path.exists(cfg.PRE_TRAINED_MODELS_DIR+'/'+cfg.MESH_NET_ENCODER_FILE):
            state_dict = \
                torch.load( cfg.PRE_TRAINED_MODELS_DIR+'/'+cfg.MESH_NET_ENCODER_FILE,
                           map_location=lambda storage, loc: storage)
            mesh_net_encoder.load_state_dict(state_dict)
            print('Load GAE MESH ENCODER NET from: ', mesh_net_path)

        if os.path.exists(cfg.PRE_TRAINED_MODELS_DIR+'/'+cfg.MESH_NET_DECODER_FILE):
            state_dict = \
                torch.load( cfg.PRE_TRAINED_MODELS_DIR+'/'+cfg.MESH_NET_DECODER_FILE,
                           map_location=lambda storage, loc: storage)
            mesh_net_decoder.load_state_dict(state_dict)
            print('Load GAE MESH DECODER NET from: ', mesh_net_decoder_path)


        #faces_summary=torch.rand(cfg.TRAIN.GAE_BATCH_SIZE,cfg.MAX_FACE_COUNT,16).cuda()
        #faces_summary=torch.rand(cfg.TRAIN.GAE_BATCH_SIZE,cfg.MAX_FACE_COUNT,cfg.MESH_FACE_DATA_SIZE).cuda()
        faces_summary_encoder_X=torch.rand(cfg.TRAIN.GAE_BATCH_SIZE,cfg.MAX_FACE_COUNT,cfg.MESH_FACE_DATA_SIZE)
        faces_summary_decoder_Y=torch.rand(cfg.TRAIN.GAE_BATCH_SIZE,cfg.MAX_FACE_COUNT,mesh_net_decoder.EMBEDDING_DIM)
        faces_summary_decoder_K=torch.rand(cfg.TRAIN.GAE_BATCH_SIZE,cfg.MAX_FACE_COUNT,mesh_net_decoder.EMBEDDING_DIM)
        faces_summary_decoder_V=torch.rand(cfg.TRAIN.GAE_BATCH_SIZE,cfg.MAX_FACE_COUNT,mesh_net_decoder.EMBEDDING_DIM)
        summary(mesh_net_encoder,input_data=[faces_summary_encoder_X] )
        summary(mesh_net_decoder,input_data=[faces_summary_decoder_Y,faces_summary_decoder_K,faces_summary_decoder_V] )

        return mesh_net_encoder,mesh_net_decoder


    def train(self, data_loader, stage=1):
        #subprocess.run(["nvidia-smi"])
        #print("-2############")
        self.mesh_net_encoder=self.mesh_net_encoder.to(device='cuda:'+str(self.gpus[0]))
        #subprocess.run(["nvidia-smi"])
        #print("-1############")
        self.mesh_net_decoder=self.mesh_net_decoder.to(device='cuda:'+str(self.gpus[1]))
        #subprocess.run(["nvidia-smi"])
        #print("0############")
        
        MAX_DIM=cfg.MAX_DIM
        #self.mesh_net_encoder.to(device='cuda:0')
        mesh_lr = cfg.TRAIN.MESH_LR
        lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH

        optimizerMeshEncoder = optim.RMSprop(self.mesh_net_encoder.parameters(),lr=cfg.TRAIN.MESH_LR)
        optimizerMeshDecoder = optim.RMSprop(self.mesh_net_decoder.parameters(),lr=cfg.TRAIN.MESH_LR)

        least_RT=10
        loss=0
        self.mesh_net_encoder.train()
        self.mesh_net_decoder.train()

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
               #self.mesh_net.zero_grad()   
               self.mesh_net_encoder.zero_grad()   
               self.mesh_net_decoder.zero_grad()   
               (triangle_coordinates,normals,centers,areas, full_mesh_path)=data
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

               #subprocess.run(["nvidia-smi"])
               #print("1############")
               
               #if cfg.CUDA:
               faceData = faceData.to(torch.float32).to(device='cuda:'+str(self.gpus[0]))

               #subprocess.run(["nvidia-smi"])
               #print("2############")


               #with torch.autocast(device_type="cuda",dtype=torch.float16):
               Z,K,V,embeddings=self.mesh_net_encoder(faceData, )

               #torch.cuda.set_device(self.gpus[0])
               #Z.detach().to(device='cuda:'+str(self.gpus[1]))
               #subprocess.run(["nvidia-smi"])
               #print("3############")
               embeddings=embeddings.to(torch.float16).to(device='cuda:'+str(self.gpus[1]))
               Z=Z.to(torch.float16).to(device='cuda:'+str(self.gpus[1]))
               K=K.to(torch.float16).to(device='cuda:'+str(self.gpus[1]))
               V=V.to(torch.float16).to(device='cuda:'+str(self.gpus[1]))
               #subprocess.run(["nvidia-smi"])
               #print("4############")
               #print(f'self.gpus={self.gpus} self.gpus[1]={self.gpus[1]} Z={Z}')
               #with torch.autocast(device_type="cuda",dtype=torch.float16):
               #with torch.cuda.amp.autocast(True):
               faceData_predicted=self.mesh_net_decoder(embeddings,K,V)
               #subprocess.run(["nvidia-smi"])
               #print("5############")
               #faceData_predicted,latent_vector=nn.parallel.data_parallel(self.mesh_net, (faceData, ), self.gpus)
    #               print(f"faceData_predicted.shape={faceData_predicted.shape}")
    #               print(f"faceData.shape={faceData.shape}")

               #faceData_predicted=faceData_predicted.cpu()
               #faceData=faceData.cpu()
               #loss=nn.MSELoss()(faceData_predicted,faceData)

               #subprocess.run(["nvidia-smi"])
               #print("6############")
               faceData=faceData.to(device='cuda:'+str(self.gpus[1]))
               #subprocess.run(["nvidia-smi"])
               #print("7############")
               loss = self.mesh_net_decoder.loss(faceData_predicted,faceData)
               #subprocess.run(["nvidia-smi"])
               #print("8############")

               ## MP: loss.backward hicbir zaman autocastin ucunde olmamalidir.
               loss.backward()
               optimizerMeshDecoder.step()
               optimizerMeshEncoder.step()



               if i % 10 == 0 :
                  end_t = time.time()
                  print('''[%d/%d][%d/%d] loss: %.8f Total Time: %.8fsec'''% (epoch, self.max_epoch, i, len(data_loader),loss,(end_t - start_t)),flush=True)
               if i>0 and (i==100 or i==500 or i==1000 or i % 10000 == 0):
                    save_mesh_encoder_model(self.mesh_net_encoder, epoch, self.model_dir)
                    save_mesh_decoder_model(self.mesh_net_decoder, epoch, self.model_dir)
                    print(f"saved_model : {i}")
                    if mesh_lr > 0.00000001:
                     mesh_lr *= 0.5#0.5  ### HER EPCOHTA learning rate 0.8 azaliyor.
                     for param_group in optimizerMeshEncoder.param_groups:
                        print(param_group['lr'])
                        param_group['lr'] = mesh_lr
                        print(param_group['lr'])
                     for param_group in optimizerMeshDecoder.param_groups:
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
                save_mesh_encoder_model(self.mesh_net_encoder, epoch, self.model_dir)
                save_mesh_decoder_model(self.mesh_net_decoder, epoch, self.model_dir)
        #
        save_mesh_encoder_model(self.mesh_net_encoder, self.max_epoch, self.model_dir)
        save_mesh_encoder_final_model(self.mesh_net_encoder)
        save_mesh_decoder_model(self.mesh_net_decoder, self.max_epoch, self.model_dir)
        save_mesh_decoder_final_model(self.mesh_net_decoder)


