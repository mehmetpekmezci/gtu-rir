from __future__ import print_function
from six.moves import range
from PIL import Image

from diffusers.models import AutoencoderKL

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
import traceback

import requests
from huggingface_hub import configure_http_backend

def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session
configure_http_backend(backend_factory=backend_factory)

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
    def load_network_stageI(self):
        print("from model import STAGE1_G")
        from model import STAGE1_G
        print("from model import STAGE1_D")
        from model import STAGE1_D



        netG = STAGE1_G()
        netG.apply(weights_init)

#        print(netG)
        netD = STAGE1_D()
        netD.apply(weights_init)
#        print(netD)

        if cfg.NET_G != '':
            state_dict = \
                torch.load(cfg.NET_G,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load NETG from: ', cfg.NET_G)
        if cfg.NET_D != '':
            state_dict = \
                torch.load(cfg.NET_D,
                           map_location=lambda storage, loc: storage)
            netD.load_state_dict(state_dict)
            print('Load NETD from: ', cfg.NET_D)

        
        SOURCE_RECEIVER_XYZ_DIM=6    
        summary(netG,[(self.batch_size,SOURCE_RECEIVER_XYZ_DIM),(self.batch_size,int(2*4*cfg.RAY_CASTING_IMAGE_RESOLUTION/8*cfg.RAY_CASTING_IMAGE_RESOLUTION/8))] )
        summary(netD,(self.batch_size,1,cfg.RIRSIZE) )
       
        image_vae = AutoencoderKL.from_pretrained("zelaki/eq-vae")
        image_vae.eval()

        if cfg.CUDA:
            netG.cuda()
            netD.cuda()
            image_vae.cuda()
        return netG, netD , image_vae


    def train(self, data_loader, stage=1):
        if stage == 1:
            netG, netD , image_vae= self.load_network_stageI()
        # else:
        #     netG, netD = self.load_network_stageII()
        
        netG.to(device='cuda')
        batch_size = self.batch_size

        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        if cfg.CUDA:
            # noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
            real_labels, fake_labels = real_labels.cuda(), fake_labels.cuda()

        generator_lr = cfg.TRAIN.GENERATOR_LR
        discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
        lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH
        # optimizerD = \
        #     optim.Adam(netD.parameters(),
        #                lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))
        optimizerD = \
            optim.RMSprop(netD.parameters(),
                       lr=cfg.TRAIN.DISCRIMINATOR_LR)
        # optimizerD =optim.Adadelta(netD.parameters())
        # optimizerD = optim.Adagrad(netD.parameters(),lr=cfg.TRAIN.DISCRIMINATOR_LR)
        netG_para = []
        for p in netG.parameters():
            if p.requires_grad:
                netG_para.append(p)
        # optimizerG = optim.Adam(netG_para,
        #                         lr=cfg.TRAIN.GENERATOR_LR,
        #                         betas=(0.5, 0.999))
        optimizerG = optim.RMSprop(netG_para,
                                lr=cfg.TRAIN.GENERATOR_LR)

        # optimizerG = optim.Adadelta(netG_para)

        # optimizerG = optim.Adagrad(netG_para,lr=cfg.TRAIN.GENERATOR_LR)
        count = 0
        least_RT=10
        L1_error_temp =150/4096
        bands = [125, 250, 500, 1000, 2000, 4000]  # which frequency bands are we interested in
        filter_length = 16384  # a magic number, not need to tweak this much
        fs =16000
        # only generate filters once and keep using them, that means you need to know the samplerate beforehand or convert to a fixed samplerate
        filters = generate_complementary_filterbank(fc=bands, fs=fs, filter_order=4, filter_length=filter_length, power=True)
        filters = cp.asarray([[filters]])
        for epoch in range(self.max_epoch):
            start_t = time.time()
            t1=start_t
#            if epoch % lr_decay_step == 0 and epoch > 0:
#                generator_lr *= 0.85#0.5
#                for param_group in optimizerG.param_groups:
#                    param_group['lr'] = generator_lr
#                discriminator_lr *= 0.85#0.5
#                for param_group in optimizerD.param_groups:
#                    param_group['lr'] = discriminator_lr

            
            # b1=0
            # b2=0
            # b3=0
            for i, data in enumerate(data_loader, 0):
              errD_total=0
              errG_total=0              
              try: 
                if  len(data['RIR']) < self.batch_size :
                    print("len(data['RIR']):",len(data))
                    print("self.batch_size:",self.batch_size)
                    continue

                ######################################################
                # (1) Prepare training data
                ######################################################
                # real_RIR_cpu, txt_embedding = data
                # b1 = time.time()
                # print("Time 1   ",(b1-b3))
               
                
                real_RIR_cpu = torch.from_numpy(np.array(data['RIR']))
                #txt_embedding = torch.from_numpy(data['source_and_receiver'])
                txt_embedding = data['source_and_receiver']
                mesh_embedding_source_image = data['mesh_embeddings_source_image'] 
                mesh_embedding_receiver_image = data['mesh_embeddings_receiver_image'] 
                print(mesh_embedding_source_image)
                print(mesh_embedding_receiver_image)
                
                
                real_RIRs = Variable(real_RIR_cpu)
                txt_embedding = Variable(txt_embedding) 
                mesh_embedding_source_image = Variable(mesh_embedding_source_image) 
                mesh_embedding_receiver_image = Variable(mesh_embedding_receiver_image) 

                
                if cfg.CUDA:
                    real_RIRs = real_RIRs.cuda()
                    txt_embedding = txt_embedding.cuda()
                    mesh_embedding_source_image = mesh_embedding_source_image.cuda()
                    mesh_embedding_receiver_image = mesh_embedding_receiver_image.cuda()
               
                #######################################################
                # (2) Generate fake images (have to modify)
                ######################################################
                with torch.no_grad():
                     mesh_embedding_source_image_latents=image_vae.encode(mesh_embedding_source_image).latent_dist.sample().reshape(self.batch_size,int(4*cfg.RAY_CASTING_IMAGE_RESOLUTION/8*cfg.RAY_CASTING_IMAGE_RESOLUTION/8))
                     mesh_embedding_receiver_image_latents=image_vae.encode(mesh_embedding_receiver_image).latent_dist.sample().reshape(self.batch_size,int(4*cfg.RAY_CASTING_IMAGE_RESOLUTION/8*cfg.RAY_CASTING_IMAGE_RESOLUTION/8))
                     #print(mesh_embedding_source_image_latents.shape)
                     #print(mesh_embedding_receiver_image_latents.shape)
                     mesh_embed=torch.concatenate((mesh_embedding_source_image_latents,mesh_embedding_receiver_image_latents),axis=1)

                #real_RIRs.detach()
                #txt_embedding.detach()
                #mesh_embed.detach()

                #print(txt_embedding.shape)#torch.Size([2, 6])
                #print(mesh_embed.shape)#torch.Size([4226, 8])

                inputs = (txt_embedding,mesh_embed)
                
                # _, fake_RIRs, mu, logvar = \
                #     nn.parallel.data_parallel(netG, inputs, self.gpus)
                
                # print("self.gpus ", [self.gpus[0]])
                _, fake_RIRs,c_code = nn.parallel.data_parallel(netG, inputs,  self.gpus)
                # input("AAA ")

                
                
                #print("Update D network")
                ############################
                # (3) Update D network
                ###########################
                netD.zero_grad()
                errD, errD_real, errD_wrong, errD_fake = \
                    compute_discriminator_loss(netD, real_RIRs, fake_RIRs,
                                               real_labels, fake_labels,
                                               c_code, self.gpus)
                
                errD_total = errD*5
                errD_total.backward()
                optimizerD.step()
                
                ############################
                # (2) Update G network
                ###########################
                # kl_loss = KL_loss(mu, logvar)
                netG.zero_grad()
                errG,L1_error,divergence_loss0,divergence_loss1,divergence_loss2,divergence_loss3,divergence_loss4,divergence_loss5,MSE_error1,MSE_error2,criterion_loss= compute_generator_loss(epoch,netD,real_RIRs, fake_RIRs,
                                              real_labels, c_code,filters, self.gpus)
                
                errG_total = errG *5#+ kl_loss * cfg.TRAIN.COEFF.KL
              
                #print(f" i={i}  errD_total={errD_total}  errG_total={errG_total}")
                errG_total.backward()
                optimizerG.step()

                # errG_total.backward()
                for p in range(2):
                    
                    # _, fake_RIRs, mu, logvar = \
                    #     nn.parallel.data_parallel(netG, inputs, self.gpus)

                    _, fake_RIRs,c_code = nn.parallel.data_parallel(netG, inputs, self.gpus)
                    netG.zero_grad()
                    errG,L1_error,divergence_loss0,divergence_loss1,divergence_loss2,divergence_loss3,divergence_loss4,divergence_loss5,MSE_error1,MSE_error2,criterion_loss  = compute_generator_loss(epoch,netD,real_RIRs, fake_RIRs,
                                              real_labels, c_code,filters, self.gpus)
                    # kl_loss = KL_loss(mu, logvar)
                    errG_total = errG *5#+ kl_loss * cfg.TRAIN.COEFF.KL
                    #print(f" i={i}  errD_total={errD_total}  errG_total={errG_total}")
                    errG_total.backward()
                    optimizerG.step()
                    ## NORMALDE BURADA BIR DE MESHNET OPTIMIZER (optimizerM) vardi , ama onu  sentetik verilerle egittik, onun icin MESHNET i egitmiyoruz.
                    # errM_total.backward()



                count = count + 1
                # print("count ",count)
                if i % 1000 == 0:
                    print("saving model ...")                    
                    save_model(netG, netD, epoch, self.model_dir)

                if generator_lr > 0.0000001 and i>0 and (i%1000==0) ):
                    rate=0.5
                    print(f"decreasing lr by 0.5 old generator_lr={generator_lr} discriminator_lr={discriminator_lr}")
                    generator_lr *= rate
                    for param_group in optimizerG.param_groups:
                        param_group['lr'] = generator_lr
                    discriminator_lr *= rate
                    for param_group in optimizerD.param_groups:
                        param_group['lr'] = discriminator_lr
                    print(f"new generator_lr={generator_lr} discriminator_lr={discriminator_lr}")

                
                if i % 10 == 0:
                    t2 = time.time()
                    print('''[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f
                     Loss_real: %.4f Loss_wrong:%.4f Loss_fake %.4f   L1_error  %.4f 
                     Total Time: %.2fsec
                         '''
                          % (epoch, self.max_epoch, i, len(data_loader),
                             errD.data, errG.data, 
                             errD_real, errD_wrong, errD_fake,L1_error*4096,(t2 - t1)),flush=True) ## MP : t1 is first set in the beggining, then reset after each print :)
                    t1=time.time()
              except:
                print(f"we had an exception  i={i}  errD_total={errD_total}  errG_total={errG_total}")
                traceback.print_exc() 
                sys.exit(1)
                continue 
            end_t = time.time()
            # print('''[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_KL: %.4f
            #          Loss_real: %.4f Loss_wrong:%.4f Loss_fake %.4f
            #          Total Time: %.2fsec
            #       '''
            #       % (epoch, self.max_epoch, i, len(data_loader),
            #          errD.data[0], errG.data[0], kl_loss.data[0],
            #          errD_real, errD_wrong, errD_fake, (end_t - start_t)))
            # print('''[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_KL: %.4f
            #          Loss_real: %.4f Loss_wrong:%.4f Loss_fake %.4f
            #          Total Time: %.2fsec
            #       '''
            #       % (epoch, self.max_epoch, i, len(data_loader),
            #          errD.data, errG.data, kl_loss.data,
            #          errD_real, errD_wrong, errD_fake, (end_t - start_t)))
            print('''[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f
                     Loss_real: %.4f Loss_wrong:%.4f Loss_fake %.4f   L1_error  %.4f 
                     Total Time: %.2fsec
                  '''
                  % (epoch, self.max_epoch, i, len(data_loader),
                     errD.data, errG.data, 
                     errD_real, errD_wrong, errD_fake,L1_error*4096,(end_t - start_t)))
            print("Divergence errors ",divergence_loss0,"  ",divergence_loss1,"  ",divergence_loss2,"  ",divergence_loss3,"  ",divergence_loss4,"  ",divergence_loss5,"  ")
            print("MSE error1  ",MSE_error1)
            print("MSE error2 ",MSE_error2)
            print("criterion_loss  ",criterion_loss)

            store_to_file ="[{}/{}][{}/{}] Loss_D: {:.4f} Loss_G: {:.4f} Loss_real: {:.4f} Loss_wrong:{:.4f} Loss_fake {:.4f}  MSE Error:{:.4f} Total Time: {:.2f}sec".format(epoch, self.max_epoch, i, len(data_loader),
                     errD.data, errG.data, errD_real, errD_wrong, errD_fake,L1_error*4096, (end_t - start_t))
            store_to_file =store_to_file+"\n" 
            with open("errors.txt", "a") as myfile:
                myfile.write(store_to_file)
            
            if (L1_error< L1_error_temp):
                L1_error_temp = L1_error
                save_model(netG, netD, epoch, self.model_dir_RT)
            if epoch % self.snapshot_interval == 0:
                save_model(netG, netD, epoch, self.model_dir)
        #
        save_model(netG, netD, self.max_epoch, self.model_dir)
        #
        # self.summary_writer.close()
        

    def sample(self,file_path,stage=1):
        if stage == 1:
            netG, _ = self.load_network_stageI()
        else:
            netG, _ = self.load_network_stageII()
        netG.eval()

        time_list =[]


       

        embedding_path = file_path
        with open(embedding_path, 'rb') as f:
            embeddings_pickle = pickle.load(f)
    
    
    
        embeddings_list =[]
        num_embeddings = len(embeddings_pickle)
        for b in range (num_embeddings):
            embeddings_list.append(embeddings_pickle[b])
    
        embeddings = np.array(embeddings_list)
    
        save_dir_GAN = "Generated_RIRs"
        mkdir_p(save_dir_GAN)    
    
             
        
        normalize_embedding = []
          
    
        batch_size = np.minimum(num_embeddings, self.batch_size)
    
       
        count = 0
        count_this = 0
        while count < num_embeddings:

            iend = count + batch_size
            if iend > num_embeddings:
                iend = num_embeddings
                count = num_embeddings - batch_size
            embeddings_batch = embeddings[count:iend]



            #txt_embedding = Variable(torch.FloatTensor(embeddings_batch))
            #if cfg.CUDA:
            #    txt_embedding = txt_embedding.cuda()
    
            #######################################################
             # (2) Generate fake images
            ######################################################
            start_t = time.time()
            #inputs = (txt_embedding,data)
            inputs = (data,)
            _, fake_RIRs,c_code = \
                nn.parallel.data_parallel(netG, inputs, [self.gpus[0]])
            end_t = time.time()
            diff_t = end_t - start_t
            time_list.append(diff_t)
    
            RIR_batch_size = batch_size #int(batch_size/2)
#            print("batch_size ", RIR_batch_size)
            channel_size = 64
            
            for i in range(channel_size):
                fs =16000
                wave_name = "RIR-"+str(count+i)+".wav"
                save_name_GAN = '%s/%s' % (save_dir_GAN,wave_name)
#                print("wave : ",save_name_GAN)
                res = {}
                res_buffer = []
                rate = 16000
                res['rate'] = rate
    
                wave_GAN = fake_RIRs[i].data.cpu().numpy()
                wave_GAN = np.array(wave_GAN[0])
    
            
                res_buffer.append(wave_GAN)
                res['samples'] = np.zeros((len(res_buffer), np.max([len(ps) for ps in res_buffer])))
                for i, c in enumerate(res_buffer):
                    res['samples'][i, :len(c)] = c

                w = WaveWriter(save_name_GAN, channels=np.shape(res['samples'])[0], samplerate=int(res['rate']))
                w.write(np.array(res['samples'])) 

#            print("counter = ",count)
            count = count+64
            count_this = count_this+1


            
