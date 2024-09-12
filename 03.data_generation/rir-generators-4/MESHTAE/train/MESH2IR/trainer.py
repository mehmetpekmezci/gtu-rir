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
    def __init__(self, output_dir,gae_mesh_net):


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

        self.gae_mesh_net=gae_mesh_net
        self.gae_mesh_net.eval()
        self.cfg=cfg
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        #self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        self.batch_size = cfg.TRAIN.BATCH_SIZE 
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

        if cfg.PRE_TRAINED_MODELS_DIR!= '' and cfg.MESH_NET_GAE_FILE != '' and os.path.exists(cfg.PRE_TRAINED_MODELS_DIR+'/'+cfg.MESH_NET_GAE_FILE):
            state_dict = \
                torch.load( cfg.PRE_TRAINED_MODELS_DIR+'/'+cfg.MESH_NET_GAE_FILE,
                           map_location=lambda storage, loc: storage)
            self.gae_mesh_net.load_state_dict(state_dict)
            print('Load GAE MESH NET from: ', cfg.PRE_TRAINED_MODELS_DIR+'/'+cfg.MESH_NET_GAE_FILE)
            
        SOURCE_RECEIVER_XYZ_DIM=6    
        summary(netG,[(self.batch_size,SOURCE_RECEIVER_XYZ_DIM),(self.batch_size,cfg.LATENT_VECTOR_SIZE)] )
        summary(netD,(self.batch_size,1,cfg.RIRSIZE) )
        
        if cfg.CUDA:
            netG.cuda()
            netD.cuda()
            #self.gae_mesh_net.cuda()
        return netG, netD

    # # ############# For training stageII GAN  #############
    # def load_network_stageII(self):
    #     from model import STAGE1_G, STAGE2_G, STAGE2_D

    #     Stage1_G = STAGE1_G()
    #     netG = STAGE2_G(Stage1_G)
    #     netG.apply(weights_init)
    #     print(netG)
    #     if cfg.NET_G != '':
    #         state_dict = \
    #             torch.load(cfg.NET_G,
    #                        map_location=lambda storage, loc: storage)
    #         netG.load_state_dict(state_dict)
    #         print('Load from: ', cfg.NET_G)
    #     elif cfg.STAGE1_G != '':
    #         state_dict = \
    #             torch.load(cfg.STAGE1_G,
    #                        map_location=lambda storage, loc: storage)
    #         netG.STAGE1_G.load_state_dict(state_dict)
    #         print('Load from: ', cfg.STAGE1_G)
    #     else:
    #         print("Please give the Stage1_G path")
    #         return

    #     netD = STAGE2_D()
    #     netD.apply(weights_init)
    #     if cfg.NET_D != '':
    #         state_dict = \
    #             torch.load(cfg.NET_D,
    #                        map_location=lambda storage, loc: storage)
    #         netD.load_state_dict(state_dict)
    #         print('Load from: ', cfg.NET_D)
    #     print(netD)

    #     if cfg.CUDA:
    #         netG.cuda()
    #         netD.cuda()
    #     return netG, netD

    def train(self, data_loader, stage=1):

        self.gae_mesh_net.eval()
        if stage == 1:
            netG, netD = self.load_network_stageI()
        # else:
        #     netG, netD = self.load_network_stageII()
        
        netG.to(device='cuda')
        self.gae_mesh_net.to(device='cuda:1')
        #self.gae_mesh_net.cpu()
        # nz = cfg.Z_DIM
        batch_size = self.batch_size
        # noise = Variable(torch.FloatTensor(batch_size, nz))
        # fixed_noise = \
        #     Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1),
        #              volatile=True)
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
            if epoch % lr_decay_step == 0 and epoch > 0:
                generator_lr *= 0.85#0.5
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = generator_lr
                discriminator_lr *= 0.85#0.5
                for param_group in optimizerD.param_groups:
                    param_group['lr'] = discriminator_lr

            
            # b1=0
            # b2=0
            # b3=0
            for i, data in enumerate(data_loader, 0):
              errD_total=0
              errG_total=0              
              try: 
                if  len(data) < self.batch_size :
#                    print("len(data)",len(data))
                    continue
                    ## MP : because we depend on batch_size in CustomInnerProductDecoder in model.py

                ######################################################
                # (1) Prepare training data
                ######################################################
                # real_RIR_cpu, txt_embedding = data
                # b1 = time.time()
                # print("Time 1   ",(b1-b3))
               
                
                
                real_RIR_cpu = torch.from_numpy(np.array(data['RIR']))
                txt_embedding = torch.from_numpy(np.array(data['embeddings']))
                
                #data.pop('RIR')
                #data.pop('embeddings')
                
                real_RIRs = Variable(real_RIR_cpu)
                txt_embedding = Variable(txt_embedding)

                #data['edge_index'] = Variable(data['edge_index']).long()
                #data['pos'] = Variable(data['pos']).float()




                triangle_coordinates=Variable(data['triangle_coordinates']).float()
                normals=Variable(data['normals']).float()
                centers=Variable(data['centers']).float()
                areas=Variable(data['areas']).float()
                #print(f"triangle_coordinates.shape = {triangle_coordinates.shape} normals.shape={normals.shape} centers.shape={centers.shape} areas.shape={areas.shape} ")
                faceDataDim=triangle_coordinates.shape[2]+centers.shape[2]+normals.shape[2]+areas.shape[2]
                faceData=torch.cat((triangle_coordinates,normals,centers,areas),2)


                
                if cfg.CUDA:
                    real_RIRs = real_RIRs.cuda()
                    txt_embedding = txt_embedding.cuda()
                    #faceData=faceData.to(device='cuda:1')
                    #edge_weights = edge_weights.cuda()
                    #data = data.cuda()
                
                #######################################################
                # (2) Generate fake images (have to modify)
                ######################################################
                
                mesh_embeds=[]
                #while gae_mesh_net_batch_count<cfg.TRAIN.BATCH_SIZE*self.num_gpus :
                for batchno in range(cfg.TRAIN.BATCH_SIZE): 
                    faceData_1_batch=faceData[batchno].unsqueeze(0).detach().to(device='cuda:1')
                    #print(f"faceData_1_batch.shape={faceData_1_batch.shape}")
                    faceData_1_batch_predicted , latent_vector_1_batch =  self.gae_mesh_net(faceData_1_batch)
                    mesh_embeds.append(latent_vector_1_batch.detach().cpu())
                
                mesh_embed=torch.vstack(mesh_embeds)      
                
#                print(txt_embedding.shape)#torch.Size([2, 6])
#                print(mesh_embed.shape)#torch.Size([4226, 8])

                if cfg.CUDA:
                    mesh_embed = mesh_embed.cuda()

                inputs = (txt_embedding,mesh_embed)
                
                # _, fake_RIRs, mu, logvar = \
                #     nn.parallel.data_parallel(netG, inputs, self.gpus)
                
                # print("self.gpus ", [self.gpus[0]])
                _, fake_RIRs,c_code = nn.parallel.data_parallel(netG, inputs,  self.gpus)
                # input("AAA ")

                
                
#                print(fake_RIRs)
#                print(real_labels)
#                print(fake_labels)
#                print(c_code)
                #print("Update D network")
                ############################
                # (3) Update D network
                ###########################
                netD.zero_grad()
                errD, errD_real, errD_wrong, errD_fake = \
                    compute_discriminator_loss(netD, real_RIRs, fake_RIRs,
                                               real_labels, fake_labels,
                                               c_code, self.gpus)
#                print("d ****************")
#                print(errD)
#                print(errD_real)
#                print(errD_wrong)
#                print(errD_fake)
#                print("d ****************")
                
                errD_total = errD*5
#                print(f"  i={i}  errD_total={errD_total}  errG_total={errG_total}")
                errD_total.backward()
                optimizerD.step()
                
                #print("Update G network")
                ############################
                # (2) Update G network
                ###########################
                # kl_loss = KL_loss(mu, logvar)
                netG.zero_grad()
                errG,L1_error,divergence_loss0,divergence_loss1,divergence_loss2,divergence_loss3,divergence_loss4,divergence_loss5,MSE_error1,MSE_error2,criterion_loss= compute_generator_loss(epoch,netD,real_RIRs, fake_RIRs,
                                              real_labels, c_code,filters, self.gpus)
                
#                print(L1_error)
#                print(divergence_loss0)
#                print(divergence_loss1)
#                print(divergence_loss2)
#                print(divergence_loss3)
#                print(divergence_loss4)
#                print(divergence_loss5)
#                print(MSE_error1)
#                print(MSE_error2)
#                print(criterion_loss)
#                print(errG)
                
                errG_total = errG *5#+ kl_loss * cfg.TRAIN.COEFF.KL
              
#                print(f" i={i}  errD_total={errD_total}  errG_total={errG_total}")
                errG_total.backward()
                optimizerG.step()

                # errG_total.backward()
                for p in range(2):
                    
                    # _, fake_RIRs, mu, logvar = \
                    #     nn.parallel.data_parallel(netG, inputs, self.gpus)

                    #mesh_embed = self.gae_mesh_net.encode(data['pos'], data['edge_index'],data['edge_weights'], data.batch)
                    #mesh_embed = self.gae_mesh_net.encode(data['pos_dense'], data['edge_index_dense']).squeeze()
                    mesh_embeds=[]
                    #while gae_mesh_net_batch_count<cfg.TRAIN.BATCH_SIZE*self.num_gpus :
                    for batchno in range(cfg.TRAIN.BATCH_SIZE): 
                        faceData_1_batch=faceData[batchno].unsqueeze(0).detach().to(device='cuda:1')    
    #                    print(f"faceData_1_batch.shape={faceData_1_batch.shape}")
                        faceData_1_batch_predicted , latent_vector_1_batch =  self.gae_mesh_net(faceData_1_batch)
                        mesh_embeds.append(latent_vector_1_batch.detach().cpu())
                
                    mesh_embed=torch.vstack(mesh_embeds)      
                
    #                print(txt_embedding.shape)#torch.Size([2, 6])
    #                print(mesh_embed.shape)#torch.Size([4226, 8])

                    if cfg.CUDA:
                        mesh_embed = mesh_embed.cuda()

                    inputs = (txt_embedding,mesh_embed)

                    _, fake_RIRs,c_code = nn.parallel.data_parallel(netG, inputs, self.gpus)
                    netG.zero_grad()
                    errG,L1_error,divergence_loss0,divergence_loss1,divergence_loss2,divergence_loss3,divergence_loss4,divergence_loss5,MSE_error1,MSE_error2,criterion_loss  = compute_generator_loss(epoch,netD,real_RIRs, fake_RIRs,
                                              real_labels, c_code,filters, self.gpus)
                    # kl_loss = KL_loss(mu, logvar)
                    errG_total = errG *5#+ kl_loss * cfg.TRAIN.COEFF.KL
#                    print(f" i={i}  errD_total={errD_total}  errG_total={errG_total}")
                    errG_total.backward()
                    optimizerG.step()
                    ## NORMALDE BURADA BIR DE MESHNET OPTIMIZER (optimizerM) vardi , ama onu  sentetik verilerle egittik, onun icin MESHNET i egitmiyoruz.
                    # errG_total.backward()


                count = count + 1
                # print("count ",count)
                if i % 1000 == 0:
                    print("saving model ...")                    
                    save_model(netG, netD, epoch, self.model_dir)

                if i % 10000 == 0:
                    print("decreasing lr by 0.85")
                    generator_lr *= 0.85#0.5
                    for param_group in optimizerG.param_groups:
                        param_group['lr'] = generator_lr
                    discriminator_lr *= 0.85#0.5
                    for param_group in optimizerD.param_groups:
                        param_group['lr'] = discriminator_lr

                
                if i % 50 == 0:
                    t2 = time.time()
                    print('''[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f
                     Loss_real: %.4f Loss_wrong:%.4f Loss_fake %.4f   L1_error  %.4f 
                     Total Time: %.2fsec
                         '''
                          % (epoch, self.max_epoch, i, len(data_loader),
                             errD.data, errG.data, 
                             errD_real, errD_wrong, errD_fake,L1_error*4096,(t2 - t1)),flush=True) ## MP : t1 is first set in the beggining, then reset after each print :)
                    t1=time.time()
                    # summary_D = summary.scalar('D_loss', errD.data[0])
                    # summary_D_r = summary.scalar('D_loss_real', errD_real)
                    # summary_D_w = summary.scalar('D_loss_wrong', errD_wrong)
                    # summary_D_f = summary.scalar('D_loss_fake', errD_fake)
                    # summary_G = summary.scalar('G_loss', errG.data[0])
                    # summary_KL = summary.scalar('KL_loss', kl_loss.data[0])
                    # summary_D = summary.scalar('D_loss', errD.data)
                    # summary_D_r = summary.scalar('D_loss_real', errD_real)
                    # summary_D_w = summary.scalar('D_loss_wrong', errD_wrong)
                    # summary_D_f = summary.scalar('D_loss_fake', errD_fake)
                    # summary_G = summary.scalar('G_loss', errG.data)
                    # summary_KL = summary.scalar('KL_loss', kl_loss.data)

                    # self.summary_writer.add_summary(summary_D, count)
                    # self.summary_writer.add_summary(summary_D_r, count)
                    # self.summary_writer.add_summary(summary_D_w, count)
                    # self.summary_writer.add_summary(summary_D_f, count)
                    # self.summary_writer.add_summary(summary_G, count)
                    # self.summary_writer.add_summary(summary_KL, count)

                    ## save the image result for each epoch
                    #inputs = (txt_embedding,mesh_embed)
                    #lr_fake, fake, _ = \
                    #    nn.parallel.data_parallel(netG, inputs, self.gpus)
                    #if(epoch%self.snapshot_interval==0):
                    #    #print(f"epoch:{epoch}")
                    #    #print("real_RIR_cpu:")
                    #    #print(real_RIR_cpu)
                    #    save_RIR_results(real_RIR_cpu, fake, epoch, self.RIR_dir,batch_size)
                    #    if lr_fake is not None:
                    #        save_RIR_results(None, lr_fake, epoch, self.RIR_dir,batch_size)
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



            txt_embedding = Variable(torch.FloatTensor(embeddings_batch))
            if cfg.CUDA:
                txt_embedding = txt_embedding.cuda()
    
            #######################################################
             # (2) Generate fake images
            ######################################################
            start_t = time.time()
            inputs = (txt_embedding,data)
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


            
