import os
import errno
import numpy as np
import cupy as cp
from cupyx.scipy.signal import fftconvolve
from copy import deepcopy
from miscc.config import cfg
from scipy.io.wavfile import write
from torch.nn import init
import torch
import torch.nn as nn
import torchvision.utils as vutils
from wavefile import WaveWriter, Format
# import RT60
from multiprocessing import Pool
from torch.nn.functional import normalize
import scipy.signal

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import torchaudio
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
from torch_geometric.io import read_ply
import pickle
#############################


def load_pickle(pickle_file):
     file_content=None
     try:
        with open(pickle_file, 'rb') as f:
            file_content = pickle.load(f)
     except:
        print(f"Can not load file :{pickle_file}")

     return file_content

def write_pickle(pickle_file,file_content):
        with open(pickle_file, 'wb') as f:
            pickle.dump(file_content,f, protocol=2)



def read_obj_file(path):
    return read_ply(full_mesh_path);


#PEKMEZ

def pure_ssim_loss(real_data,generated_data):
         generated_data_tiled=torch.tile(generated_data, (3, 1)) ## duplicate 1d data to 2d
         real_data_tiled=torch.tile(real_data, (3, 1)) ## duplicate 1d data to 2d
         generated_data_tiled=torch.reshape(generated_data_tiled,(1,generated_data_tiled.shape[0],generated_data_tiled.shape[1],generated_data_tiled.shape[2]))
         real_data_tiled=torch.reshape(real_data_tiled,(1,real_data_tiled.shape[0],real_data_tiled.shape[1],real_data_tiled.shape[2]))

         SSIM=ssim(generated_data_tiled,real_data_tiled, data_range=2, size_average=True).item()
         return SSIM

def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


def compute_discriminator_loss(netD, real_RIRs, fake_RIRs,
                               real_labels, fake_labels,
                               conditions, gpus):
    #print(f"fake_labels.shape={fake_labels.shape}")
    criterion = nn.BCELoss()
    batch_size = real_RIRs.size(0)
    cond = conditions.detach()
    fake = fake_RIRs.detach()
    real_features = nn.parallel.data_parallel(netD, (real_RIRs), gpus)
    #print(f"real_features.shape={real_features.shape}")
    fake_features = nn.parallel.data_parallel(netD, (fake), gpus)
    #print(f"fake_features.shape={fake_features.shape}")
    # real pairs
    #print("util conditions ",cond.size())
    inputs = (real_features, cond)
    #print(f"inputs[0].shape={inputs[0].shape}")
    real_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    #print(f"real_logits.shape={real_logits.shape}")
    #print(f"real_labels.shape={real_labels.shape}")
    errD_real = criterion(real_logits, real_labels)
    # wrong pairs
    inputs = (real_features[:(batch_size-1)], cond[1:])
    wrong_logits = \
        nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_wrong = criterion(wrong_logits, fake_labels[1:])
    # fake pairs
    inputs = (fake_features, cond)
    fake_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_fake = criterion(fake_logits, fake_labels)

    if netD.get_uncond_logits is not None:
        real_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (real_features), gpus)
        fake_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (fake_features), gpus)
        uncond_errD_real = criterion(real_logits, real_labels)
        uncond_errD_fake = criterion(fake_logits, fake_labels)
        #
        errD = ((errD_real + uncond_errD_real) / 2. +
                (errD_fake + errD_wrong + uncond_errD_fake) / 3.)
        errD_real = (errD_real + uncond_errD_real) / 2.
        errD_fake = (errD_fake + uncond_errD_fake) / 2.
    else:
        errD = errD_real + (errD_fake + errD_wrong) * 0.5
    return errD, errD_real.data, errD_wrong.data, errD_fake.data
    # return errD, errD_real.data[0], errD_wrong.data[0], errD_fake.data[0]


def  compute_generator_loss_ssim(epoch,netD,real_RIRs, fake_RIRs, real_labels, conditions,filters, gpus,cfg):
    criterion = nn.BCELoss()
    loss = nn.L1Loss() #nn.MSELoss()
    loss1 = nn.MSELoss()
    mseloss=loss1
    RT_error = 0
    cond = conditions.detach()
    fake_features = nn.parallel.data_parallel(netD, (fake_RIRs), gpus)
    inputs = (fake_features, cond)
    fake_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    L1_error = loss(real_RIRs,fake_RIRs)
    MSE_COEF=0.5
    SSIM_LOSS_COEF=0.5
    MSE_error1= MSE_COEF*mseloss(real_RIRs[:,:,0:3968],fake_RIRs[:,:,0:3968])+SSIM_LOSS_COEF*(1 - pure_ssim_loss(real_RIRs[:,:,0:3968],fake_RIRs[:,:,0:3968]))
    MSE_error2 = loss1(real_RIRs[:,:,3968:4096],fake_RIRs[:,:,3968:4096])
    ######################Energy Decay Start############################
    filter_length = 16384  # a magic number, not need to tweak this much
    mult1 = 10
    mult2 = 1
    real_ec = convert_IR2EC_batch(cp.asarray(real_RIRs), filters, filter_length)
    fake_ec = convert_IR2EC_batch(cp.asarray(fake_RIRs.to("cpu").detach()), filters, filter_length)
    divergence_loss0 = loss1(real_ec[:,:,:,0],fake_ec[:,:,:,0]) * mult1
    divergence_loss1 = loss1(real_ec[:,:,:,1],fake_ec[:,:,:,1]) * mult1
    divergence_loss2 = loss1(real_ec[:,:,:,2],fake_ec[:,:,:,2]) * mult1
    divergence_loss3 = loss1(real_ec[:,:,:,3],fake_ec[:,:,:,3]) * mult1
    divergence_loss4 = loss1(real_ec[:,:,:,4],fake_ec[:,:,:,4]) * mult1
    divergence_loss5 = loss1(real_ec[:,:,:,5],fake_ec[:,:,:,5]) * mult2
    divergence_loss = divergence_loss0 + divergence_loss1 + divergence_loss2 + divergence_loss3 + divergence_loss4 + divergence_loss5
    ######################Energy Decay End############################
    # print("criterion loss ",criterion(fake_logits, real_labels))
    MSE_ERROR11 = MSE_error1*4096*10
    MSE_ERROR21 = MSE_error2*cfg.TRAIN.BATCH_SIZE*10000 ## MP BATCH_SIZE
    MSE_ERROR = MSE_ERROR11+MSE_ERROR21
    criterion_loss = criterion(fake_logits, real_labels)
    # errD_fake = criterion(fake_logits, real_labels) + 5* 4096 * MSE_error1 #+ 40 * RT_error
    errD_fake = 2*criterion_loss + divergence_loss+(MSE_ERROR) #+ 5* 4096*MSE_error1
    if netD.get_uncond_logits is not None:
        fake_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (fake_features), gpus)
        uncond_errD_fake = criterion(fake_logits, real_labels)
        errD_fake += uncond_errD_fake
    return errD_fake, L1_error,divergence_loss0,divergence_loss1,divergence_loss2,divergence_loss3,divergence_loss4,divergence_loss5, MSE_ERROR11,MSE_ERROR21 ,criterion_loss #,RT_error


def compute_generator_loss(epoch,netD,real_RIRs, fake_RIRs, real_labels, conditions,filters, gpus,cfg):
    criterion = nn.BCELoss()
    loss = nn.L1Loss() #nn.MSELoss()
    loss1 = nn.MSELoss()
    RT_error = 0
    cond = conditions.detach()
    fake_features = nn.parallel.data_parallel(netD, (fake_RIRs), gpus)
    inputs = (fake_features, cond)
    fake_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    L1_error = loss(real_RIRs,fake_RIRs)
    MSE_error1 = loss1(real_RIRs[:,:,0:3968],fake_RIRs[:,:,0:3968])
    MSE_error2 = loss1(real_RIRs[:,:,3968:4096],fake_RIRs[:,:,3968:4096])
    ######################Energy Decay Start############################
    filter_length = 16384  # a magic number, not need to tweak this much
    mult1 = 10
    mult2 = 1
    real_ec = convert_IR2EC_batch(cp.asarray(real_RIRs), filters, filter_length)
    fake_ec = convert_IR2EC_batch(cp.asarray(fake_RIRs.to("cpu").detach()), filters, filter_length)
    divergence_loss0 = loss1(real_ec[:,:,:,0],fake_ec[:,:,:,0]) * mult1
    divergence_loss1 = loss1(real_ec[:,:,:,1],fake_ec[:,:,:,1]) * mult1
    divergence_loss2 = loss1(real_ec[:,:,:,2],fake_ec[:,:,:,2]) * mult1
    divergence_loss3 = loss1(real_ec[:,:,:,3],fake_ec[:,:,:,3]) * mult1
    divergence_loss4 = loss1(real_ec[:,:,:,4],fake_ec[:,:,:,4]) * mult1
    divergence_loss5 = loss1(real_ec[:,:,:,5],fake_ec[:,:,:,5]) * mult2
    divergence_loss = divergence_loss0 + divergence_loss1 + divergence_loss2 + divergence_loss3 + divergence_loss4 + divergence_loss5
    ######################Energy Decay End############################
    # print("criterion loss ",criterion(fake_logits, real_labels))
    MSE_ERROR11 = MSE_error1*4096*10
    MSE_ERROR21 = MSE_error2*cfg.TRAIN.BATCH_SIZE*10000 ## MP BATCH_SIZE
    MSE_ERROR = MSE_ERROR11+MSE_ERROR21
    criterion_loss = criterion(fake_logits, real_labels)
    # errD_fake = criterion(fake_logits, real_labels) + 5* 4096 * MSE_error1 #+ 40 * RT_error
    errD_fake = 2*criterion_loss + divergence_loss+(MSE_ERROR) #+ 5* 4096*MSE_error1
    if netD.get_uncond_logits is not None:
        fake_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (fake_features), gpus)
        uncond_errD_fake = criterion(fake_logits, real_labels)
        errD_fake += uncond_errD_fake
    return errD_fake, L1_error,divergence_loss0,divergence_loss1,divergence_loss2,divergence_loss3,divergence_loss4,divergence_loss5, MSE_ERROR11,MSE_ERROR21 ,criterion_loss #,RT_error


#############################
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


#############################
def save_RIR_results(data_RIR, fake, epoch, RIR_dir,cfg):
   return
   ## no need to generate sample RIRs
   try:
    #num = 64# cfg.VIS_COUNT
    num = cfg.TRAIN.BATCH_SIZE #MPEKMEZCI
    fake = fake[0:num]
    # data_RIR is changed to [0,1]
    if data_RIR is not None:
        data_RIR = data_RIR[0:num]
        for i in range(num):
            # #print("came 1")
            real_RIR_path = RIR_dir+"/real_sample"+str(i)+"_epoch_"+str(epoch)+".wav" 
            fake_RIR_path = RIR_dir+"/fake_sample"+str(i)+"_epoch_"+str(epoch)+".wav"
            fs =16000

            real_IR = np.array(data_RIR[i].to("cpu").detach())
            fake_IR = np.array(fake[i].to("cpu").detach())

            r = WaveWriter(real_RIR_path, channels=1, samplerate=fs)
            r.write(np.array(real_IR))
            r.close()
            f = WaveWriter(fake_RIR_path, channels=1, samplerate=fs)
            f.write(np.array(fake_IR))           
            f.close()


            # write(real_RIR_path,fs,real_IR)
            # write(fake_RIR_path,fs,fake_IR)


            # write(real_RIR_path,fs,real_IR)
            # write(fake_RIR_path,fs,fake_IR)

        # vutils.save_image(
        #     data_RIR, '%s/real_samples.png' % RIR_dir,
        #     normalize=True)
        # # fake.data is still [-1, 1]
        # vutils.save_image(
        #     fake.data, '%s/fake_samples_epoch_%03d.png' %
        #     (RIR_dir, epoch), normalize=True)
    else:
        for i in range(num):
            # #print("came 2")
            fake_RIR_path = RIR_dir+"/small_fake_sample"+str(i)+"_epoch_"+str(epoch)+".wav"
            fs =16000
            fake_IR = np.array(fake[i].to("cpu").detach())
            f = WaveWriter(fake_RIR_path, channels=1, samplerate=fs)
            f.write(np.array(fake_IR))
            f.close()
            
            # write(fake_RIR_path,fs,fake[i].astype(np.float32))

        # vutils.save_image(
        #     fake.data, '%s/lr_fake_samples_epoch_%03d.png' %
        #     (RIR_dir, epoch), normalize=True)

   except :  # Python >2.5
       print("MP: We had an exception while writing generated RIR WAV to disk")


def save_model(netG, netD,mesh_net, epoch, model_dir):
    torch.save(
        netG.state_dict(),
        '%s/netG_epoch_%d.pth' % (model_dir, epoch))
    torch.save(
        mesh_net.state_dict(),
        '%s/mesh_net_epoch_%d.pth' % (model_dir, epoch))
    torch.save(
        netD.state_dict(),
        '%s/netD_epoch_last.pth' % (model_dir))
    #print('Save G/D models')


def save_model(netG, netD, epoch, model_dir):
    torch.save(
        netG.state_dict(),
        '%s/netG_epoch_%d.pth' % (model_dir, epoch))
    torch.save(
        netD.state_dict(),
        '%s/netD_epoch_last.pth' % (model_dir))
    #print('Save G/D models')


def save_mesh_model(mesh_net, epoch, model_dir):
    torch.save(
        mesh_net.state_dict(),
        '%s/mesh_net_epoch_%d.pth' % (model_dir, epoch))


def save_mesh_final_model(mesh_net):
    mkdir_p(cfg.PRE_TRAINED_MODELS_DIR)
    torch.save(mesh_net.state_dict(),cfg.PRE_TRAINED_MODELS_DIR+'/'+cfg.MESH_NET_GAE_FILE)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def convert_IR2EC(rir, filters, filter_length):
    subband_ECs = np.zeros((len(rir), filters.shape[1]))
    for i in range(filters.shape[1]):
        subband_ir = scipy.signal.fftconvolve(rir, filters[:, i])
        subband_ir = subband_ir[(filter_length - 1):]
        squared = np.square(subband_ir[:len(rir)])
        subband_ECs[:, i] = np.cumsum(squared[::-1])[::-1]
    return subband_ECs

def convert_IR2EC_batch(rir, filters, filter_length):
    # filters = cp.asarray([[filters]])
    rir = rir[:,:,0:3968]
    subband_ECs = cp.zeros((rir.shape[0],rir.shape[1],rir.shape[2], filters.shape[3]))
    for i in range(filters.shape[3]):
        subband_ir = fftconvolve(rir, filters[:,:,:, i])
        subband_ir = subband_ir[:,:,(filter_length - 1):]
        squared = cp.square(subband_ir[:,:,:rir.shape[2]])
        subband_ECs[:, :,:,i] = cp.cumsum(squared[:,:,::-1],axis=2)[:,:,::-1]
    subband_ECs = torch.tensor(subband_ECs,device='cuda')
    return subband_ECs



def generate_complementary_filterbank(
        fc=[125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0],
        fs=16000,
        filter_order=4,
        filter_length=16384,
        power=True):
    """Return a zero-phase power (or amplitude) complementary filterbank via Butterworth prototypes.
    Parameters:
        fc - filter center frequencies
        fs - sampling rate
        filter_order - order of the prototype Butterworth filters
        filter_length - length of the resulting zero-phase FIR filters
        power - boolean to set if the filter is power or amplitude complementary
    """

    # sort in increasing cutoff
    fc = np.sort(fc)

    assert fc[-1] <= fs/2

    numFilts = len(fc)
    nbins = filter_length
    signal_z1 = np.zeros(2 * nbins)
    signal_z1[0] = 1
    irBands = np.zeros((2 * nbins, numFilts))

    for i in range(numFilts - 1):
        wc = fc[i] / (fs/2.0)
        # if wc >= 1:
        #     wc = .999999

        B_low, A_low = scipy.signal.butter(filter_order, wc, btype='low')
        B_high, A_high = scipy.signal.butter(filter_order, wc, btype='high')


        # Store the low band
        irBands[:, i] = scipy.signal.lfilter(B_low, A_low, signal_z1)

        # Store the high
        signal_z1 = scipy.signal.lfilter(B_high, A_high, signal_z1)

        # Repeat for the last band of the filter bank
    irBands[:, -1] = signal_z1

    # Compute power complementary filters
    if power:
        ir2Bands = np.real(np.fft.ifft(np.square(np.abs(np.fft.fft(irBands, axis=0))), axis=0))
    else:
        ir2Bands = np.real(np.fft.ifft(np.abs(np.abs(np.fft.fft(irBands, axis=0))), axis=0))

    ir2Bands = np.concatenate((ir2Bands[nbins:(2 * nbins), :], ir2Bands[0:nbins, :]), axis=0)

    return ir2Bands



def convert_to_trimesh(pos,faces):
    r"""Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`trimesh.Trimesh`.
    data (torch_geometric.data.Data): The data object.
    """
    #print(faces)
    #print(f"len(pos)={len(pos)}")
    #print(pos)
    mesh=trimesh.Trimesh(vertices=pos,faces=faces)
    return mesh

def save_mesh_as_obj(mesh,path):
    with open(path,'w') as meshfile:
         meshfile.write(mesh.export(file_type='obj'))
    
def save_pos_face_as_obj(pos,faces,path):
    with open(path, "w") as objfile:
         objfile.write(trimesh.Trimesh(vertices=pos,faces=faces).export(file_type='obj'))
         print(f"{path} is written")
 
 
 
 
def coordinates_patch_to_pos_and_face(cordinates_patcha):
    pos=[]
    faces=[]
    triangle_cordinates=cordinates_patcha.reshape(int(cordinates_patcha.shape[0]*cordinates_patcha.shape[1]),9)
    for triangle_cordinate in triangle_cordinates:
        v1=list(triangle_cordinate[:3])
        v2=list(triangle_cordinate[3:6])
        v3=list(triangle_cordinate[6:9])
        if v1 not in pos:
           pos.append(v1) 
        if v2 not in pos:
           pos.append(v2) 
        if v3 not in pos:
           pos.append(v3) 
    
    for triangle_cordinate in triangle_cordinates:
        v1=list(triangle_cordinate[:3])
        v2=list(triangle_cordinate[3:6])
        v3=list(triangle_cordinate[6:9])
        
        v1_index=0
        v2_index=0
        v3_index=0

        v1_found=False
        v2_found=False
        v3_found=False
        
        
        i=0
        for node in pos :
            if v1 == node :
               v1_index=i
               v1_found=True   
            if v2 == node :
               v2_index=i
               v2_found=True   
            if v3 == node :
               v3_index=i
               v3_found=True   
            if v1_found and v2_found and v3_found :
               break
            i=i+1
        if [v1_index,v2_index,v3_index] not in faces :
           faces.append([v1_index,v2_index,v3_index])
    return pos,faces
            
def save_coordinate_patch_as_obj(cordinates_patcha,path):
    pos,faces=coordinates_patch_to_pos_and_face(cordinates_patcha)
    save_pos_faces_as_obj(pos,faces,path)

def save_pos_faces_as_obj(pos,faces,path):
    with open(path, "w") as objfile:
         objfile.write(trimesh.Trimesh(vertices=pos,faces=faces).export(file_type='obj'))
         print(f"{path} is written")


 
    
def plot_mesh(trimesh,MESH_dir,file_name):
    mesh=trimesh
    if len(mesh.faces)==0:
        return
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(mesh.vertices[:, 2], mesh.vertices[:,0], triangles=mesh.faces, Z=mesh.vertices[:,1])
    plt.savefig(MESH_dir+"/graph."+file_name+".png")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(mesh.vertices[:, 2],mesh.vertices[:,0],mesh.vertices[:,1])
    plt.savefig(MESH_dir+"/graph."+file_name+"-scatter.png")
    plt.close()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(mesh.vertices[:10, 2],mesh.vertices[:10,0],mesh.vertices[:10,1],color=['red','green','blue','orange','purple','brown','pink','gray','olive','cyan'],s=[20,40,60,80,100,120,140,160,180,200])
    plt.savefig(MESH_dir+"/graph."+file_name+"-scatter.10points.png")
    plt.close()
    
    
    
def plot_points(points,MESH_dir,file_name):
    if len(points)==0:
        return
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 2],points[:,0],points[:,1])
    plt.savefig(MESH_dir+"/graph."+file_name+"-scatter-direct-points.png")
    plt.close()

def edge_index_to_face(edge_index):

    #print(f"edge_index.shape={edge_index.shape}")

    edge_index_length=edge_index.shape[1]

    #print(f"edge_index_length={edge_index_length}")

    faces=[]

    connection_list={}
    for index1 in range(edge_index_length):
     node1=edge_index[0][index1].item()   
     node2=edge_index[1][index1].item()   
     if node1 not in connection_list:
        connection_list[node1]=[]
     if node2 not in connection_list:
        connection_list[node2]=[]
     connection_list[node1].append(node2)
     connection_list[node2].append(node1)

    #print(f"len(connection_list)={len(connection_list)}")

    for node1 in connection_list:
        for node2 in connection_list[node1]:
            for node3 in connection_list[node2]:
                if node1 in connection_list[node3]:
                   s=[node1,node2,node3]
                   s.sort()
                   if s not in faces:
                      faces.append(s) 
                        
    print(f" len(faces)={len(faces)}")
    return faces

    

def adj_to_face(A):
    faces=[]
    indices=np.argwhere(A ==1 )

    k_indices=sorted(set(list(indices[:,1])))
    
    for i_j in indices:
        i=i_j[0]
        j=i_j[1]

        for k in k_indices:
          if i!=j and i!=k and j!=k and A[j,k]==1 and A[k,i]==1 :
             s=[i,j,k]
             s.sort()
             if s not in faces:
                faces.append(s) 
                if len(faces)%2000 == 0 :
                    #print(f"adj_to_face : THRESHOLD={THRESHOLD}  i:{i}/{A.shape[0]} j:{j}/{A.shape[0]} k:{k}/{A.shape[0]} len(faces)={len(faces)}")
                    print(f"adj_to_face :  i:{i}/{A.shape[0]} j:{j}/{A.shape[0]} k:{k}/{A.shape[0]} len(faces)={len(faces)}")
                 


                        
    print(f"1. len(faces)={len(faces)}")
    return faces

    



