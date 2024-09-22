#!/usr/bin/env python

from __future__ import division
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy import fft
from scipy.io import wavfile
import scipy.signal as sig
import tempfile
import os
import pathlib
import sys
import pyaudio
import librosa
import wave
import torch
import torchaudio
import librosa.display


from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

import torchaudio
import torch
import torch.nn.functional as TF

np.set_printoptions(threshold=sys.maxsize)


sr=16000

def getSpectrogram(data):
         sample_rate = 16000 ;  num_mfccs=4096

         mfcc_transform_fn=torchaudio.transforms.MFCC(sample_rate=sample_rate,n_mfcc=num_mfccs,melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": num_mfccs, "center": False},)

         mfccs= mfcc_transform_fn( torch.Tensor(data) ).numpy()

         return mfccs

def saveRealAndGeneratedPlots(real_data,generated_data,MSE,SSIM,glitch_points,MFCC_MSE,MFCC_SSIM,MFCC_CROSS_ENTROPY,saveToPath,REAL_ON_TOP):
     #plt.clf()

     plt.clf()
     minValue=np.min(real_data)
     minValue2=np.min(generated_data)
     if minValue2 < minValue:
        minValue=minValue2

     #plt.axes().set_facecolor("black")
     plt.rcParams['font.family'] = 'sans-serif'
     plt.rcParams['font.size'] = '11'
     label1='real_data'
     label2='generated_data'
     minValue=min(np.min(real_data),np.min(generated_data))
     plt.subplot(1,1,1)
     if REAL_ON_TOP == 0 :
        plt.plot(real_data,color='#101010',label=label1)
        plt.plot(generated_data,color='#909090',label=label2)
     else:
        plt.plot(generated_data,color='#909090',label=label2)
        plt.plot(real_data,color='#101010',label=label1)

     plt.text(2200, minValue+abs(minValue)/24, f"MSE={float(MSE):.4f}\nSSIM={float(SSIM):.4f}\nGLITCH={int(len(glitch_points))}", style='italic',
        bbox={'facecolor': 'gray', 'alpha': 0.6, 'pad': 10})

     x=glitch_points
     y=generated_data[x]
     plt.scatter(x,y,color="black",alpha=0.5,label="glitch_point")

     #plt.title(title)
     plt.xlabel('Time')
     plt.ylabel('Amplitude')
     plt.legend(loc = "upper right")


     
     plt.savefig(saveToPath)

     plt.close()
     
     
     
def getGlitchPoints(reduced_sampling_rate,generated,real):
     INSENSITIVITY=5 ## DETECT ONLY VERY BIG DIFFERENCES
     glitchThreshold=np.std(np.abs(real))*INSENSITIVITY
     glitchPoints=[]
     for i in range(len(generated)):
         if  abs(abs(generated[i])-abs(real[i]) )> glitchThreshold :
             glitchPoints.append(i)
     return glitchPoints


def getLocalArgMax(limit,data):
     maximum_value=np.max(data[:limit])*4/5 # 20% error threshold for max
     return np.argmax(data[:limit]>=maximum_value)


padsize=1000


first_sound_file=sys.argv[1]
second_sound_file=sys.argv[2]


first_sound_file_data,sampling_rate=librosa.load(first_sound_file,sr=16000)
if len(first_sound_file_data.shape) > 1:
   first_sound_file_data=first_sound_file_data[:,0]+first_sound_file_data[:,1]



second_sound_file_data,sampling_rate=librosa.load(second_sound_file,sr=16000)

second_sound_file_data=second_sound_file_data[0:3500]

first_sound_file_data=first_sound_file_data[0:second_sound_file_data.shape[0]]
#first_sound_file_data=first_sound_file_data[0:4096]
#second_sound_file_data=second_sound_file_data[0:4096]
#new_second_sound_file_data=np.zeros(4096)
#new_second_sound_file_data[0:second_sound_file_data.shape[0]]=second_sound_file_data[:]
#second_sound_file_data=new_second_sound_file_data
#       second_sound_file_data=second_sound_file_data[0:4096]
real_data=first_sound_file_data
generated_data=second_sound_file_data

max_point_index_within_first_1000_points_real_data=getLocalArgMax(1000,real_data)
max_point_index_within_first_1000_points_generated_data=getLocalArgMax(1000,generated_data)



diff=int(abs(max_point_index_within_first_1000_points_real_data-max_point_index_within_first_1000_points_generated_data)/2)



print(f"diff={diff}")

if diff > 0 :
    if    max_point_index_within_first_1000_points_real_data > max_point_index_within_first_1000_points_generated_data :
           new_generated_data=np.zeros(generated_data.shape)
           new_generated_data[diff:]=generated_data[:-diff]
           generated_data=new_generated_data
            
           new_real_data=np.zeros(real_data.shape)
           new_real_data[:-diff]=real_data[diff:]
           real_data=new_real_data

    else :
           new_generated_data=np.zeros(generated_data.shape)
           #new_generated_data[diff:]=generated_data[:-diff]
           new_generated_data[:-diff]=generated_data[diff:]
           generated_data=new_generated_data

           new_real_data=np.zeros(real_data.shape)
           #new_real_data[:-diff]=real_data[diff:]
           new_real_data[diff:]=real_data[:-diff]
           real_data=new_real_data


######### BEGIN : YATAY ESITLEME (dikey esitleme zaten maksimum noktalarini esitliyerek yapilmisti)
generated_data=generated_data-np.sum(generated_data)/generated_data.shape[0]
## bu sekilde ortalamasi 0'a denk gelecek
######### END: YATAY ESITLEME (dikey esitleme zaten maksimum noktalarini esitliyerek yapilmisti)


#generated_data_sum=np.max(np.abs(generated_data))
#real_data_sum=np.max(np.abs(real_data))
#generated_data_sum=np.sum(np.abs(generated_data))
#real_data_sum=np.sum(np.abs(real_data))
#ratio_sum=real_data_sum/generated_data_sum

#generated_data=generated_data*ratio_sum

generated_data_max=np.max(np.abs(generated_data))
generated_data=generated_data/generated_data_max
real_data_max=np.max(np.abs(real_data))
real_data=real_data/real_data_max


         
MSE=np.square(np.subtract(real_data,generated_data)).mean()
MFCC_CROSS_ENTROPY=TF.cross_entropy(torch.from_numpy(real_data), torch.from_numpy(generated_data)).item()

generated_spectrogram=getSpectrogram(generated_data)
generated_spectrogram=np.reshape(generated_spectrogram,(generated_spectrogram.shape[0],generated_spectrogram.shape[1],1))
         
real_spectrogram=getSpectrogram(real_data)
real_spectrogram=np.reshape(real_spectrogram,(real_spectrogram.shape[0],real_spectrogram.shape[1],1))
generated_spectrogram=np.reshape(generated_spectrogram,(1,1,generated_spectrogram.shape[0],generated_spectrogram.shape[1]))
real_spectrogram=np.reshape(real_spectrogram,(1,1,real_spectrogram.shape[0],real_spectrogram.shape[1]))
MFCC_SSIM=ssim( torch.Tensor(generated_spectrogram), torch.Tensor(real_spectrogram), data_range=255, size_average=False).item()
MFCC_MSE=np.square(np.subtract(real_spectrogram,generated_spectrogram)).mean()
         
generated_data_tiled=np.tile(generated_data, (3, 1)) ## duplicate 1d data to 2d
real_data_tiled=np.tile(real_data, (3, 1)) ## duplicate 1d data to 2d
generated_data_tiled=np.reshape(generated_data_tiled,(1,1,generated_data_tiled.shape[0],generated_data_tiled.shape[1]))
real_data_tiled=np.reshape(real_data_tiled,(1,1,real_data_tiled.shape[0],real_data_tiled.shape[1]))
generated_data_tensor=torch.from_numpy(generated_data_tiled)
real_data_tensor=torch.from_numpy(real_data_tiled)
#/usr/local/lib/python3.8/dist-packages/pytorch_msssim/ssim.py
# data_range  = np.max(real_data)-np.min(real_data) --> bunu bi 2 olarak set ediyoruz.
#SSIM=ssim(generated_data_tensor,real_data_tensor, data_range=2.0,size_average=True).item()
SSIM=ssim(generated_data_tensor,real_data_tensor,data_range=4.0,size_average=True).item()

glitch_points=getGlitchPoints(sr,generated_data,real_data)

f = open(second_sound_file+".generated_data.txt", "w")
f.write(f"{np.array(generated_data)}")
f.close()

f = open(first_sound_file+".real_data.txt", "w")
f.write(f"{np.array(real_data)}")
f.close()

REAL_ON_TOP=0
saveRealAndGeneratedPlots(real_data,generated_data,MSE,SSIM,glitch_points,MFCC_MSE,MFCC_SSIM,MFCC_CROSS_ENTROPY,second_sound_file+'.new.png',REAL_ON_TOP)

REAL_ON_TOP=1
#Real signal is plot in front of generated one.
saveRealAndGeneratedPlots(real_data,generated_data,MSE,SSIM,glitch_points,MFCC_MSE,MFCC_SSIM,MFCC_CROSS_ENTROPY,second_sound_file+'.new_real_front.png',REAL_ON_TOP)




