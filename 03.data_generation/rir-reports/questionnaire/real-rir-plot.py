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


from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

import torchaudio
import torch



sr=16000



def saveRealAndGeneratedPlots(real_data,generated_data,saveToPath,real_front=False):
     

     MSE=np.square(np.subtract(real_data,generated_data)).mean()


     generated_data_tiled=np.tile(generated_data, (2, 1)) ## duplicate 1d data to 2d
     real_data_tiled=np.tile(real_data, (2, 1)) ## duplicate 1d data to 2d

     generated_data_tiled=np.reshape(generated_data_tiled,(1,1,generated_data_tiled.shape[0],generated_data_tiled.shape[1]))
     real_data_tiled=np.reshape(real_data_tiled,(1,1,real_data_tiled.shape[0],real_data_tiled.shape[1]))

     generated_data_tensor=torch.from_numpy(generated_data_tiled)
     real_data_tensor=torch.from_numpy(real_data_tiled)

     #print(f"np.max(generated_data)={np.max(generated_data)}")
     #print(f"np.max(real_data)={np.max(real_data)}")

     #    # data_range  = np.max(real_data)-np.min(real_data) --> bunu bi 2 olarak set ediyoruz.
     #    #SSIM=ssim(generated_data_tensor,real_data_tensor, data_range=2.0,size_average=True).item()
     SSIM=ssim(generated_data_tensor,real_data_tensor,data_range=4.0,size_average=True).item()

     glitch_points=getGlitchPoints(generated_data,real_data)

     plt.subplot(1,1,1)
     minValue=np.min(real_data)
     minValue2=np.min(generated_data)
     if minValue2 < minValue:
        minValue=minValue2

     if real_front:
        plt.plot(real_data,color='#101010', label='song_recored_by_microphone')
        plt.plot(generated_data,color='#909090', label='rir_convolved_with_transmitted_song')

     else:
        plt.plot(generated_data,color='#909090', label='rir_convolved_with_transmitted_song')
        plt.plot(real_data,color='#101010', label='song_recorded_by_microphone')

     plt.text(3300, minValue+abs(minValue)/11, f"MSE={float(MSE):.4f}\nSSIM={float(SSIM):.4f}\nGLITCH={int(len(glitch_points))}", style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

     #plt.title(title)
     plt.xlabel('Time')
     plt.ylabel('Amlpitude')
     plt.legend(loc = "upper right")

     x=glitch_points
     y=generated_data[x]
     plt.scatter(x,y,color="black")

     plt.savefig(saveToPath)

     plt.close()

def getGlitchPoints(generated,real):
     INSENSITIVITY=3
     glitchThreshold=np.std(np.abs(real))*INSENSITIVITY
     glitchPoints=[]
     for i in range(len(generated)):
         if  abs(abs(generated[i])-abs(real[i]) )> glitchThreshold :
             glitchPoints.append(i)
     return glitchPoints





rir_fname=sys.argv[1]

rir_data,sampling_rate=librosa.load(rir_fname,sr=16000)

rir_data=rir_data*1/np.max(rir_data)

saveRealAndGeneratedPlots(rir_data,rir_data,rir_fname+'.rir.coherence.plot.png')



