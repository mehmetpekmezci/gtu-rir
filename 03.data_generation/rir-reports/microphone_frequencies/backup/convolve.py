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




padsize=1000

SILENCE_SECONDS_TRANSMITTER=2

transmit_fname=sys.argv[1]
rir_fname=sys.argv[2]
real_song_fname=sys.argv[3]

transmit_data,sampling_rate=librosa.load(transmit_fname,sr=16000)
if len(transmit_data.shape) > 1 :
   transmit_data=transmit_data[:,0]+transmit_data[:,1]

rir_data,sampling_rate=librosa.load(rir_fname,sr=16000)
real_song,sampling_rate=librosa.load(real_song_fname,sr=16000)

rir_data=rir_data*1/np.max(rir_data)
#print(rir_data.shape)
#print(rir_data)
#print(np.max(rir_data))
#print(transmit_data)

## our reference is real_song
#transmit_data=transmit_data[:-int(transmit_data.shape[0]-real_song.shape[0])]

reverbed_data=sig.fftconvolve(transmit_data,rir_data,'full')
log_reverbed_data_power=np.log(np.sum(np.abs(reverbed_data)))
#print(log_reverbed_data_power)
reverbed_data=reverbed_data/((9/4)*log_reverbed_data_power).astype(np.float32)

#reverbed_data=reverbed_data[:-int(reverbed_data.shape[0]-real_song.shape[0])]

print(transmit_data.shape)
print(real_song.shape)
print(reverbed_data.shape)






max_point_index_real_song=np.argmax(real_song)
max_point_index_reverbed_data=np.argmax(reverbed_data)

diff=int(max_point_index_reverbed_data-max_point_index_real_song)

print(max_point_index_real_song)
print(max_point_index_reverbed_data)

print(f"diff={diff}")

if diff > 0 :
   new_reverbed_data=np.zeros(real_song.shape[0])
   new_reverbed_data[:max_point_index_real_song]=reverbed_data[diff:max_point_index_reverbed_data]
   print(f"max_point_index_real_song={max_point_index_real_song}")
   print(f"max_point_index_reverbed_data={max_point_index_reverbed_data}")
   print(f"real_song.shape[0]={real_song.shape[0]}")
   print(f"max_point_index_reverbed_data+int(real_song.shape[0]-max_point_index_real_song)={max_point_index_reverbed_data+int(real_song.shape[0]-max_point_index_real_song)}")
   print(f"new_reverbed_data.shape={new_reverbed_data.shape}")
   if max_point_index_reverbed_data+int(real_song.shape[0]-max_point_index_real_song) < reverbed_data.shape[0]:
      new_reverbed_data[max_point_index_real_song:]=reverbed_data[max_point_index_reverbed_data:max_point_index_reverbed_data+int(new_reverbed_data.shape[0]-max_point_index_real_song)]
   else:
     new_reverbed_data[max_point_index_real_song:max_point_index_real_song+int(reverbed_data.shape[0]-max_point_index_reverbed_data)]=reverbed_data[max_point_index_reverbed_data:]
   reverbed_data=new_reverbed_data
else:
   #diff=-diff
   new_reverbed_data=np.zeros(real_song.shape[0])
   new_reverbed_data[-diff:max_point_index_real_song]=reverbed_data[:max_point_index_reverbed_data]
   new_reverbed_data[max_point_index_real_song:max_point_index_real_song+int(reverbed_data.shape[0]-max_point_index_reverbed_data)]=reverbed_data[max_point_index_reverbed_data:]
   reverbed_data=new_reverbed_data



#reverbed_data=reverbed_data[int(reverbed_data.shape[0]-real_song.shape[0]):]

reverbed_data=reverbed_data.astype(np.float32)

#####ONEMLI
#reverbed_data=reverbed_data* 1/np.max(reverbed_data)
#real_song=real_song* 1/np.max(real_song)
#####ONEMLI




#receive_data=receive_data/3000000
#play_sound(receive_data,rate)
#plot_freqs(receive_data,rate)
#plot_wave(receive_data)


#reverbed_data_sum=np.sum(np.abs(reverbed_data))
#transmit_data_sum=np.sum(np.abs(transmit_data))
#ratio=transmit_data_sum/reverbed_data_sum
#reverbed_data=reverbed_data*ratio


wavfile.write( rir_fname+'.reverbed_song.wav',16000,np.array(reverbed_data).astype(np.float32))

saveRealAndGeneratedPlots(real_song,reverbed_data,rir_fname+'.reverbed_song_cherence_plot.png')
saveRealAndGeneratedPlots(real_song,reverbed_data,rir_fname+'.reverbed_song_cherence_plot.real_front.png',real_front=True)


#writer = wave.open(rir_fname+'.reverbed.wav', 'wb')
#writer.setnchannels(1) # 1 channel, mono (one channel is active at an instance)
#writer.setsampwidth(2) # 16bit
#writer.setframerate(rate) # sample rate
##writer.writeframes(ir_receive_data.astype(np.int16).tostring())
##writer.writeframes(receive_data.astype(np.int16).tostring())
#writer.writeframes(receive_data.tostring())
#print(rir_fname+'.reverbed.wav')
#writer.close()

