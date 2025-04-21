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

from scipy.fftpack import fft


sr=44100 # sample rate
#sr=16000 # sample rate

datas=[]
micNos=[]

def generateFrequencyPlot(datas,micNos,saveToPath):
    
     for i in range(len(datas)):
         print(i)
         #plt.clf()
         plt.subplot(1,1,1)
         plt.rcParams['font.family'] = 'sans-serif'
         data=datas[i]
         micNo=str(int(micNos[i])+1)
         #data=data[:sr]
         N=len(data)
         T=1/sr
         y_freq=fft(data)
         domain = len(y_freq) // 2
         x_freq = np.linspace(0, sr//2, N//2)

         ### frequency spectrum
         y=abs(np.array(y_freq[:domain]))
         plt.plot(x_freq, y,color='#505050',label='Microphone '+micNo)

         ### missing frequencies 
         #y=abs(np.array(y_freq[:domain]))+0.5
         #y=y.astype(int)
         #one=np.ones(y.shape[0])
         #y=1-np.minimum(y,one).astype(int)
         #plt.bar(x_freq, y,color='#505050',label='Microphone '+micNo)
         plt.legend(loc = "upper left")
         plt.xlabel('Frequency Hz')
         plt.ylabel('Amplitude')
         plt.savefig(micNo+"."+saveToPath)
         plt.close()


for i in range(6):
    wavfile=sys.argv[i*2+1]
    micNo=sys.argv[i*2+2]
    data,sr=librosa.load(wavfile,sr=sr)
    datas.append(data)
    micNos.append(micNo)

generateFrequencyPlot(datas,micNos,'frequency.plot.png')



