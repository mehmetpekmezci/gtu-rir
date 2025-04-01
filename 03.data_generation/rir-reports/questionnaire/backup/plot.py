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

sr=16000

def pcm2float(sig, dtype='float32'):
    """Convert PCM signal to floating point with a range from -1 to 1.
    Use dtype='float32' for single precision.
    Parameters
    ----------
    sig : array_like
        Input array, must have integral type.
    dtype : data type, optional
        Desired (floating point) data type.
    Returns
    -------
    numpy.ndarray
        Normalized floating point data.
    See Also
    --------
    float2pcm, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max

def saveRealAndGeneratedPlots(data1,data2,saveToPath):
     #plt.clf()

     label1='real_data'
     label2='generated_data'
     if 'real.song' in saveToPath:
         label1='transmitted_signal_from_speaker'
         label2='received_signal_from_microphone'         
     plt.subplot(1,1,1)
     plt.plot(data1,color='r', label=label1)
     if data2 is not None:
        plt.plot(data2,color='b', label=label2)

     #plt.title(title)
     plt.xlabel('Time')
     plt.ylabel('Amlpitude')
     plt.legend(loc = "upper right")

     plt.savefig(saveToPath)

     plt.close()

padsize=1000


if len(sys.argv) < 2 :
    saveRealAndGeneratedPlots(np.zeros((4096,)),None,'plot0.png')
    exit(0)

first_sound_file=sys.argv[1]

second_sound_file=None
if len(sys.argv) > 2 :
    second_sound_file=sys.argv[2]


first_sound_file_data,sampling_rate=librosa.load(first_sound_file,sr=16000)
if len(first_sound_file_data.shape) > 1:
   first_sound_file_data=first_sound_file_data[:,0]+first_sound_file_data[:,1]
#print(np.max(first_sound_file_data))
#print(type(first_sound_file_data[0]))

second_sound_file_data=None
if second_sound_file is not None:
   second_sound_file_data,sampling_rate=librosa.load(second_sound_file,sr=16000)

if second_sound_file is not None:
    max_point_index_first_sound_file_data=np.argmax(first_sound_file_data)
    max_point_index_second_sound_file_data=np.argmax(second_sound_file_data)

    diff=int(abs(max_point_index_first_sound_file_data-max_point_index_second_sound_file_data))

    print(f"diff={diff}")

    if diff > 0 :
       new_second_sound_file_data=np.zeros(second_sound_file_data.shape[0]+diff)
       new_second_sound_file_data[diff:]=second_sound_file_data[:]
       second_sound_file_data=new_second_sound_file_data
   
    saveRealAndGeneratedPlots(first_sound_file_data,second_sound_file_data,second_sound_file+'.r.a.g.png')
else :
    if 'real.rir' in first_sound_file:
       first_sound_file_data=first_sound_file_data[:4096]
    saveRealAndGeneratedPlots(first_sound_file_data,second_sound_file_data,first_sound_file+'.single.png')


#writer = wave.open(rir_fname+'.reverbed.wav', 'wb')
#writer.setnchannels(1) # 1 channel, mono (one channel is active at an instance)
#writer.setsampwidth(2) # 16bit
#writer.setframerate(rate) # sample rate
##writer.writeframes(ir_receive_data.astype(np.int16).tostring())
##writer.writeframes(receive_data.astype(np.int16).tostring())
#writer.writeframes(receive_data.tostring())
#print(rir_fname+'.reverbed.wav')
#writer.close()

