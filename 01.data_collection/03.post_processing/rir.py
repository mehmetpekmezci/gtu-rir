#!/usr/bin/env python


### https://dsp.stackexchange.com/questions/41696/calculating-the-inverse-filter-for-the-exponential-sine-sweep-method

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


def play_sound(sound_data,SOUND_RECORD_SAMPLING_RATE):
  p = pyaudio.PyAudio()
  stream = p.open(format=pyaudio.paInt16, channels=1, rate=SOUND_RECORD_SAMPLING_RATE, output=True)
  #stream = p.open(format=pyaudio.paFloat32, channels=1, rate=SOUND_RECORD_SAMPLING_RATE, output=True)
  for i in range(int(sound_data.shape[0]/SOUND_RECORD_SAMPLING_RATE)):
     print("Playing : "+str(i*SOUND_RECORD_SAMPLING_RATE)+" -- "+str((i+1)*SOUND_RECORD_SAMPLING_RATE))
     stream.write(sound_data[i*SOUND_RECORD_SAMPLING_RATE:(i+1)*SOUND_RECORD_SAMPLING_RATE],SOUND_RECORD_SAMPLING_RATE)
  if int(sound_data.shape[0]/SOUND_RECORD_SAMPLING_RATE) * SOUND_RECORD_SAMPLING_RATE < sound_data.shape[0] :
     print("Playing : "+str(int(sound_data.shape[0]/SOUND_RECORD_SAMPLING_RATE) * SOUND_RECORD_SAMPLING_RATE)+" -- "+str(sound_data.shape[0]))
     stream.write(sound_data[int(sound_data.shape[0]/SOUND_RECORD_SAMPLING_RATE) * SOUND_RECORD_SAMPLING_RATE:],SOUND_RECORD_SAMPLING_RATE)
  stream.stop_stream()
  stream.close()
  p.terminate()

def plot_freqs(signal,sample_rate):
    #sampFreq=sample_rate
    #fft_spectrum = np.fft.rfft(signal)
    #freq = np.fft.rfftfreq(signal.size, d=1./sampFreq)
    #fft_spectrum_abs = np.abs(fft_spectrum)
    #plt.plot(freq, fft_spectrum_abs)
    #plt.xlabel("frequency, Hz")
    #plt.ylabel("Amplitude, units")
    #plt.show()

    #FFT
    t = np.arange(signal.shape[0])
    freq = np.fft.fftfreq(t.shape[-1])*sample_rate
    sp = np.fft.fft(signal)

    # Plot spectrum
    plt.plot(freq, abs(sp.real))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Spectrum of Signal')
    plt.show()
    #plt.xlim((0, 2000))
    #plt.grid()



def plot_wave(sound):
    plt.subplot(1,1,1)
    plt.plot(sound, 'b')
    plt.xlabel("Wave Plot")
    #plt.tight_layout()
    plt.show()

padsize=1000

SILENCE_SECONDS_TRANSMITTER=2

receive_fname=sys.argv[1]
transmit_fname=sys.argv[2]


transmit_data,rate=librosa.load(transmit_fname,sr=44100,mono=True)
transmit_data=np.array(transmit_data).astype(np.float32)
transmit_data=transmit_data[SILENCE_SECONDS_TRANSMITTER*rate:-SILENCE_SECONDS_TRANSMITTER*rate]

#play_sound(transmit_data,rate)
#plot_freqs(transmit_data,rate)
#plot_wave(transmit_data)

receive_data_original,rate=librosa.load(receive_fname,sr=44100,mono=True)
receive_data_original=np.array(receive_data_original).astype(np.float32)
#receive_data=receive_data_original[SILENCE_SECONDS_TRANSMITTER*rate:-SILENCE_SECONDS_TRANSMITTER*rate]
receive_data=receive_data_original[:transmit_data.shape[0]]

#play_sound(receive_data,rate)
#plot_freqs(receive_data,rate)
#plot_wave(receive_data)

print("rate="+str(rate))
print("receive_data.shape="+str(receive_data.shape))
print("transmit_data.shape="+str(transmit_data.shape))
print("transmit_data.dtype="+str(transmit_data.dtype))
print("transmit_data.len="+str(len(transmit_data)))
##print("transmit_data[15]="+str(transmit_data[15]))
## select section of data that is noise



f1 = 15
f2 = 22e3
T = 14
fs = 44100
#t = np.arange(0, T*fs)/fs
t = np.arange(0, receive_data.shape[0])/fs
R = np.log(f2/f1)

# ESS generation
#x = np.sin((2*np.pi*f1*T/R)*(np.exp(t*R/T)-1))
x=transmit_data
# Inverse filter
k = np.exp(t*R/T)
f = x[::-1]/k
    
ir_receive_data= sig.fftconvolve(receive_data,f, mode='same')

#plot_wave(transmit_data)
#plot_wave(receive_data)
#plot_wave(ir_receive_data)

ir_receive_data=ir_receive_data/10000000 # 10^7
#plot_wave(ir_receive_data.astype(np.int16))

#print(transmit_data[0:30])
#print(receive_data[0:30])
#print(ir_receive_data[0:30])

#print(transmit_data[0:30].astype(np.int16))
#print(receive_data[0:30].astype(np.int16))
#print(ir_receive_data[0:30].astype(np.int16))

#print(np.max(transmit_data.astype(np.int16)))
#print(np.max(receive_data.astype(np.int16)))
#print(np.max(ir_receive_data.astype(np.int16)))

#print(transmit_data.shape)
#print(receive_data.shape)
#print(ir_receive_data.shape)


### CUT USEFUL PART
START_INDEX=0

max_index=np.argmax(ir_receive_data)

if  max_index - 0.05 * rate > 0 :
     START_INDEX = int (max_index - 0.05 * rate)
     
END_INDEX=int(START_INDEX+2*rate)

alligned_receive_data=ir_receive_data[START_INDEX:END_INDEX]

max_alligned_receive_data=np.max(alligned_receive_data)
min_from_beginning=np.min(np.abs(alligned_receive_data[0:10]))+0.00001

MAX_RATIO=max_alligned_receive_data/min_from_beginning

print(max_alligned_receive_data)
print(min_from_beginning)
print(MAX_RATIO)

MAX_RATIO_THRESHOLD=100

if MAX_RATIO > MAX_RATIO_THRESHOLD :
   wavfile.write(receive_fname+'.ir.wav',rate,np.array(alligned_receive_data).astype(np.float32))
else :
   print("Maximum diff for RIR "+receive_fname+" is "+str(MAX_RATIO)+" which is below "+str(MAX_RATIO_THRESHOLD)+", so not writing as wav file")






