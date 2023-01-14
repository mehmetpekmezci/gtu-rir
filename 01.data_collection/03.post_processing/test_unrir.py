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

transmit_fname=sys.argv[1]
rir_fname=sys.argv[2]


rate,transmit_data=wavfile.read(transmit_fname)
transmit_data_1d=transmit_data[:,0]+transmit_data[:,1]
transmit_data=transmit_data_1d[SILENCE_SECONDS_TRANSMITTER*rate:-SILENCE_SECONDS_TRANSMITTER*rate]

#play_sound(transmit_data,rate)
#plot_freqs(transmit_data,rate)
#plot_wave(transmit_data)


rate,rir_data=wavfile.read(rir_fname)

receive_data=sig.fftconvolve(transmit_data,rir_data,'full')
receive_data=receive_data/3000000
#play_sound(receive_data,rate)
#plot_freqs(receive_data,rate)
#plot_wave(receive_data)

writer = wave.open(rir_fname+'.received.wav', 'wb')
writer.setnchannels(1) # 1 channel, mono (one channel is active at an instance)
writer.setsampwidth(2) # 16bit
writer.setframerate(rate) # sample rate
#writer.writeframes(ir_receive_data.astype(np.int16).tostring())
writer.writeframes(receive_data.astype(np.int16).tostring())
print(rir_fname+'.received.wav')
writer.close()

