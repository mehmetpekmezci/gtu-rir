import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy import fft
from scipy.io import wavfile
import tempfile
import os
import pathlib
import sys
import pyaudio
import librosa
import wave


# METHOD is copied from https://pypi.org/project/syncstart/ ( Thanks to Roland Puntaier)

# python3 alligner.py receivedEssSignal-1.wav transmittedEssSignal.wav


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
    plt.plot(sound, 'r')
    plt.xlabel("Wave Plot")
    #plt.tight_layout()
    plt.show()



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

if receive_data_original.shape[0] >= transmit_data.shape[0]+SILENCE_SECONDS_TRANSMITTER*rate :
   receive_data=receive_data_original[SILENCE_SECONDS_TRANSMITTER*rate:SILENCE_SECONDS_TRANSMITTER*rate+transmit_data.shape[0]]
elif receive_data_original.shape[0]> transmit_data.shape[0]:
   receive_data=receive_data_original[(receive_data_original.shape[0]-transmit_data.shape[0]):]
else :
   receive_data=receive_data_original 
#receive_data=receive_data[:transmit_data.shape[0]]

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

#fs=rate

#s1pad=receive_data
#s2pad=transmit_data

#corr = fft.ifft(fft.fft(s1pad)*np.conj(fft.fft(s2pad)))
#ca = np.absolute(corr)
#xmax = np.argmax(ca)
#print(xmax)

##padsize=1000
##if xmax > padsize // 2:
##    offset = (padsize-xmax)/fs
##    #second signal (s2) to cut
##else:
##    offset = xmax/fs
##    #first signal (s1) to cut

##print(offset)

#alligned_receive_data=receive_data_original[SILENCE_SECONDS_TRANSMITTER*rate+xmax:]
alligned_receive_data=receive_data
print("Alligned receive_data.shape="+str(receive_data.shape))
#plot_freqs(receive_data,rate)
#plot_wave(receive_data)
#play_sound(transmit_data,rate)
#play_sound(receive_data,rate)


wavfile.write(receive_fname,rate,np.array(alligned_receive_data).astype(np.float32))




