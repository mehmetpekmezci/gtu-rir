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
import librosa.display



sr=16000

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

#def pcm2float(sig, dtype='float32'):
#    """Convert PCM signal to floating point with a range from -1 to 1.
#    Use dtype='float32' for single precision.
#    Parameters
#    ----------
#    sig : array_like
#        Input array, must have integral type.
#    dtype : data type, optional
#        Desired (floating point) data type.
#    Returns
#    -------
#    numpy.ndarray
#        Normalized floating point data.
#    See Also
#    --------
#    float2pcm, dtype
#    """
#    sig = np.asarray(sig)
#    if sig.dtype.kind not in 'iu':
#        raise TypeError("'sig' must be an array of integers")
#    dtype = np.dtype(dtype)
#    if dtype.kind != 'f':
#        raise TypeError("'dtype' must be a floating point type")
#
#    i = np.iinfo(sig.dtype)
#    abs_max = 2 ** (i.bits - 1)
#    offset = i.min + abs_max
#    return (sig.astype(dtype) - offset) / abs_max

def saveSingleWavPlot(data1,saveToPath):
     plt.subplot(1,1,1)
     plt.plot(data1,color='#202020')
     plt.xlabel('Time')
     plt.ylabel('Amlpitude')
     plt.legend(loc = "upper right")
     plt.savefig(saveToPath)
     plt.close()

def getSpectrogram(data):
         sample_rate = 16000 ;  num_mfccs=4096
         #self.mfcc_transform_fn=torchaudio.transforms.MFCC(sample_rate=sample_rate,n_mfcc=num_mfccs,melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": num_mfccs, "center": False},).to("cuda")
         mfcc_transform_fn=torchaudio.transforms.MFCC(sample_rate=sample_rate,n_mfcc=num_mfccs,melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": num_mfccs, "center": False},)
         mfccs= mfcc_transform_fn( torch.Tensor(data) ).numpy()
         return mfccs

def saveRealAndGeneratedPlots(real_data,generated_data,saveToPath):
     #plt.clf()
     plt.rcParams['font.family'] = 'sans-serif'
     #generated_data=generated_data[int(abs(real_data.shape[0]-generated_data.shape[0])):] 
     generated_data=generated_data[:int(real_data.shape[0])] 

     generated_spectrogram=getSpectrogram(generated_data)
     generated_spectrogram=np.reshape(generated_spectrogram,(generated_spectrogram.shape[0],generated_spectrogram.shape[1],1))

     real_spectrogram=getSpectrogram(real_data)
     real_spectrogram=np.reshape(real_spectrogram,(real_spectrogram.shape[0],real_spectrogram.shape[1],1))

     MSE=np.square(np.subtract(real_data,generated_data)).mean()
     generated_spectrogram=np.reshape(generated_spectrogram,(1,1,generated_spectrogram.shape[0],generated_spectrogram.shape[1]))
     real_spectrogram=np.reshape(real_spectrogram,(1,1,real_spectrogram.shape[0],real_spectrogram.shape[1]))
     SSIM=ssim( torch.Tensor(generated_spectrogram), torch.Tensor(real_spectrogram), data_range=255, size_average=True).item()

     plt.subplot(1,1,1)
     minValue=np.min(real_data)
     minValue2=np.min(generated_data)
     if minValue2 < minValue:
        minValue=minValue2

     plt.plot(real_data,color='#101010', label='transmitted_song')
     plt.plot(generated_data,color='#909090', label='rir_convolved_with_transmitted_song')
     plt.text(3300, minValue+0.1, f"MSE={float(MSE):.4f}\nSSIM={float(SSIM):.4f}", style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

     #plt.title(title)
     plt.xlabel('Time')
     plt.ylabel('Amlpitude')
     plt.legend(loc = "upper right")

     plt.savefig(saveToPath)

     plt.close()


def plotSpectrogram(title,power_to_db,sr,show=False,saveToPath=None):

     ###plt.figure(figsize=(8, 7))
     fig, ax = plt.subplots()
     #fig.set_figwidth(self.SPECTROGRAM_DIM)
     #fig.set_figheight(self.SPECTROGRAM_DIM)
     #img=librosa.display.specshow(power_to_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma', hop_length=hop_length,ax=ax)
     #img=librosa.display.specshow(power_to_db, sr=sr, x_axis='time',  cmap='magma',  ax=ax)
     img=librosa.display.specshow(power_to_db, sr=sr, x_axis='time',  cmap='Greys',  ax=ax)

#ValueError: 'auto' is not a valid value for name; supported values are 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'

     fig.colorbar(img, ax=ax,label='')
     #plt.title('Mel-Spectrogram (dB)'+title, fontdict=dict(size=18))
     plt.title('MFCC '+title, fontdict=dict(size=18))
     plt.xlabel('', fontdict=dict(size=15))
     plt.ylabel('', fontdict=dict(size=15))
     #plt.axis('off')

     ###if show :
     ###   plt.show()
     ###if saveToPath is not None :

     ###plt.savefig(saveToPath, bbox_inches='tight', pad_inches=0)
     plt.savefig(saveToPath)

     plt.close()

def getSpectrogram(data,title=None,saveToPath=None):
         sample_rate = 16000 ;
         #num_mfccs=4096
         num_mfccs=40

         #do_also_librosa_for_comparison=False

         #if do_also_librosa_for_comparison:
         #  mfcc_librosa = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=num_mfccs)
         #  if saveToPath is not None :
         #   self.plotSpectrogram(title+"_librosa",mfcc_librosa,sample_rate,saveToPath=saveToPath+".librosa.png")

         #self.mfcc_transform_fn=torchaudio.transforms.MFCC(sample_rate=sample_rate,n_mfcc=num_mfccs,melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": num_mfccs, "center": False},).to("cuda")
         mfcc_transform_fn=torchaudio.transforms.MFCC(sample_rate=sample_rate,n_mfcc=num_mfccs,melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": num_mfccs, "center": False},)

         #data=np.reshape(data,(1,1,data.shape[0]))

         #print(data.shape)

         mfccs= mfcc_transform_fn( torch.Tensor(data) ).numpy()
         if saveToPath is not None :
            plotSpectrogram(title,mfccs,sample_rate,saveToPath=saveToPath+".spectrogram.png")
            saveSingleWavPlot(data,saveToPath+".signal.png")
         return mfccs

padsize=1000

SILENCE_SECONDS_TRANSMITTER=2

transmit_fname=sys.argv[1]
rir_fname=sys.argv[2]

transmit_data,sampling_rate=librosa.load(transmit_fname,sr=16000)
if len(transmit_data.shape) > 1 :
   transmit_data=transmit_data[:,0]+transmit_data[:,1]

rir_data,sampling_rate=librosa.load(rir_fname,sr=16000)


#print(rir_data.shape)
#print(rir_data)
#print(transmit_data.shape)
#print(np.max(rir_data))
#print(transmit_data)


reverbed_data=sig.fftconvolve(transmit_data,rir_data,'full')
log_reverbed_data_power=np.log(np.sum(np.abs(reverbed_data)))
print(log_reverbed_data_power)
reverbed_data=reverbed_data/((9/4)*log_reverbed_data_power)




max_point_index_transmit_data=np.argmax(transmit_data)
max_point_index_reverbed_data=np.argmax(reverbed_data)

diff=int(abs(max_point_index_transmit_data-max_point_index_reverbed_data))



print(f"diff={diff}")

if diff > 0 :
   new_reverbed_data=np.zeros(reverbed_data.shape[0]+diff)
   new_reverbed_data[:-diff]=reverbed_data
   reverbed_data=new_reverbed_data


#receive_data=receive_data/3000000
#play_sound(receive_data,rate)
#plot_freqs(receive_data,rate)
#plot_wave(receive_data)


#reverbed_data_sum=np.sum(np.abs(reverbed_data))
#transmit_data_sum=np.sum(np.abs(transmit_data))
#ratio=transmit_data_sum/reverbed_data_sum
#reverbed_data=reverbed_data*ratio


wavfile.write( rir_fname+'.reverbed.wav',16000,np.array(reverbed_data).astype(np.float32))

print("saveToPath="+rir_fname+".Example")
#getSpectrogram(reverbed_data,title="Reverbed Data",saveToPath=rir_fname+".Example")
getSpectrogram(rir_data,title=" RIR Data",saveToPath=rir_fname+".Example")

saveRealAndGeneratedPlots(transmit_data,reverbed_data,rir_fname+'.r.a.g.png')


#writer = wave.open(rir_fname+'.reverbed.wav', 'wb')
#writer.setnchannels(1) # 1 channel, mono (one channel is active at an instance)
#writer.setsampwidth(2) # 16bit
#writer.setframerate(rate) # sample rate
##writer.writeframes(ir_receive_data.astype(np.int16).tostring())
##writer.writeframes(receive_data.astype(np.int16).tostring())
#writer.writeframes(receive_data.tostring())
#print(rir_fname+'.reverbed.wav')
#writer.close()

