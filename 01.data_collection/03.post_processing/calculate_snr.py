import numpy as np
import scipy.io
import matplotlib
from matplotlib import pyplot as plt
import sys
import os
from scipy.io import wavfile
import librosa

#https://stackoverflow.com/questions/63177236/how-to-calculate-signal-to-noise-ratio-using-python


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def plot_wave(sound):
    plt.subplot(1,1,1)
    plt.plot(sound, 'b')
    plt.xlabel("Wave Plot")
    #plt.tight_layout()
    plt.show()

receive_fname=sys.argv[1]


receive_data,rate=librosa.load(receive_fname,sr=44100,mono=True)
receive_data=np.array(receive_data).astype(np.float32)
#plot_wave(receive_data)

## CHECK SHAPE 
if receive_data.shape[0] < 44100*10 :
   print("SNR_NEGATIVE:1")
   exit(0)

snr = signaltonoise(receive_data)

if snr<0 :
   print("SNR_NEGATIVE")
else :
   print("SNR_POSITIVE")

#if snr < 0 :
#   print("Negative SNR means noise>signal "+str(snr))
#   os.remove(receive_fname+".ir.wav")
#   os.remove(receive_fname+".alligned")
#   os.rename(receive_fname,receive_fname+".negative_snr")
