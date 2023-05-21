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


first_sound_file=sys.argv[1]

first_sound_file_data=np.zeros((4096,))
#print(np.max(first_sound_file_data))
#print(type(first_sound_file_data[0]))

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

