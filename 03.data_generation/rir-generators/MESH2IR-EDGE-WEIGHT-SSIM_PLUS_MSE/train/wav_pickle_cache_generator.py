# from PIL import Image
import soundfile as sf
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import pandas as pd
from scipy import signal

import torch
import torch_geometric
from torch_geometric.io import read_ply
import librosa

import io
import sys
import time
import glob



def get_RIR(full_RIR_path):
        # wav,fs = sf.read(full_RIR_path) 
        wav,fs = librosa.load(full_RIR_path)
 
        # wav_resample = librosa.resample(wav,16000,fs)
        wav_resample = librosa.resample(wav,orig_sr=fs,target_sr=16000)

        length = wav_resample.size

        crop_length = 3968 #int(16384)
        if(length<crop_length):
            zeros = np.zeros(crop_length-length)
            std_value = np.std(wav_resample) * 10
            std_array = np.repeat(std_value,128)
            wav_resample_new = np.concatenate([wav_resample,zeros])/std_value
            RIR_original = np.concatenate([wav_resample_new,std_array])
        else:
            wav_resample_new = wav_resample[0:crop_length]
            std_value = np.std(wav_resample_new)  * 10
            std_array = np.repeat(std_value,128)
            wav_resample_new =wav_resample_new/std_value
            RIR_original = np.concatenate([wav_resample_new,std_array])

        RIR = RIR_original

        RIR = np.array([RIR]).astype('float32')

        return RIR



def load_pickle(pickle_file):
        with open(pickle_file, 'rb') as f:
            file_content = pickle.load(f)
        return file_content

def write_pickle(pickle_file,file_content):
        with open(pickle_file, 'wb') as f:
            pickle.dump(file_content,f, protocol=2)


dataset_path = str(sys.argv[1]).strip()

total_dir_count=len(glob.glob(dataset_path+'/*-*-*-*-*'))

count=0

for i in glob.glob(dataset_path+'/*-*-*-*-*'):
  count+=1
  if count%10 == 0 :
     print(f'{count}/{total_dir_count}')
  dirname=os.path.basename(i)
  if glob.glob(dataset_path+'/cache/'+dirname+'/.cacheGenerated'):
      #print(dataset_path+'/cache/'+dirname+' is already generated')
      a=0
  else :
     #print('Generating : '+dataset_path+'/cache/'+dirname)
     if not glob.glob(dataset_path+'/cache/'+dirname):
        os.mkdir(dataset_path+'/cache/'+dirname)

     for j in glob.glob(i+'/hybrid/*.wav'):
      wavname=os.path.basename(j)
      if not glob.glob(dataset_path+'/cache/'+dirname+'/'+wavname+'.pickle'):
       r1=get_RIR(j)
       #r1=[]
       write_pickle(dataset_path+'/cache/'+dirname+'/'+wavname+'.pickle',r1)

     with open(dataset_path+'/cache/'+dirname+'/.cacheGenerated', "w") as myfile:
        myfile.write('')





