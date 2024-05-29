
import numpy as np
import sys
import os
import gc
import scipy.io.wavfile
from scipy import signal
from scipy import stats
import librosa.display
import librosa
import matplotlib.pyplot as plt


from acoustics.utils import _is_1d
from acoustics.signal import bandpass
from acoustics.bands import (_check_band_type, octave_low, octave_high, third_low, third_high)


np.set_printoptions(threshold=sys.maxsize)


from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

import torchaudio
import torch
import torch.nn.functional as TF


import scipy.fft
from scipy.spatial import distance

import traceback

class RIRData :
 def __init__(self,reportDir):
 
   self.script_dir=os.path.dirname(os.path.realpath(__file__))
   self.reduced_sampling_rate=16000
   self.data_length=4096
   self.sampleSignals=self.generateSampleSignals()
   self.reportDir=reportDir

 def plotWav(self,real_data,generated_data,MSE,SSIM,glitch_points,title,show=False,saveToPath=None):
     plt.clf()
     minValue=np.min(real_data)
     minValue2=np.min(generated_data)
     if minValue2 < minValue:
        minValue=minValue2

     plt.text(7, minValue+abs(minValue)/11, f"MSE={float(MSE):.4f}\nSSIM={float(SSIM):.4f}\nGLITCH={int(len(glitch_points))}", style='italic', bbox={'facecolor': 'gray', 'alpha': 0.5, 'pad': 10})
        
     plt.plot(real_data,color='#101010', label='real_data')
     plt.plot(generated_data,color='#909090', label='generated_data')

    
     x=glitch_points
     y=generated_data[x]
     plt.scatter(x,y,color="black",label="glitch_point")


     plt.title(title)
     plt.xlabel('Time')
     plt.ylabel('Amlpitude')
     plt.legend(loc = "upper right")
     
     if show :
        plt.show()
     if saveToPath is not None :
        plt.savefig(saveToPath)
        
 def generateSampleSignals(self):
     #real_data=librosa.resample(self.rir_data[i][-1], orig_sr=44100, target_sr=sr) 
     signal0=np.array([-0.5,0.6,-0.5,0.4,-0.4,0.3,-0.3,0.2,-0.1,0.05])
     signal1=np.array([-0.45,0.65,-0.45,0.35,-0.35,0.25,-0.25,0.75,-0.15,0])
     signal2=np.array([-0.9,0.9,-0.9,0.35,-0.35,0.25,-0.25,0.15,-0.15,0])
     return [signal0,signal1,signal2]
 
 def generateSampleReport(self):
     
     real_data=self.sampleSignals[0]
     real_data_tiled=np.tile(real_data, (2, 1)) ## duplicate 1d data to 2d
     real_data_tiled=np.reshape(real_data_tiled,(1,1,real_data_tiled.shape[0],real_data_tiled.shape[1]))
     real_data_tensor=torch.from_numpy(real_data_tiled)
     #glitched_data=self.sampleSignals[1]
     #non_glitch_but_same_mse_data=self.sampleSignals[2]
     #glitch_but_same_ssim_data=self.sampleSignals[3]



     for i in range(len(self.sampleSignals)-1):
        generated_data=self.sampleSignals[i+1]
        MSE=np.square(np.subtract(real_data,generated_data)).mean()

        generated_data_tiled=np.tile(generated_data, (2, 1)) ## duplicate 1d data to 2d
        generated_data_tiled=np.reshape(generated_data_tiled,(1,1,generated_data_tiled.shape[0],generated_data_tiled.shape[1]))
        generated_data_tensor=torch.from_numpy(generated_data_tiled)
        SSIM=ssim(generated_data_tensor,real_data_tensor,data_range=4.0,size_average=True).item()

        glitch_points=self.getGlitchPoints(generated_data,real_data)
        title=""
        file_name="no_title"
        if i == 0 :
           file_name="glitch"
        if i == 1 :
           file_name="no_glitch"


        self.plotWav(real_data,generated_data,MSE,SSIM,glitch_points,title,saveToPath=self.reportDir+"/"+file_name+".explanation.png")
         
 
 def getGlitchPoints(self,generated,real):
     INSENSITIVITY=3
     glitchThreshold=np.std(np.abs(real))*INSENSITIVITY
     #glitchThreshold=np.max(real)*1/2
     glitchPoints=[]
     #checkNextN=int(self.reduced_sampling_rate/50)
     #for i in range(len(generated)-checkNextN):
         #if  self.isBiggerThanNextN(i,checkNextN,glitchThreshold,generated,real):
     for i in range(len(generated)):
         if  abs(abs(generated[i])-abs(real[i]) )> glitchThreshold :
             glitchPoints.append(i)
     return glitchPoints







def main(reportDir):
  rirData=RIRData(reportDir)
  rirData.generateSampleReport()
  print("SCRIPT IS FINISHED")



if __name__ == '__main__':
 main(str(sys.argv[1]).strip())



