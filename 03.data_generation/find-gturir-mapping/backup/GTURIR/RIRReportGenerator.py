#!/usr/bin/env python3


import gc
import glob
from RIRHeader import *
import scipy.io.wavfile
from scipy import signal
from scipy import stats
import librosa.display
import librosa
import matplotlib.pyplot as plt

#import tensorflow as tf

import shutil

from acoustics.utils import _is_1d
from acoustics.signal import bandpass
from acoustics.bands import (_check_band_type, octave_low, octave_high, third_low, third_high)

from PIL import Image,ImageDraw,ImageFont

from scipy.io import wavfile

np.set_printoptions(threshold=sys.maxsize)

class  RIRReportGenerator  :
 def __init__(self,rirData):
   self.logger  = rirData.logger
   self.rirData=rirData
   self.roomMeanMetricValues={}
   self.overallMeanMetricValues={}
   self.totalNumberOfRecords=0

   self.metrics=["MSE","SSIM","GLITCH_COUNT"] # metric."db.txt"
   
   for metric in self.metrics:
       self.roomMeanMetricValues[metric]={}
    
 def generateSummary(self):
     workDir=self.rirData.report_dir
     summaryDir=workDir+'/summary'
     if not os.path.exists(summaryDir):
            os.makedirs(summaryDir) 
     print ("\n\n\n")
     print ("-------------------------------------------------------------------------------")
     print ("SUMMARY TABLE :")

     for metric in self.metrics:
       self.roomMeanMetricValues[metric]={}
       self.overallMeanMetricValues[metric]=0
   
     r="ROOM"
     tableTitle=f"{r:>12}\t"
     for metric in self.metrics:
         if metric == "MSE" or metric == "SSIM":
            tableTitle+=f"{metric}\t\t"
         else :
            tableTitle+=f"{metric}\t"

     print (tableTitle)

     for roomPath in sorted(glob.glob(workDir+"/room-*/*")):
       room=os.path.basename(os.path.dirname(roomPath))
       #print(f"room={room}")
       numberOfRecords=0
          
       #Precision = TP/ TP+FP = Kesinlik
       #Recall    = TP/ TP+FN = Sensitivity=Duyarlik
       #Accuracy  = TP+TN/ TP+FP+TN+FN = Doğruluk
          
       for metric in self.metrics:
         with open(f"{roomPath}/{metric}.db.txt", encoding='utf8') as f:
           for line in f:
            if metric == self.metrics[0] : ## it is sufficient to count first metric's number of records.
               numberOfRecords=numberOfRecords+1
            lineSplit=line.strip().split("=")
            if room not in self.roomMeanMetricValues[metric]:
               self.roomMeanMetricValues[metric][room]=0
            self.roomMeanMetricValues[metric][room]+=float(lineSplit[1])
          
         if metric == self.metrics[0] : ## it is sufficient to count first metric's number of records.
            self.totalNumberOfRecords=self.totalNumberOfRecords+numberOfRecords
         self.overallMeanMetricValues[metric]+=self.roomMeanMetricValues[metric][room]
         self.roomMeanMetricValues[metric][room]=self.roomMeanMetricValues[metric][room]/numberOfRecords


       roomDataLine=f"{room[:12]:>12}\t"
       for metric in self.metrics:
         roomDataLine+=f"{float(self.roomMeanMetricValues[metric][room]):.4f}\t\t"
       print (roomDataLine)

     overAllMeanMetricValues=""
     for metric in self.metrics:
         overAllMeanMetricValues+=f"MEAN_{metric}={float(self.overallMeanMetricValues[metric]/self.totalNumberOfRecords):.4f}\n"

     with open(f"{summaryDir}/summary.db.txt",'w', encoding='utf8') as f:
         f.write(overAllMeanMetricValues)


