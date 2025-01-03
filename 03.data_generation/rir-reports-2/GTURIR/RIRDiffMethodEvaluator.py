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



import shutil

from acoustics.utils import _is_1d
from acoustics.signal import bandpass
from acoustics.bands import (_check_band_type, octave_low, octave_high, third_low, third_high)

from PIL import Image,ImageDraw,ImageFont

from scipy.io import wavfile

import random

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

import torchaudio
import torch



np.set_printoptions(threshold=sys.maxsize)

class  RIRDiffMethodEvaluator  :
 def __init__(self,rirData):

  
   
   self.logger  = rirData.logger
   self.rirData=rirData
   
   self.MSE_THRESHOLD=0.01 # we will select the records having lower MSE than this value
   self.LMDS_THRESHOLD=0.01 # we will select the records having lower MSE than this value
   self.SSIM_THRESHOLD=0.5 # we will select the records having higher SSIM than this value
   
   self.roomIdFieldNumber=int(self.rirData.rir_data_field_numbers['roomId'])
   self.roomWidthFieldNumber=int(self.rirData.rir_data_field_numbers['roomWidth'])
   self.roomHeightFieldNumber=int(self.rirData.rir_data_field_numbers['roomHeight'])
   self.roomDepthFieldNumber=int(self.rirData.rir_data_field_numbers['roomDepth'])
   self.rirDataFieldNumber=int(self.rirData.rir_data_field_numbers['rirData'])
   
   self.microphoneStandInitialCoordinateXFieldNumber=int(self.rirData.rir_data_field_numbers['microphoneStandInitialCoordinateX'])
   self.microphoneStandInitialCoordinateYFieldNumber=int(self.rirData.rir_data_field_numbers['microphoneStandInitialCoordinateY'])
   self.mic_RelativeCoordinateXFieldNumber=int(self.rirData.rir_data_field_numbers['mic_RelativeCoordinateX'])
   self.mic_RelativeCoordinateYFieldNumber=int(self.rirData.rir_data_field_numbers['mic_RelativeCoordinateY'])
   self.mic_RelativeCoordinateZFieldNumber=int(self.rirData.rir_data_field_numbers['mic_RelativeCoordinateZ'])
   
   self.speakerStandInitialCoordinateXFieldNumber=int(self.rirData.rir_data_field_numbers['speakerStandInitialCoordinateX'])
   self.speakerStandInitialCoordinateYFieldNumber=int(self.rirData.rir_data_field_numbers['speakerStandInitialCoordinateY'])
   self.speakerRelativeCoordinateXFieldNumber=int(self.rirData.rir_data_field_numbers['speakerRelativeCoordinateX'])
   self.speakerRelativeCoordinateYFieldNumber=int(self.rirData.rir_data_field_numbers['speakerRelativeCoordinateY'])
   self.speakerRelativeCoordinateZFieldNumber=int(self.rirData.rir_data_field_numbers['speakerRelativeCoordinateZ'])
   
   self.roomIds=[]
   self.volumes={}
   self.roomDimensions={}
   self.speakerCoordinates=[]
   self.microphoneCoordinates=[]
   
   CENT=100
   
   for i in range(len(self.rirData.rir_data)):
         roomId=str(self.rirData.rir_data[i][self.roomIdFieldNumber])
         if roomId not in self.roomIds:
            self.roomIds.append(roomId)
            roomWidth=float(self.rirData.rir_data[i][self.roomWidthFieldNumber])/100#cm to m
            roomHeight=float(self.rirData.rir_data[i][self.roomHeightFieldNumber])/100
            roomDepth=float(self.rirData.rir_data[i][self.roomDepthFieldNumber])/100
            self.volumes[roomId]=roomWidth*roomHeight*roomDepth
            self.roomDimensions[roomId]=[roomDepth,roomWidth,roomHeight]


   for dataline in self.rirData.rir_data:
       microphoneCoordinatesX=round(float(dataline[self.microphoneStandInitialCoordinateXFieldNumber])/CENT +float(dataline[self.mic_RelativeCoordinateXFieldNumber])/CENT,1) # CM to M 
       microphoneCoordinatesY=round(float(dataline[self.microphoneStandInitialCoordinateYFieldNumber])/CENT +float(dataline[self.mic_RelativeCoordinateYFieldNumber])/CENT,1) # CM to M
       microphoneCoordinatesZ=round(float(dataline[self.mic_RelativeCoordinateZFieldNumber])/CENT,1)
       if [microphoneCoordinatesX,microphoneCoordinatesY,microphoneCoordinatesZ] not in self.microphoneCoordinates:
          self.microphoneCoordinates.append([microphoneCoordinatesX,microphoneCoordinatesY,microphoneCoordinatesZ])
       
       speakerCoordinatesX=round(float(dataline[self.speakerStandInitialCoordinateXFieldNumber])/CENT +float(dataline[self.speakerRelativeCoordinateXFieldNumber])/CENT,1) # CM to M
       speakerCoordinatesY=round(float(dataline[self.speakerStandInitialCoordinateYFieldNumber])/CENT +float(dataline[self.speakerRelativeCoordinateYFieldNumber])/CENT,1) # CM to M
       speakerCoordinatesZ=round(float(dataline[self.speakerRelativeCoordinateZFieldNumber])/CENT,1)
       if [speakerCoordinatesX,speakerCoordinatesY,speakerCoordinatesZ] not in self.speakerCoordinates:
          self.speakerCoordinates.append([speakerCoordinatesX,speakerCoordinatesY,speakerCoordinatesZ])



            

 def ssim(self,data1,data2):
 
 
 
     mfcc1=self.rirData.getSpectrogram(data1)
     mfcc1 = np.reshape(mfcc1,(1,1,mfcc1.shape[0],mfcc1.shape[1]))
     
     mfcc2=self.rirData.getSpectrogram(data2)
     mfcc2 = np.reshape(mfcc2,(1,1,mfcc2.shape[0],mfcc2.shape[1]))
 
 
     return ssim( torch.Tensor(mfcc1), torch.Tensor(mfcc2), data_range=255, size_average=True).item()
     
     
     
     
     
     
     
     
 
 def mse(self,data1,data2):
     return np.square(np.subtract(data1,data2)).mean()      
 
 def compareRandomPointsInARoom(self,roomId,dataPointsInaRoom,K,show=True):
         if len(dataPointsInaRoom) == 0 :
            return  [0,0,0,0,0,0,0,0,0,0,0,0,0]
         firstPoints=random.sample(dataPointsInaRoom,k=K)
         secondPoints=random.sample(dataPointsInaRoom,k=K)
         totalMse=0;totalSsim=0;totalLmds=0
         maxMse=0;meanMse=0;minMse=-100;mse_001=0
         maxSsim=0;meanSsim=0;minSsim=-100;ssim_05=0
         maxLmds=0;meanLmds=0;minLmds=-100;lmds_05=0
         
         for i in range(len(firstPoints)):
             firstRirData=firstPoints[i]
             secondRirData=secondPoints[i]
             
             mse=self.mse(firstRirData,secondRirData)              
             ssim=self.ssim(firstRirData,secondRirData)
             lmds=self.rirData.localMaxDiffSum(firstRirData,secondRirData)
             
             if maxMse < mse :
                maxMse=mse
             elif minMse == -100 or minMse > mse :
                minMse=mse
             totalMse=totalMse+mse
             
             if mse < self.MSE_THRESHOLD:
                mse_001=mse_001+1
                       
             if maxSsim < ssim :
                maxSsim=ssim
             elif minSsim == -100 or minSsim > ssim :
                minSsim=ssim
             totalSsim=totalSsim+ssim
             
             if ssim > self.SSIM_THRESHOLD:
                ssim_05=ssim_05+1
                       
             if maxLmds < lmds :
                maxLmds=lmds
             elif minLmds == -100 or minLmds > lmds :
                minLmds=lmds
             totalLmds=totalLmds+lmds
             
             if lmds < self.LMDS_THRESHOLD:
                lmds_05=lmds_05+1
                       
         meanMse=totalMse/len(firstPoints)
         meanSsim=totalSsim/len(firstPoints)
         meanLmds=totalLmds/len(firstPoints)
         
         if roomId in self.volumes :
            VOLUME=self.volumes[roomId]
         else:
            VOLUME=0
         if minSsim == -100:
            minSsim = 0 
         if minMse == -100:
            minMse = 0 
         if minLmds == -100:
            minLmds = 0 
              
         if show :   
            print (f"{roomId[:20]:>20}\t\t{VOLUME:.4f}\t\t{float(maxMse):.4f}\t\t{float(meanMse):.4f}\t\t{float(minMse):.4f}\t\t{int(mse_001)}\t\t{float(maxSsim):.4f}\t\t{float(meanSsim):.4f}\t\t{float(minSsim):.4f}\t\t{int(ssim_05)}\t\t{float(maxLmds):.4f}\t\t{float(meanLmds):.4f}\t\t{float(minLmds):.4f}\t\t{int(lmds_05)}", flush=True)
       
         return [VOLUME,maxMse,meanMse,minMse,mse_001,maxSsim,meanSsim,minSsim,ssim_05,maxLmds,meanLmds,minLmds,lmds_05]

 def compareRandomDataPointPairsInRealData(self):
     print ("\n\n\n")
     print (f"1.0 COMPARE - RANDOM 1000  DATA POINT PAIRS - REAL DATA :")
     print ("-------------------------------------------------------------------")
     print ("SUMMARY TABLE :")
     print (f"{'ROOM':>20}\t\tVOLUME\t\tMAX_MSE\t\tMEAN_MSE\tMIN_MSE\t\tMSE<0.01\tMAX_SSIM\tMEAN_SSIM\tMIN_SSIM\tSSIM>0.5\tMAX_LMDS\tMEAN_LMDS\tMIN_LMDS\tLMDS<0.01", flush=True)
     workDir=self.rirData.fast_rir_dir+"/code_new/Generated_RIRs_report/"
     
     dataPointsInaRoom=[]
     for dataPoint in self.rirData.rir_data:
         dataPointsInaRoom.append(dataPoint[self.rirDataFieldNumber])
     self.compareRandomPointsInARoom("ANY ROOM",dataPointsInaRoom,1000)    
             

 def compareRandomDataPointPairsInGeneratedData(self):
     print ("\n\n\n")
     print (f"1.1 COMPARE - RANDOM 1000  DATA POINT PAIRS - GENERATED DATA :")
     print ("------------------------------------------------------------------------")
     print ("SUMMARY TABLE :")
     print (f"{'ROOM':>20}\t\tVOLUME\t\tMAX_MSE\t\tMEAN_MSE\tMIN_MSE\t\tMSE<0.01\tMAX_SSIM\tMEAN_SSIM\tMIN_SSIM\tSSIM>0.5\tMAX_LMDS\tMEAN_LMDS\tMIN_LMDS\tLMDS<0.01", flush=True)

     dataPointsInaRoom=[]
     for i in range(len(self.rirData.fastRirInputData)):
                labels_embeddings_batch=(self.rirData.fastRirInputData[i]+1)*5
                record_name = f"RIR-DEPTH-{labels_embeddings_batch[6]}-WIDTH-{labels_embeddings_batch[7]}-HEIGHT-{labels_embeddings_batch[8]}-RT60-{labels_embeddings_batch[9]}-MX-{labels_embeddings_batch[0]}-MY-{labels_embeddings_batch[1]}-MZ-{labels_embeddings_batch[2]}-SX-{labels_embeddings_batch[3]}-SY-{labels_embeddings_batch[4]}-SZ-{labels_embeddings_batch[5]}-{i}"
                wave_name=record_name+".wav"
                generated_data,rate=librosa.load(self.rirData.fast_rir_dir+"/code_new/Generated_RIRs/"+wave_name,sr=16000,mono=True)
                dataPointsInaRoom.append(generated_data)
            
     self.compareRandomPointsInARoom("ANY ROOM",dataPointsInaRoom,1000)


             
 def compareRandomDataPointPairsInRealDataSameRoom(self):
     print ("\n\n\n")
     print (f"2.0 COMPARE - RANDOM 500  DATA POINT PAIRS - REAL DATA - SAME ROOM:")
     print ("-------------------------------------------------------------------------------")
     print ("SUMMARY TABLE :")
     print (f"{'ROOM':>20}\t\tVOLUME\t\tMAX_MSE\t\tMEAN_MSE\tMIN_MSE\t\tMSE<0.01\tMAX_SSIM\tMEAN_SSIM\tMIN_SSIM\tSSIM>0.5\tMAX_LMDS\tMEAN_LMDS\tMIN_LMDS\tLMDS<0.01", flush=True)
     workDir=self.rirData.fast_rir_dir+"/code_new/Generated_RIRs_report/"
     
     for roomId in self.roomIds:
         dataPointsInaRoom=[]
         for dataPoint in self.rirData.rir_data:
             if roomId == str(dataPoint[self.roomIdFieldNumber]) :
                dataPointsInaRoom.append(dataPoint[self.rirDataFieldNumber])
         self.compareRandomPointsInARoom(roomId,dataPointsInaRoom,500)    
             

 def compareRandomDataPointPairsInGeneratedDataSameRoom(self):
     print ("\n\n\n")
     print (f"2.1 COMPARE - RANDOM 500  DATA POINT PAIRS - GENERATED DATA - SAME ROOM :")
     print ("-------------------------------------------------------------------------------------")
     print ("SUMMARY TABLE :")
     print (f"{'ROOM':>20}\t\tVOLUME\t\tMAX_MSE\t\tMEAN_MSE\tMIN_MSE\t\tMSE<0.01\tMAX_SSIM\tMEAN_SSIM\tMIN_SSIM\tSSIM>0.5\tMAX_LMDS\tMEAN_LMDS\tMIN_LMDS\tLMDS<0.01", flush=True)

     for roomId in self.roomIds:
         dataPointsInaRoom=[]
         for i in range(len(self.rirData.fastRirInputData)):
             generatedRecordRoomId=str(self.rirData.rir_data[i][self.roomIdFieldNumber])
             if roomId == generatedRecordRoomId:
                labels_embeddings_batch=(self.rirData.fastRirInputData[i]+1)*5
                record_name = f"RIR-DEPTH-{labels_embeddings_batch[6]}-WIDTH-{labels_embeddings_batch[7]}-HEIGHT-{labels_embeddings_batch[8]}-RT60-{labels_embeddings_batch[9]}-MX-{labels_embeddings_batch[0]}-MY-{labels_embeddings_batch[1]}-MZ-{labels_embeddings_batch[2]}-SX-{labels_embeddings_batch[3]}-SY-{labels_embeddings_batch[4]}-SZ-{labels_embeddings_batch[5]}-{i}"
                wave_name=record_name+".wav"
                generated_data,rate=librosa.load(self.rirData.fast_rir_dir+"/code_new/Generated_RIRs/"+wave_name,sr=16000,mono=True)
                dataPointsInaRoom.append(generated_data)
            
         self.compareRandomPointsInARoom(roomId,dataPointsInaRoom,500)
     
     


    
 def compareDataPointPairsThatHasTheSameSpeakerPointInRealData(self):
     print ("\n\n\n")
     print (f"3.0 COMPARE - RANDOM 100 DATA POINT PAIRS - REAL DATA - ANY ROOM - SIMILAR SPEAKER COORDINATES:")
     print ("---------------------------------------------------------------------------------------------------")
     print ("SUMMARY TABLE :")
     print (f"{'ROOM':>20}\t\tVOLUME\t\tMAX_MSE\t\tMEAN_MSE\tMIN_MSE\t\tMSE<0.01\tMAX_SSIM\tMEAN_SSIM\tMIN_SSIM\tSSIM>0.5\tMAX_LMDS\tMEAN_LMDS\tMIN_LMDS\tLMDS<0.01", flush=True)
     CENT=100
     VOLUME=0;maxMse=0;meanMse=0;minMse=0;mse_001=0;maxSsim=0;meanSsim=0;minSsim=0;ssim_05=0;maxLmds=0;meanLmds=0;minLmds=0;lmds_05=0
     totalDataPoints=0
     
     for speakerXYZ in self.speakerCoordinates:
         dataPointsInaRoom=[]
         for dataline in self.rirData.rir_data:
             speakerCoordinatesX=round(float(dataline[self.speakerStandInitialCoordinateXFieldNumber])/CENT +float(dataline[self.speakerRelativeCoordinateXFieldNumber])/CENT,1) # CM to M
             speakerCoordinatesY=round(float(dataline[self.speakerStandInitialCoordinateYFieldNumber])/CENT +float(dataline[self.speakerRelativeCoordinateYFieldNumber])/CENT,1) # CM to M
             speakerCoordinatesZ=round(float(dataline[self.speakerRelativeCoordinateZFieldNumber])/CENT,1)
             if [speakerCoordinatesX,speakerCoordinatesY,speakerCoordinatesZ] == speakerXYZ :
                dataPointsInaRoom.append(dataline[self.rirDataFieldNumber])
                       
         [new_VOLUME,new_maxMse,new_meanMse,new_minMse,new_mse_001,new_maxSsim,new_meanSsim,new_minSsim,new_ssim_05,new_maxLmds,new_meanLmds,new_minLmds,new_lmds_05]=self.compareRandomPointsInARoom("",dataPointsInaRoom,min(100,len(dataPointsInaRoom)),show=False)    
         
         VOLUME=VOLUME+new_VOLUME; 
         
         if maxMse < new_maxMse : maxMse= new_maxMse
         if minMse > new_minMse : minMse= new_minMse
         meanMse=meanMse+new_meanMse
         mse_001=mse_001+new_mse_001
         
         if maxSsim < new_maxSsim : maxSsim= new_maxSsim
         if minSsim > new_minSsim : minSsim= new_minSsim
         meanSsim=meanSsim+new_meanSsim
         ssim_05=ssim_05+new_ssim_05
         
         if maxLmds < new_maxLmds : maxLmds= new_maxLmds
         if minLmds > new_minLmds : minLmds= new_minLmds
         meanLmds=meanLmds+new_meanLmds
         lmds_05=lmds_05+new_lmds_05
         
         totalDataPoints=totalDataPoints+len(dataPointsInaRoom)
     
     print (f"{str(totalDataPoints):>20}\t\t{VOLUME:.2f}\t\t{float(maxMse):.4f}\t\t{float(meanMse)/len(self.speakerCoordinates):.4f}\t\t{float(minMse):.4f}\t\t{int(mse_001)}\t\t{float(maxSsim):.4f}\t\t{float(meanSsim)/len(self.speakerCoordinates):.4f}\t\t{float(minSsim):.4f}\t\t{int(ssim_05)}\t\t{float(maxLmds):.4f}\t\t{float(meanLmds)/len(self.speakerCoordinates):.4f}\t\t{float(minLmds):.4f}\t\t{int(lmds_05)}", flush=True)


 def compareDataPointPairsThatHasTheSameMicrophonePointInRealData(self):
     print ("\n\n\n")
     print (f"3.1 COMPARE - RANDOM 100 DATA POINT PAIRS - REAL DATA - ANY ROOM - SIMILAR MICROPHONE COORDINATES:")
     print ("---------------------------------------------------------------------------------------------------")
     print ("SUMMARY TABLE :")
     print (f"{'ROOM':>20}\t\tVOLUME\t\tMAX_MSE\t\tMEAN_MSE\tMIN_MSE\t\tMSE<0.01\tMAX_SSIM\tMEAN_SSIM\tMIN_SSIM\tSSIM>0.5\tMAX_LMDS\tMEAN_LMDS\tMIN_LMDS\tLMDS<0.01", flush=True)
     CENT=100
     VOLUME=0;maxMse=0;meanMse=0;minMse=0;mse_001=0;maxSsim=0;meanSsim=0;minSsim=0;ssim_05=0;maxLmds=0;meanLmds=0;minLmds=0;lmds_05=0
     totalDataPoints=0
     for microphoneXYZ in self.microphoneCoordinates:
         dataPointsInaRoom=[]
         for dataline in self.rirData.rir_data:
             microphoneCoordinatesX=round(float(dataline[self.microphoneStandInitialCoordinateXFieldNumber])/CENT +float(dataline[self.mic_RelativeCoordinateXFieldNumber])/CENT,1) # CM to M 
             microphoneCoordinatesY=round(float(dataline[self.microphoneStandInitialCoordinateYFieldNumber])/CENT +float(dataline[self.mic_RelativeCoordinateYFieldNumber])/CENT,1) # CM to M
             microphoneCoordinatesZ=round(float(dataline[self.mic_RelativeCoordinateZFieldNumber])/CENT,1)
             if [microphoneCoordinatesX,microphoneCoordinatesY,microphoneCoordinatesZ] == microphoneXYZ :
                dataPointsInaRoom.append(dataline[self.rirDataFieldNumber])

         [new_VOLUME,new_maxMse,new_meanMse,new_minMse,new_mse_001,new_maxSsim,new_meanSsim,new_minSsim,new_ssim_05,new_maxLmds,new_meanLmds,new_minLmds,new_lmds_05]=self.compareRandomPointsInARoom("",dataPointsInaRoom,min(100,len(dataPointsInaRoom)),show=False)    
         
         VOLUME=VOLUME+new_VOLUME; 
         
         if maxMse < new_maxMse : maxMse= new_maxMse
         if minMse > new_minMse : minMse= new_minMse
         meanMse=meanMse+new_meanMse
         mse_001=mse_001+new_mse_001
         
         if maxSsim < new_maxSsim : maxSsim= new_maxSsim
         if minSsim > new_minSsim : minSsim= new_minSsim
         meanSsim=meanSsim+new_meanSsim
         ssim_05=ssim_05+new_ssim_05
         
         if maxLmds < new_maxLmds : maxLmds= new_maxLmds
         if minLmds > new_minLmds : minLmds= new_minLmds
         meanLmds=meanLmds+new_meanLmds
         lmds_05=lmds_05+new_lmds_05
         
         totalDataPoints=totalDataPoints+len(dataPointsInaRoom)
     
     print (f"{str(totalDataPoints):>20}\t\t{VOLUME:.2f}\t\t{float(maxMse):.4f}\t\t{float(meanMse)/len(self.speakerCoordinates):.4f}\t\t{float(minMse):.4f}\t\t{int(mse_001)}\t\t{float(maxSsim):.4f}\t\t{float(meanSsim)/len(self.speakerCoordinates):.4f}\t\t{float(minSsim):.4f}\t\t{int(ssim_05)}\t\t{float(maxLmds):.4f}\t\t{float(meanLmds)/len(self.speakerCoordinates):.4f}\t\t{float(minLmds):.4f}\t\t{int(lmds_05)}", flush=True)

 
 def compareDataPointPairsThatHasTheSameSpeakerPointInGeneratedData(self):
     print ("\n\n\n")
     print (f"4.0 COMPARE - RANDOM 100 DATA POINT PAIRS - GENERATED DATA - ANY ROOM - SIMILAR SPEAKER COORDINATES:")
     print ("---------------------------------------------------------------------------------------------------")
     print ("SUMMARY TABLE :")
     print (f"{'TOTAL DATA POINTS':>20}\t\tVOLUME\t\tMAX_MSE\t\tMEAN_MSE\tMIN_MSE\t\tMSE<0.01\tMAX_SSIM\tMEAN_SSIM\tMIN_SSIM\tSSIM>0.5\tMAX_LMDS\tMEAN_LMDS\tMIN_LMDS\tLMDS<0.01", flush=True)
     VOLUME=0;maxMse=0;meanMse=0;minMse=0;mse_001=0;maxSsim=0;meanSsim=0;minSsim=0;ssim_05=0;maxLmds=0;meanLmds=0;minLmds=0;lmds_05=0
     totalDataPoints=0
     
     for speakerXYZ in self.speakerCoordinates:
         dataPointsInaRoom=[]
         for i in range(len(self.rirData.fastRirInputData)):
             labels_embeddings_batch=(self.rirData.fastRirInputData[i]+1)*5
             speakerCoordinatesX=round(labels_embeddings_batch[3],1)
             speakerCoordinatesY=round(labels_embeddings_batch[4],1)
             speakerCoordinatesZ=round(labels_embeddings_batch[5],1)
             
             
             if speakerCoordinatesX == speakerXYZ[0] and speakerCoordinatesY == speakerXYZ[1] and speakerCoordinatesZ == speakerXYZ[2] :
                record_name = f"RIR-DEPTH-{labels_embeddings_batch[6]}-WIDTH-{labels_embeddings_batch[7]}-HEIGHT-{labels_embeddings_batch[8]}-RT60-{labels_embeddings_batch[9]}-MX-{labels_embeddings_batch[0]}-MY-{labels_embeddings_batch[1]}-MZ-{labels_embeddings_batch[2]}-SX-{labels_embeddings_batch[3]}-SY-{labels_embeddings_batch[4]}-SZ-{labels_embeddings_batch[5]}-{i}"
                wave_name=record_name+".wav"
                generated_data,rate=librosa.load(self.rirData.fast_rir_dir+"/code_new/Generated_RIRs/"+wave_name,sr=16000,mono=True)
                dataPointsInaRoom.append(generated_data)
         
         [new_VOLUME,new_maxMse,new_meanMse,new_minMse,new_mse_001,new_maxSsim,new_meanSsim,new_minSsim,new_ssim_05,new_maxLmds,new_meanLmds,new_minLmds,new_lmds_05]=self.compareRandomPointsInARoom("",dataPointsInaRoom,min(100,len(dataPointsInaRoom)),show=False)    
         
         VOLUME=VOLUME+new_VOLUME; 
         
         if maxMse < new_maxMse : maxMse= new_maxMse
         if minMse > new_minMse : minMse= new_minMse
         meanMse=meanMse+new_meanMse
         mse_001=mse_001+new_mse_001
         
         if maxSsim < new_maxSsim : maxSsim= new_maxSsim
         if minSsim > new_minSsim : minSsim= new_minSsim
         meanSsim=meanSsim+new_meanSsim
         ssim_05=ssim_05+new_ssim_05
         
         if maxLmds < new_maxLmds : maxLmds= new_maxLmds
         if minLmds > new_minLmds : minLmds= new_minLmds
         meanLmds=meanLmds+new_meanLmds
         lmds_05=lmds_05+new_lmds_05
         
         totalDataPoints=totalDataPoints+len(dataPointsInaRoom)
     
     print (f"{str(totalDataPoints):>20}\t\t{VOLUME:.2f}\t\t{float(maxMse):.4f}\t\t{float(meanMse)/len(self.speakerCoordinates):.4f}\t\t{float(minMse):.4f}\t\t{int(mse_001)}\t\t{float(maxSsim):.4f}\t\t{float(meanSsim)/len(self.speakerCoordinates):.4f}\t\t{float(minSsim):.4f}\t\t{int(ssim_05)}\t\t{float(maxLmds):.4f}\t\t{float(meanLmds)/len(self.speakerCoordinates):.4f}\t\t{float(minLmds):.4f}\t\t{int(lmds_05)}", flush=True)
         

 def compareDataPointPairsThatHasTheSameMicrophonePointInGeneratedData(self):
     print ("\n\n\n")
     print (f"4.1 COMPARE - RANDOM 100 DATA POINT PAIRS - GENERATED DATA - ANY ROOM - SIMILAR MICROPHONE COORDINATES:")
     print ("---------------------------------------------------------------------------------------------------")
     print ("SUMMARY TABLE :")
     print (f"{'TOTAL DATA POINTS':>20}\t\tVOLUME\t\tMAX_MSE\t\tMEAN_MSE\tMIN_MSE\t\tMSE<0.01\tMAX_SSIM\tMEAN_SSIM\tMIN_SSIM\tSSIM>0.5\tMAX_LMDS\tMEAN_LMDS\tMIN_LMDS\tLMDS<0.01", flush=True)
     VOLUME=0;maxMse=0;meanMse=0;minMse=0;mse_001=0;maxSsim=0;meanSsim=0;minSsim=0;ssim_05=0;maxLmds=0;meanLmds=0;minLmds=0;lmds_05=0
     totalDataPoints=0
     
     for microphoneXYZ in self.microphoneCoordinates:
         dataPointsInaRoom=[]
         for i in range(len(self.rirData.fastRirInputData)):
             labels_embeddings_batch=(self.rirData.fastRirInputData[i]+1)*5
             micCoordinatesX=round(labels_embeddings_batch[0],1)
             micCoordinatesY=round(labels_embeddings_batch[1],1)
             micCoordinatesZ=round(labels_embeddings_batch[2],1)
             
             
             if micCoordinatesX == microphoneXYZ[0] and micCoordinatesY == microphoneXYZ[1] and micCoordinatesZ == microphoneXYZ[2] :
                record_name = f"RIR-DEPTH-{labels_embeddings_batch[6]}-WIDTH-{labels_embeddings_batch[7]}-HEIGHT-{labels_embeddings_batch[8]}-RT60-{labels_embeddings_batch[9]}-MX-{labels_embeddings_batch[0]}-MY-{labels_embeddings_batch[1]}-MZ-{labels_embeddings_batch[2]}-SX-{labels_embeddings_batch[3]}-SY-{labels_embeddings_batch[4]}-SZ-{labels_embeddings_batch[5]}-{i}"
                wave_name=record_name+".wav"
                generated_data,rate=librosa.load(self.rirData.fast_rir_dir+"/code_new/Generated_RIRs/"+wave_name,sr=16000,mono=True)
                dataPointsInaRoom.append(generated_data)
         

         [new_VOLUME,new_maxMse,new_meanMse,new_minMse,new_mse_001,new_maxSsim,new_meanSsim,new_minSsim,new_ssim_05,new_maxLmds,new_meanLmds,new_minLmds,new_lmds_05]=self.compareRandomPointsInARoom("",dataPointsInaRoom,min(100,len(dataPointsInaRoom)),show=False)    
         
         VOLUME=VOLUME+new_VOLUME; 
         
         if maxMse < new_maxMse : maxMse= new_maxMse
         if minMse > new_minMse : minMse= new_minMse
         meanMse=meanMse+new_meanMse
         mse_001=mse_001+new_mse_001
         
         if maxSsim < new_maxSsim : maxSsim= new_maxSsim
         if minSsim > new_minSsim : minSsim= new_minSsim
         meanSsim=meanSsim+new_meanSsim
         ssim_05=ssim_05+new_ssim_05

         if maxLmds < new_maxLmds : maxLmds= new_maxLmds
         if minLmds > new_minLmds : minLmds= new_minLmds
         meanLmds=meanLmds+new_meanLmds
         lmds_05=lmds_05+new_lmds_05
                  
         totalDataPoints=totalDataPoints+len(dataPointsInaRoom)
     
     print (f"{str(totalDataPoints):>20}\t\t{VOLUME:.2f}\t\t{float(maxMse):.4f}\t\t{float(meanMse)/len(self.speakerCoordinates):.4f}\t\t{float(minMse):.4f}\t\t{int(mse_001)}\t\t{float(maxSsim):.4f}\t\t{float(meanSsim)/len(self.speakerCoordinates):.4f}\t\t{float(minSsim):.4f}\t\t{int(ssim_05)}\t\t{float(maxLmds):.4f}\t\t{float(meanLmds)/len(self.speakerCoordinates):.4f}\t\t{float(minLmds):.4f}\t\t{int(lmds_05)}", flush=True)


 
 
