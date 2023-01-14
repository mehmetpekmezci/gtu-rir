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

import tensorflow as tf

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
   self.mseValues={}
   self.ssimValues={}
   self.lmdsValues={}
   self.maxMseRecords={}
   self.meanMseRecords={}
   self.minMseRecords={}
   self.maxSsimRecords={}
   self.meanSsimRecords={}
   self.minSsimRecords={}
   self.maxLmdsRecords={}
   self.meanLmdsRecords={}
   self.minLmdsRecords={}
   self.volumes={}
   self.volumeMaxMse={}
   self.volumeMeanMse={}
   self.volumeMinMse={}
   self.volumeMaxSsim={}
   self.volumeMeanSsim={}
   self.volumeMinSsim={}
   self.volumeMaxLmds={}
   self.volumeMeanLmds={}
   self.volumeMinLmds={}
   
   self.MSE_THRESHOLD=0.01 # we will select the records having lower MSE than this value
   self.SSIM_THRESHOLD=0.5 # we will select the records having higher SSIM than this value
   self.LMDS_THRESHOLD=0.01 # we will select the records having lower Local Max Diff Sum than this value
   
   self.numberOfRecordsHavingMseLowerThanThresholdPerRoom={}
   self.numberOfRecordsHavingSsimHigherThanThresholdPerRoom={}
   self.numberOfRecordsHavingLmdsLowerThanThresholdPerRoom={}
   self.numberOfRecordsHavingMseLowerThanThresholdPerVolume={}
   self.numberOfRecordsHavingSsimHigherThanThresholdPerVolume={}
   self.numberOfRecordsHavingLmdsLowerThanThresholdPerVolume={}
   self.numberOfRecordsHavingMseLowerThanThresholdPerRoomListOfRecords={}
   self.numberOfRecordsHavingSsimHigherThanThresholdPerRoomListOfRecords={}
   self.numberOfRecordsHavingLmdsLowerThanThresholdPerRoomListOfRecords={}
   

    
 def generateSummary(self):
     workDir=self.rirData.fast_rir_dir+"/code_new/Generated_RIRs_report/"
     summaryDir=workDir+'/summary'
     if not os.path.exists(summaryDir):
            os.makedirs(summaryDir) 
     print ("\n\n\n")
     print (f"5.1 COMPARE - GENERATED DATA - REAL DATA :")
     print ("-------------------------------------------------------------------------------")
     print ("SUMMARY TABLE :")
     r="ROOM"
     #self.logger.info (f"{r:>12}\t\tVOLUME\t\tMAX_MSE\t\tMEAN_MSE\t\tMIN_MSE\t\tMAX_SSIM\t\tMEAN_SSIM MIN_SSIM")
     print (f"{r:>12}\tVOLUME\t\tMAX_MSE\t\tMEAN_MSE\tMIN_MSE\t\tMSE<0.01\tMAX_SSIM\tMEAN_SSIM\tMIN_SSIM\tSSIM>0.5\tMAX_LMDS\tMEAN_LMDS\tMIN_LMDS\tLMDS<0.01")
     
     ##LMDS


     
     for roomPath in sorted(glob.glob(workDir+"/room-*")):
          room=os.path.basename(roomPath)
          #print(room)
          self.mseValues[room]={}
          maxMseRecord=""
          meanMseValue=0
          minMseRecord=""
          meanMseRecord=""
          
          self.ssimValues[room]={}
          maxSsimRecord=""
          meanSsimValue=0
          minSsimRecord=""
          meanSsimRecord=""
          
          self.lmdsValues[room]={}
          maxLmdsRecord=""
          meanLmdsValue=0
          minLmdsRecord=""
          meanLmdsRecord=""
          
          DEPTH=0
          WIDTH=0
          HEIGHT=0
          VOLUME=0
          self.numberOfRecordsHavingMseLowerThanThresholdPerRoom[room]=0
          self.numberOfRecordsHavingSsimHigherThanThresholdPerRoom[room]=0
          self.numberOfRecordsHavingLmdsLowerThanThresholdPerRoom[room]=0


          self.numberOfRecordsHavingMseLowerThanThresholdPerRoomListOfRecords[room]=[]
          self.numberOfRecordsHavingSsimHigherThanThresholdPerRoomListOfRecords[room]=[]
          self.numberOfRecordsHavingLmdsLowerThanThresholdPerRoomListOfRecords[room]=[]
          
          #Precision = TP/ TP+FP = Kesinlik
          #Recall    = TP/ TP+FN = Sensitivity=Duyarlik
          #Accuracy  = TP+TN/ TP+FP+TN+FN = Doğruluk
          
          numberOfRecords=0
          with open(roomPath+"/MSE.db.txt", encoding='utf8') as f:
           for line in f:
               numberOfRecords=numberOfRecords+1
               lineSplit=line.strip().split("=")
               if DEPTH == 0 :
                  vals=lineSplit[0].split("-")
                  DEPTH=float(vals[2])
                  WIDTH=float(vals[4])
                  HEIGHT=float(vals[6])
                  RT60=float(vals[8])
                  VOLUME=DEPTH*WIDTH*HEIGHT
                  self.numberOfRecordsHavingMseLowerThanThresholdPerVolume[VOLUME]=0
                  self.numberOfRecordsHavingSsimHigherThanThresholdPerVolume[VOLUME]=0
                  self.numberOfRecordsHavingLmdsLowerThanThresholdPerVolume[VOLUME]=0
                  
               
               self.mseValues[room][lineSplit[0]]=float(lineSplit[1])
               if maxMseRecord =="" :
                  maxMseRecord=lineSplit[0]
               elif self.mseValues[room][lineSplit[0]]> self.mseValues[room][maxMseRecord] :
                  maxMseRecord=lineSplit[0]
                  
               if minMseRecord =="" :
                  minMseRecord=lineSplit[0]
               elif self.mseValues[room][lineSplit[0]]< self.mseValues[room][minMseRecord] :
                  minMseRecord=lineSplit[0]
                  
               meanMseValue=float(lineSplit[1])+meanMseValue
               
               if self.MSE_THRESHOLD>float(lineSplit[1]):
                  self.numberOfRecordsHavingMseLowerThanThresholdPerRoom[room]=1+self.numberOfRecordsHavingMseLowerThanThresholdPerRoom[room]
                  self.numberOfRecordsHavingMseLowerThanThresholdPerVolume[VOLUME]=1+self.numberOfRecordsHavingMseLowerThanThresholdPerVolume[VOLUME]
                  self.numberOfRecordsHavingMseLowerThanThresholdPerRoomListOfRecords[room].append(lineSplit[0])

           meanMseValue=meanMseValue/numberOfRecords
           f.seek(0)
           ## just to find the record that has the closest mse value to the mean value. 
           for line in f:
               lineSplit=line.strip().split("=")
               if meanMseRecord =="" :
                  meanMseRecord=lineSplit[0]
               elif abs(self.mseValues[room][lineSplit[0]] - meanMseValue) < abs(self.mseValues[room][meanMseRecord] - meanMseValue ):
                  meanMseRecord=lineSplit[0]
             
               
               
               


                  
                     
          with open(roomPath+"/SSIM.db.txt", encoding='utf8') as f:
           for line in f:
               lineSplit=line.strip().split("=")
               self.ssimValues[room][lineSplit[0]]=float(lineSplit[1])
               #print(self.ssimValues[room][lineSplit[0]])
                
               if maxSsimRecord =="" :
                  maxSsimRecord=lineSplit[0]
               elif self.ssimValues[room][lineSplit[0]] > self.ssimValues[room][maxSsimRecord] :

                  maxSsimRecord=lineSplit[0]
                  #print("--------------------------")
                  #print(maxSsimRecord)
                  #print(self.ssimValues[room][maxSsimRecord])
            
                  
               if minSsimRecord =="" :
                  minSsimRecord=lineSplit[0]
               elif self.ssimValues[room][lineSplit[0]]< self.ssimValues[room][minSsimRecord] :
                  minSsimRecord=lineSplit[0]
                  
               meanSsimValue=float(lineSplit[1])+meanSsimValue

               if self.SSIM_THRESHOLD<float(lineSplit[1]):
                  self.numberOfRecordsHavingSsimHigherThanThresholdPerRoom[room]=1+self.numberOfRecordsHavingSsimHigherThanThresholdPerRoom[room]
                  self.numberOfRecordsHavingSsimHigherThanThresholdPerVolume[VOLUME]=1+self.numberOfRecordsHavingSsimHigherThanThresholdPerVolume[VOLUME]
                  self.numberOfRecordsHavingSsimHigherThanThresholdPerRoomListOfRecords[room].append(lineSplit[0])

           meanSsimValue=meanSsimValue/numberOfRecords
           f.seek(0)    
           for line in f:
               lineSplit=line.strip().split("=")
               self.ssimValues[room][lineSplit[0]]=float(lineSplit[1])
               #print(self.ssimValues[room][lineSplit[0]])

               if meanSsimRecord =="" :
                  meanSsimRecord=lineSplit[0]
               elif abs(self.ssimValues[room][lineSplit[0]]-meanSsimValue)< abs(self.ssimValues[room][meanSsimRecord]-meanSsimValue) :
                  meanSsimRecord=lineSplit[0]
          
          
          
          with open(roomPath+"/LMDS.db.txt", encoding='utf8') as f:
           for line in f:
               lineSplit=line.strip().split("=")
               self.lmdsValues[room][lineSplit[0]]=float(lineSplit[1])
               if maxLmdsRecord =="" :
                  maxLmdsRecord=lineSplit[0]
               elif self.lmdsValues[room][lineSplit[0]]> self.lmdsValues[room][maxLmdsRecord] :
                  maxLmdsRecord=lineSplit[0]
                  
               if minLmdsRecord =="" :
                  minLmdsRecord=lineSplit[0]
               elif self.lmdsValues[room][lineSplit[0]]< self.lmdsValues[room][minLmdsRecord] :
                  minLmdsRecord=lineSplit[0]
                  
               meanLmdsValue=float(lineSplit[1])+meanLmdsValue
               
               if self.LMDS_THRESHOLD>float(lineSplit[1]):
                  self.numberOfRecordsHavingLmdsLowerThanThresholdPerRoom[room]=1+self.numberOfRecordsHavingLmdsLowerThanThresholdPerRoom[room]
                  self.numberOfRecordsHavingLmdsLowerThanThresholdPerVolume[VOLUME]=1+self.numberOfRecordsHavingLmdsLowerThanThresholdPerVolume[VOLUME]
                  self.numberOfRecordsHavingLmdsLowerThanThresholdPerRoomListOfRecords[room].append(lineSplit[0])

           meanLmdsValue=meanLmdsValue/numberOfRecords
           f.seek(0)
           ## just to find the record that has the closest lmds value to the mean value. 
           for line in f:
               lineSplit=line.strip().split("=")
               self.lmdsValues[room][lineSplit[0]]=float(lineSplit[1])
               
               if meanLmdsRecord =="" :
                  meanLmdsRecord=lineSplit[0]
               elif abs(self.lmdsValues[room][lineSplit[0]] - meanLmdsValue) < abs(self.lmdsValues[room][meanLmdsRecord] - meanLmdsValue ):
                  meanLmdsRecord=lineSplit[0]
             
               
               
               


           self.maxMseRecords[room]=maxMseRecord
           self.meanMseRecords[room]=meanMseRecord
           self.minMseRecords[room]=minMseRecord

           self.maxSsimRecords[room]=maxSsimRecord
           self.meanSsimRecords[room]=meanSsimRecord
           self.minSsimRecords[room]=minSsimRecord

           self.maxLmdsRecords[room]=maxLmdsRecord
           self.meanLmdsRecords[room]=meanLmdsRecord
           self.minLmdsRecords[room]=minLmdsRecord

           self.volumes[room]=VOLUME

           self.volumeMaxMse[VOLUME]=self.mseValues[room][maxSsimRecord]
           self.volumeMeanMse[VOLUME]=meanMseValue
           self.volumeMinMse[VOLUME]=self.mseValues[room][minMseRecord] 

           self.volumeMaxSsim[VOLUME]=self.ssimValues[room][maxSsimRecord] 
           self.volumeMeanSsim[VOLUME]=meanSsimValue
           self.volumeMinSsim[VOLUME]=self.ssimValues[room][minSsimRecord] 
   
           self.volumeMaxLmds[VOLUME]=self.lmdsValues[room][maxLmdsRecord] 
           self.volumeMeanLmds[VOLUME]=meanLmdsValue
           self.volumeMinLmds[VOLUME]=self.lmdsValues[room][minLmdsRecord] 
   


           

           print (f"{room[:12]:>12}\t{VOLUME:.2f}\t\t{float(self.mseValues[room][maxMseRecord]):.4f}\t\t{float(meanMseValue):.4f}\t\t{float(self.mseValues[room][minMseRecord]):.4f}\t\t{int(self.numberOfRecordsHavingMseLowerThanThresholdPerRoom[room])}\t\t{float(self.ssimValues[room][maxSsimRecord]):.4f}\t\t{float(meanSsimValue):.4f}\t\t{float(self.ssimValues[room][minSsimRecord]):.4f}\t\t{int(self.numberOfRecordsHavingSsimHigherThanThresholdPerRoom[room])}\t\t{float(self.lmdsValues[room][maxLmdsRecord]):.4f}\t\t{float(meanLmdsValue):.4f}\t\t{float(self.lmdsValues[room][minLmdsRecord]):.4f}\t\t{int(self.numberOfRecordsHavingLmdsLowerThanThresholdPerRoom[room])}", flush=True)

     for roomPath in glob.glob(workDir+"/room-*"):
          room=os.path.basename(roomPath)
          
          images = [Image.open(x) for x in [roomPath+'/'+self.maxMseRecords[room]+'.wave.png',roomPath+'/'+self.meanMseRecords[room]+'.wave.png', roomPath+'/'+self.minMseRecords[room]+'.wave.png', roomPath+'/'+self.maxSsimRecords[room]+'.wave.png', roomPath+'/'+self.meanSsimRecords[room]+'.wave.png', roomPath+'/'+self.minSsimRecords[room]+'.wave.png', roomPath+'/'+self.maxLmdsRecords[room]+'.wave.png', roomPath+'/'+self.meanLmdsRecords[room]+'.wave.png', roomPath+'/'+self.minLmdsRecords[room]+'.wave.png']]
          width, height = images[0].size

          total_width = 3*width+200
          max_height = 3*height
          
          fontsize=40

          new_im = Image.new('RGB', (total_width, max_height))

          '''
          new_im.paste(images[0], (0,0))
          
          ImageDraw.Draw(new_im).rectangle([(2*width,0),(total_width,max_height)],fill="white")
          
          
          ImageDraw.Draw(new_im).rectangle([(10,10),(width,height)],outline="#ff3300",width=10)
          new_im.paste(images[1], (width,0))
          ImageDraw.Draw(new_im).rectangle([(width+10,10),(2*width-10,height)],outline="#00ff00",width=10)
          ImageDraw.Draw(new_im).text((2*width+20,int(height/2)-20),"MSE",fill="black",font=ImageFont.truetype("arial.ttf", fontsize))
          new_im.paste(images[2], (0,height))
          ImageDraw.Draw(new_im).text((2*width+20,height-20),room,fill="black",font=ImageFont.truetype("arial.ttf", 20))
          ImageDraw.Draw(new_im).text((2*width+20,height+20),f"VOL:{self.volumes[room]:.2f}",fill="black",font=ImageFont.truetype("arial.ttf", 20))
          
          ImageDraw.Draw(new_im).rectangle([(10,height+10),(width,2*height-10)],outline="#00ff00",width=10)
          new_im.paste(images[3], (width,height))
          ImageDraw.Draw(new_im).rectangle([(width+10,height+10),(2*width-10,2*height-10)],outline="#ff3300",width=10)
          ImageDraw.Draw(new_im).text((2*width+20,int(height+height/2)-20),"SSIM",fill="black",font=ImageFont.truetype("arial.ttf", fontsize))
          '''


          ImageDraw.Draw(new_im).rectangle([(3*width,0),(total_width,max_height)],fill="white")
          
          new_im.paste(images[0], (0,0))
          ImageDraw.Draw(new_im).rectangle([(10,10),(width,height)],outline="#ff3300",width=10)
          
          new_im.paste(images[1], (width,0))
          ImageDraw.Draw(new_im).rectangle([(width+10,10),(2*width-10,height)],outline="#0000ff",width=10)
          
          new_im.paste(images[2], (2*width,0))
          ImageDraw.Draw(new_im).rectangle([(2*width+10,10),(3*width-10,height)],outline="#00ff00",width=10)
          
          ImageDraw.Draw(new_im).text((3*width+30,int(height/2)-30),"MSE",fill="black",font=ImageFont.truetype("arial.ttf", fontsize))
          
          ImageDraw.Draw(new_im).text((3*width+20,height-20),room,fill="black",font=ImageFont.truetype("arial.ttf", 20))
          ImageDraw.Draw(new_im).text((3*width+20,height+20),f"VOL:{self.volumes[room]:.2f}",fill="black",font=ImageFont.truetype("arial.ttf", 20))
          
          new_im.paste(images[5], (0,height))
          ImageDraw.Draw(new_im).rectangle([(10,height+10),(width,2*height-10)],outline="#ff3300",width=10)
          
          new_im.paste(images[4], (width,height))
          ImageDraw.Draw(new_im).rectangle([(width+10,height+10),(2*width-10,2*height-10)],outline="#0000ff",width=10)
          
          new_im.paste(images[3], (2*width,height))
          ImageDraw.Draw(new_im).rectangle([(2*width+10,height+10),(3*width-10,2*height-10)],outline="#00ff00",width=10)
          
          
          ImageDraw.Draw(new_im).text((3*width+20,int(height+height/2)-20),"SSIM",fill="black",font=ImageFont.truetype("arial.ttf", fontsize))


          new_im.paste(images[6], (0,2*height))
          ImageDraw.Draw(new_im).rectangle([(10,2*height+10),(width,3*height-10)],outline="#ff3300",width=10)
          
          new_im.paste(images[7], (width,2*height))
          ImageDraw.Draw(new_im).rectangle([(width+10,2*height+10),(2*width-10,3*height-10)],outline="#0000ff",width=10)
          
          new_im.paste(images[8], (2*width,2*height))
          ImageDraw.Draw(new_im).rectangle([(2*width+10,2*height+10),(3*width-10,3*height-10)],outline="#00ff00",width=10)
          
          ImageDraw.Draw(new_im).text((3*width+20,int(2*height+height/2)-20),"LMDS",fill="black",font=ImageFont.truetype("arial.ttf", fontsize))




          new_im.save(summaryDir+'/report.wave.'+room+'.png')


          #print (f"\n\n{room[:12]:>12}\n\tMAX MSE RECORD:{self.maxMseRecords[room]}\n\tMIN MSE RECORD:{self.minMseRecords[room]}\n\tMAX SSIM RECORD:{self.maxSsimRecords[room]}\n\tMIN SSIM RECORD:{self.minSsimRecords[room]}")

          shutil.copyfile(self.rirData.fast_rir_dir+'/code_new/Generated_RIRs/'+self.maxMseRecords[room]+'.wav', summaryDir+'/report.'+self.maxMseRecords[room]+'.max.mse.generated.wav')
          i=int(self.maxMseRecords[room].split("-")[-1])
          real_data=librosa.resample(self.rirData.rir_data[i][-1], orig_sr=44100, target_sr=self.rirData.reduced_sampling_rate)[:self.rirData.data_length]
          wavfile.write( summaryDir+'/report.'+self.maxMseRecords[room]+'.max.mse.real.wav',self.rirData.reduced_sampling_rate,np.array(real_data).astype(np.float32))
         
          shutil.copyfile(self.rirData.fast_rir_dir+'/code_new/Generated_RIRs/'+self.minMseRecords[room]+'.wav', summaryDir+'/report.'+self.minMseRecords[room]+'.min.mse.generated.wav')
          i=int(self.minMseRecords[room].split("-")[-1])
          real_data=librosa.resample(self.rirData.rir_data[i][-1], orig_sr=44100, target_sr=self.rirData.reduced_sampling_rate)[:self.rirData.data_length]
          wavfile.write( summaryDir+'/report.'+self.minMseRecords[room]+'.min.mse.real.wav',self.rirData.reduced_sampling_rate,np.array(real_data).astype(np.float32))
         
          shutil.copyfile(self.rirData.fast_rir_dir+'/code_new/Generated_RIRs/'+self.maxSsimRecords[room]+'.wav', summaryDir+'/report.'+self.maxSsimRecords[room]+'.max.ssim.generated.wav')
          i=int(self.maxSsimRecords[room].split("-")[-1])
          real_data=librosa.resample(self.rirData.rir_data[i][-1], orig_sr=44100, target_sr=self.rirData.reduced_sampling_rate)[:self.rirData.data_length]
          wavfile.write( summaryDir+'/report.'+self.maxSsimRecords[room]+'.max.ssim.real.wav',self.rirData.reduced_sampling_rate,np.array(real_data).astype(np.float32))
         
          shutil.copyfile(self.rirData.fast_rir_dir+'/code_new/Generated_RIRs/'+self.minSsimRecords[room]+'.wav', summaryDir+'/report.'+self.minSsimRecords[room]+'.min.ssim.generated.wav')
          i=int(self.minSsimRecords[room].split("-")[-1])
          real_data=librosa.resample(self.rirData.rir_data[i][-1], orig_sr=44100, target_sr=self.rirData.reduced_sampling_rate)[:self.rirData.data_length]
          wavfile.write( summaryDir+'/report.'+self.minSsimRecords[room]+'.min.ssim.real.wav',self.rirData.reduced_sampling_rate,np.array(real_data).astype(np.float32))
                  
          #self.plotWavSummarySamples(room,summaryDir)
 
          print("Graphics are generated for room : "+room, flush=True)
         
     self.plotWav(list(self.volumes.values()),list(self.volumeMaxMse.values()),saveToPath=summaryDir+"/volume-maxMse.png",title="MAX MSE")        
     self.plotWav(list(self.volumes.values()),list(self.volumeMeanMse.values()),saveToPath=summaryDir+"/volume-meanMse.png",title="MEAN MSE")        
     self.plotWav(list(self.volumes.values()),list(self.volumeMinMse.values()),saveToPath=summaryDir+"/volume-minMse.png",title="MIN MSE")        
     self.plotWav(list(self.volumes.values()),list(self.volumeMaxSsim.values()),saveToPath=summaryDir+"/volume-maxSSIM.png",title="MAX SSIM")        
     self.plotWav(list(self.volumes.values()),list(self.volumeMeanSsim.values()),saveToPath=summaryDir+"/volume-meanSSIM.png",title="MEAN SSIM")        
     self.plotWav(list(self.volumes.values()),list(self.volumeMinSsim.values()),saveToPath=summaryDir+"/volume-minLmds.png",title="MIN LMDS")        
     self.plotWav(list(self.volumes.values()),list(self.volumeMaxLmds.values()),saveToPath=summaryDir+"/volume-maxLmds.png",title="MAX LMDS")        
     self.plotWav(list(self.volumes.values()),list(self.volumeMeanLmds.values()),saveToPath=summaryDir+"/volume-meanLmds.png",title="MEAN LMDS")        
     self.plotWav(list(self.volumes.values()),list(self.volumeMinLmds.values()),saveToPath=summaryDir+"/volume-minLmds.png",title="MIN LMDS")        
     self.plotWav(list(self.volumes.values()),list(self.numberOfRecordsHavingMseLowerThanThresholdPerVolume.values()),saveToPath=summaryDir+"/volume-MSE_001.png",title="MSE < 0.01")        
     self.plotWav(list(self.volumes.values()),list(self.numberOfRecordsHavingSsimHigherThanThresholdPerVolume.values()),saveToPath=summaryDir+"/volume-SSIM_05.png",title="SSIM > 0.5")        
     self.plotWav(list(self.volumes.values()),list(self.numberOfRecordsHavingLmdsLowerThanThresholdPerVolume.values()),saveToPath=summaryDir+"/volume-LMDS_001.png",title="LMDS < 0.01")        
      


  
 
 def takeFirst(self,elem):
    return elem[0]
 
 def plotWav(self,volumes,data,title="",show=False,saveToPath=None):
     #plt.clf()
     
     plt.subplot(1,1,1)
     
     #print("volumes")
     #print(volumes)
     #print("data")
     #print(data)
     d=[(volumes[i], data[i]) for i in range(0, len(volumes))]
     #print("d")
     #print(d)
     d.sort(key=self.takeFirst)
     #print("d.sorted")
     #print(d)
     plt.scatter(*zip(*d),color='b')

     plt.title(title)
     plt.xlabel('Volume')
     plt.ylabel('Values')

     if show :
        plt.show()
     if saveToPath is not None :
        plt.savefig(saveToPath)
        
     plt.close()   




 def plotWavSummarySamples(self,room,summaryDir):
 
       
       generated_datas=[]
       real_datas=[]
       for record_name in self.numberOfRecordsHavingMseLowerThanThresholdPerRoomListOfRecords[room]:
           wave_name=record_name+".wav"
           generated_data,rate=librosa.load(self.rirData.fast_rir_dir+"/code_new/Generated_RIRs/"+wave_name,sr=self.rirData.reduced_sampling_rate,mono=True)
           i=int(record_name.split("-")[-1])
           real_datas.append(librosa.resample(self.rirData.rir_data[i][-1], orig_sr=44100, target_sr=self.rirData.reduced_sampling_rate)[:self.rirData.data_length])
           generated_datas.append(generated_data)
       self.plotGroupWav(real_datas,generated_datas,room+" RECORDS HAVING MSE < 0.01 ",saveToPath=summaryDir+"/"+room+"-RECORDS-MSE_LESS_THAN_0.01.png")    
              
       generated_datas=[]
       real_datas=[]
       for record_name in self.numberOfRecordsHavingSsimHigherThanThresholdPerRoomListOfRecords[room]:
           wave_name=record_name+".wav"
           generated_data,rate=librosa.load(self.rirData.fast_rir_dir+"/code_new/Generated_RIRs/"+wave_name,sr=self.rirData.reduced_sampling_rate,mono=True)
           i=int(record_name.split("-")[-1])
           real_datas.append(librosa.resample(self.rirData.rir_data[i][-1], orig_sr=44100, target_sr=self.rirData.reduced_sampling_rate)[:self.rirData.data_length])
           generated_datas.append(generated_data)
       self.plotGroupWav(real_datas,generated_datas,room+" RECORDS HAVING SSIM > 0.5 ",saveToPath=summaryDir+"/"+room+"-RECORDS-SSIM_MORE_THAN_0.5.png")    
       
       generated_datas=[]
       real_datas=[]
       for record_name in self.numberOfRecordsHavingLmdsLowerThanThresholdPerRoomListOfRecords[room]:
           wave_name=record_name+".wav"
           generated_data,rate=librosa.load(self.rirData.fast_rir_dir+"/code_new/Generated_RIRs/"+wave_name,sr=self.rirData.reduced_sampling_rate,mono=True)
           i=int(record_name.split("-")[-1])
           real_datas.append(librosa.resample(self.rirData.rir_data[i][-1], orig_sr=44100, target_sr=self.rirData.reduced_sampling_rate)[:self.rirData.data_length])
           generated_datas.append(generated_data)
       self.plotGroupWav(real_datas,generated_datas,room+" RECORDS HAVING LMDS < 0.01 ",saveToPath=summaryDir+"/"+room+"-RECORDS-LMDS_LESS_THAN_0.01.png")    
       
              
 def plotGroupWav(self,real_data,generated_data,title,show=False,saveToPath=None):
     #plt.clf()
     
     N=int(math.sqrt(len(real_data)))+1
     fig, axs = plt.subplots(N, N,figsize=(100, 100))
     
     for i in range(N):
         for j in range(N):
           if i*N+j < len(real_data) :
              axs[i, j].plot(real_data[i*N+j],color='r', label='real_data')
              axs[i, j].plot(generated_data[i*N+j],color='b', label='generated_data')
     

     plt.title(title)

     if show :
        plt.show()
     if saveToPath is not None :
        plt.savefig(saveToPath)
        
     plt.close()    
         
         
'''
Tables :

roomId - Dimensions - VOLUME - MAX MSE - MEAN MSE - MIN MSE - MAX SSIM - MEAN SSIM - MIN SSIM




Graphs :



ROOM-ID ( 11 adet)  :  Room's Best MSE Wav Graph (Red/Blue) , Room's Medium MSE Wav Graph (Red/Blue), Room's Worst MSE Match Wav Graph (Red/Blue)  
                       Room's Best SSIM Wav Graph (Red/Blue) , Room's Medium SSIM Wav Graph (Red/Blue), Room's Worst SSIM Match Wav Graph (Red/Blue)  
                       
                       
volume- MSE grafiği  --BLUE
volume - SSIM grafiği -- RED


heatmap - Mx-My-MSE
heatmap - Mx-My-SSIM
heatmap - Sx-Sy-MSE
heatmap - Sx-Sy-SSIM
heatmap - Sx-Sy-MSE
heatmap - Sx-Sy-SSIM
'''
