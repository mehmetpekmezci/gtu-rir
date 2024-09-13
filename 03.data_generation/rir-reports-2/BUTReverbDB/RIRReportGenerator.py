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

   self.metrics=["MSE","SSIM","MFCC_MSE","MFCC_SSIM","MFCC_CROSS_ENTROPY","GLITCH_COUNT"] # metric."db.txt"
   
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

     for roomPath in sorted(glob.glob(workDir+"/[H,V]*/*")):
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


'''
     for roomPath in glob.glob(workDir+"/room-*"):
          room=os.path.basename(roomPath)
          
          #images = [Image.open(x) for x in [roomPath+'/'+self.maxMseRecords[room]+'.wave.png',roomPath+'/'+self.meanMseRecords[room]+'.wave.png', roomPath+'/'+self.minMseRecords[room]+'.wave.png', roomPath+'/'+self.maxSsimRecords[room]+'.wave.png', roomPath+'/'+self.meanSsimRecords[room]+'.wave.png', roomPath+'/'+self.minSsimRecords[room]+'.wave.png', roomPath+'/'+self.maxLmdsRecords[room]+'.wave.png', roomPath+'/'+self.meanLmdsRecords[room]+'.wave.png', roomPath+'/'+self.minLmdsRecords[room]+'.wave.png']]
          images = [Image.open(x) for x in [roomPath+'/'+self.maxMseRecords[room]+'.wave.png',roomPath+'/'+self.meanMseRecords[room]+'.wave.png', roomPath+'/'+self.minMseRecords[room]+'.wave.png', roomPath+'/'+self.maxSsimRecords[room]+'.wave.png', roomPath+'/'+self.meanSsimRecords[room]+'.wave.png', roomPath+'/'+self.minSsimRecords[room]+'.wave.png']]
          width, height = images[0].size

          total_width = 3*width+200
          max_height = 2*height
          
          fontsize=40

          new_im = Image.new('RGB', (total_width, max_height))

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


          new_im.save(summaryDir+'/report.wave.'+room+'.png')


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
     self.plotWav(list(self.volumes.values()),list(self.volumeMinSsim.values()),saveToPath=summaryDir+"/volume-minSSIM.png",title="MIN SSIM")        
     self.plotWav(list(self.volumes.values()),list(self.numberOfRecordsHavingMseLowerThanThresholdPerVolume.values()),saveToPath=summaryDir+"/volume-MSE_001.png",title="MSE < 0.01")        
     self.plotWav(list(self.volumes.values()),list(self.numberOfRecordsHavingSsimHigherThanThresholdPerVolume.values()),saveToPath=summaryDir+"/volume-SSIM_05.png",title="SSIM > 0.5")        
      


  
 
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
       
#       generated_datas=[]
#       real_datas=[]
#       for record_name in self.numberOfRecordsHavingLmdsLowerThanThresholdPerRoomListOfRecords[room]:
#           wave_name=record_name+".wav"
#           generated_data,rate=librosa.load(self.rirData.fast_rir_dir+"/code_new/Generated_RIRs/"+wave_name,sr=self.rirData.reduced_sampling_rate,mono=True)
#           i=int(record_name.split("-")[-1])
#           real_datas.append(librosa.resample(self.rirData.rir_data[i][-1], orig_sr=44100, target_sr=self.rirData.reduced_sampling_rate)[:self.rirData.data_length])
#           generated_datas.append(generated_data)
#       self.plotGroupWav(real_datas,generated_datas,room+" RECORDS HAVING LMDS < 0.01 ",saveToPath=summaryDir+"/"+room+"-RECORDS-LMDS_LESS_THAN_0.01.png")    
       
              
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
