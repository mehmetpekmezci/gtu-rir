

import gc
from RIRHeader import *
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
 def __init__(self, logger,data_dir,report_dir,selected_room_id):
 
   self.logger  = logger
   self.data_dir=data_dir
   self.report_dir=report_dir 
   self.selected_room_id=selected_room_id
   self.script_dir=os.path.dirname(os.path.realpath(__file__))
   self.sampling_rate=44100 # 44100 sample points per second
   self.reduced_sampling_rate=16000
   self.rir_seconds=2
   self.track_length=self.rir_seconds*self.sampling_rate 
   self.final_sound_data_length=int(self.track_length/self.rir_seconds)
   self.roomProperties={}
   self.rooms_and_configs={}
   self.data_length=4096
   self.SPECTROGRAM_DIM=11

   plt.subplot(1,1,1)

   print("self.data_dir="+self.data_dir)
          
   self.rir_data_file_path=self.data_dir+"/RIR.pickle.dat"
   
   self.rir_data=[]  ##  "RIR.dat" --> list of list [34]

   if  os.path.exists( self.rir_data_file_path) :
         rir_data_file=open(self.rir_data_file_path,'rb')
         self.rir_data=pickle.load(rir_data_file)
         rir_data_file.close()
   else :
         print(self.rir_data_file_path+" not exists")
         exit(1)

   print("rirData Length ="+str(len(self.rir_data)))
   self.rir_data_field_numbers={"timestamp":0,"speakerMotorIterationNo":1,"microphoneMotorIterationNo":2,"speakerMotorIterationDirection":3,"currentActiveSpeakerNo":4,"currentActiveSpeakerChannelNo":5,
                                "physicalSpeakerNo":6,"microphoneStandInitialCoordinateX":7,"microphoneStandInitialCoordinateY":8,"microphoneStandInitialCoordinateZ":9,"speakerStandInitialCoordinateX":10,
                                "speakerStandInitialCoordinateY":11,"speakerStandInitialCoordinateZ":12,"microphoneMotorPosition":13,"speakerMotorPosition":14,"temperatureAtMicrohponeStand":15,
                                "humidityAtMicrohponeStand":16,"temperatureAtMSpeakerStand":17,"humidityAtSpeakerStand":18,"tempHumTimestamp":19,"speakerRelativeCoordinateX":20,"speakerRelativeCoordinateY":21,
                                "speakerRelativeCoordinateZ":22,"microphoneStandAngle":23,"speakerStandAngle":24,"speakerAngleTheta":25,"speakerAnglePhi":26,"mic_RelativeCoordinateX":27,"mic_RelativeCoordinateY":28,
                                "mic_RelativeCoordinateZ":29,"mic_DirectionX":30,"mic_DirectionY":31,"mic_DirectionZ":32,"mic_Theta":33,"mic_Phi":34,"essFilePath":35,
                                "roomId":36,"configId":37,"micNo":38, ## THESE VALUES WILL BE PARSED FROM essFilePath
                                "roomWidth":39,"roomHeight":40,"roomDepth":41, ## THESE VALUES WILL BE RETREIVED FROM ROOM PREOPERTIES                              
                                "rt60":42, ## RT60 will be calculated                              
                                "rirData":43 ## will be loaded from wav file   
                              } 
                             
   ## essFilePath =   <room_id> / <config_id> / <spkstep-SPKSTEPNO-micstep-MICSTEPNO-spkno-SPKNO> / receivedEssSignal-MICNO.wav
   
   self.transmittedEssWav=None
                                 
   # micNo
   #
   #    5          1
   #    |          |
   # 4-------||-------0
   #    |    ||    |
   #    6    ||    2                    


   # physicalSpeakerNo
   #              
   # 3---2---||
   #         || \    
   #         ||   1                      
   #         ||     \                      
   #         ||      0                    

  








 def plotWav(self,real_data,generated_data,MSE,SSIM,glitch_points,title,show=False,saveToPath=None):
     plt.clf()

     #plt.subplot(1,1,1)
     minValue=np.min(real_data)
     minValue2=np.min(generated_data)
     if minValue2 < minValue:
        minValue=minValue2

     plt.text(2600, minValue+abs(minValue)/11, f"MSE={float(MSE):.4f}\nSSIM={float(SSIM):.4f}\nGLITCH={int(len(glitch_points))}", style='italic',
        bbox={'facecolor': 'gray', 'alpha': 0.5, 'pad': 10})

        
     #plt.title(r'$\alpha_i > \beta_i$', fontsize=20)
     #plt.text(1, -0.6, r'$\sum_{i=0}^\infty x_i$', fontsize=20)
     #plt.text(0.6, 0.6, r'$\mathcal{A}\mathrm{sin}(2 \omega t)$',
     #    fontsize=20)   
        
     #plt.plot(real_data,color='r', label='real_data')
     plt.plot(real_data,color='#101010', label='real_data')
     plt.plot(generated_data,color='#909090', label='generated_data')
     plt.title(title)
     plt.xlabel('Time')
     plt.ylabel('Amlpitude')
     plt.legend(loc = "upper right")
    
     x=glitch_points
     y=generated_data[x]
     plt.scatter(x,y,color="black")

     if show :
        plt.show()
     if saveToPath is not None :
        plt.savefig(saveToPath)
        
     #plt.close()    
         
         
 def plotSpectrogram(self,title,power_to_db,sr,show=False,saveToPath=None):
 
 
     
     
     ###plt.figure(figsize=(8, 7))
     fig, ax = plt.subplots()
     #fig.set_figheight(self.SPECTROGRAM_DIM)
     #img=librosa.display.specshow(power_to_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma', hop_length=hop_length,ax=ax)
     img=librosa.display.specshow(power_to_db, sr=sr, x_axis='time',  cmap='magma',  ax=ax)
     
     
     fig.colorbar(img, ax=ax,label='dB')
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
         
 

 def allignVertically(self,generated_data,real_data):
#         return generated_data,real_data

         generated_data_max=np.max(np.abs(generated_data))
         real_data_max=np.max(np.abs(real_data))
         generated_data=generated_data/generated_data_max
         real_data=real_data/real_data_max
         return generated_data,real_data


 def allignHorizontally(self,generated_data,real_data):
         max_point_index_within_first_1000_points_real_data=self.getLocalArgMax(1000,real_data) #np.argmax(real_data[0:1000])
         max_point_index_within_first_1000_points_generated_data=self.getLocalArgMax(1000,generated_data)#np.argmax(generated_data[0:1000])

         #print("max_point_index_within_first_1000_points_real_data"+str(max_point_index_within_first_1000_points_real_data))
         #print("max_point_index_within_first_1000_points_generated_data"+str(max_point_index_within_first_1000_points_generated_data))

         diff=int(abs(max_point_index_within_first_1000_points_real_data-max_point_index_within_first_1000_points_generated_data)/2)

         if diff > 0 :
           if    max_point_index_within_first_1000_points_real_data > max_point_index_within_first_1000_points_generated_data :
                 new_generated_data=np.zeros(generated_data.shape)
                 new_generated_data[diff:]=generated_data[:-diff]
                 generated_data=new_generated_data

                 new_real_data=np.zeros(real_data.shape)
                 new_real_data[:-diff]=real_data[diff:]
                 real_data=new_real_data
           else :
                 new_generated_data=np.zeros(generated_data.shape)
                 #new_generated_data[diff:]=generated_data[:-diff]
                 new_generated_data[:-diff]=generated_data[diff:]
                 generated_data=new_generated_data

                 new_real_data=np.zeros(real_data.shape)
                 #new_real_data[:-diff]=real_data[diff:]
                 new_real_data[diff:]=real_data[:-diff]
                 real_data=new_real_data

         
         #print("0.MAX ALLIGNMENT : np.argmax(real_data[0:1000]):"+str(self.getLocalArgMax(1000,real_data)))
         #print("0.MAX ALLIGNMENT : np.argmax(generated_data[0:1000]):"+str(self.getLocalArgMax(1000,generated_data)))

         localArgMaxReal=self.getLocalArgMax(1000,real_data)
         localArgMaxGenerted=self.getLocalArgMax(1000,generated_data)
         diff=1
         if localArgMaxReal ==localArgMaxGenerted+diff :
                  new_generated_data=np.zeros(generated_data.shape)
                  new_generated_data[diff:]=generated_data[:-diff]
                  generated_data=new_generated_data

         elif  localArgMaxGenerted == localArgMaxReal+diff:
                  new_generated_data=np.zeros(generated_data.shape)
                  #new_generated_data[diff:]=generated_data[:-diff]
                  new_generated_data[:-diff]=generated_data[diff:]
                  generated_data=new_generated_data

                  

         #real_data1=librosa.resample(self.rir_data[i][-1], orig_sr=44100, target_sr=sr)
         #real_data1=real_data1[:generated_data.shape[0]]
         #print("1.MAX ALLIGNMENT : np.argmax(real_data1[0:1000]):"+str(self.getLocalArgMax(1000,real_data1)))
         # test edildi problem yok :)
         return generated_data,real_data
 
 
 
         
 def diffBetweenGeneratedAndRealRIRData(self):

     if  os.path.exists( self.report_dir+"/."+self.selected_room_id+".wavesAndSpectrogramsGenerated") :
        print("wavesAndSpectrograms already generated for "+self.selected_room_id)
        return
        
     sr=16000
     ## STRUCTURAL SIMILARITY  librosa.segment.cross_similarity
     ## https://pytorch.org/ignite/generated/ignite.metrics.SSIM.html
     ## https://torchmetrics.readthedocs.io/en/stable/image/structural_similarity.html
     ## https://stackoverflow.com/questions/53956932/use-pytorch-ssim-loss-function-in-my-model
     ## https://github.com/VainF/pytorch-msssim
     ## https://github.com/francois-rozet/piqa
     
     
     #https://www.tensorflow.org/api_docs/python/tf/image/ssim
     
     
     
     #https://www.kaggle.com/code/msripooja/steps-to-convert-audio-clip-to-spectrogram
     #https://www.frank-zalkow.de/en/create-audio-spectrograms-with-python.html  ## this is with STFT
     #https://stackoverflow.com/questions/44787437/how-to-convert-a-wav-file-to-a-spectrogram-in-python3
     #https://analyticsindiamag.com/hands-on-guide-to-librosa-for-handling-audio-files/
     #https://dsp.stackexchange.com/questions/72027/python-audio-analysis-which-spectrogram-should-i-use-and-why
     
    
     #MSE
     #SSIM
     #MFCC-MSE
     #MFCC-SSIM
     #MFCC-CROSS_ENTROPY
     # (MFCC-MSE + MFCC-SSIM) -- no need.

     print("len(self.rir_data):{len(self.rir_data)}")
     print(f"selected_room_id={self.selected_room_id}")

     NOT_EXISTING_WAV_FILE_COUNT=0

     for i in range(len(self.rir_data)):
       dataline=self.rir_data[i] 
       essFilePath=str(dataline[int(self.rir_data_field_numbers['essFilePath'])])
       roomId=dataline[int(self.rir_data_field_numbers['roomId'])] 
       if roomId != self.selected_room_id :
             continue


       configId=dataline[int(self.rir_data_field_numbers['configId'])] 
       roomWorkDir=self.report_dir+"/"+roomId+"/"+configId
       rt60=str(self.rir_data[i][int(self.rir_data_field_numbers['rt60'])])

       speakerIterationNo=int(dataline[int(self.rir_data_field_numbers['speakerMotorIterationNo'])])
       microphoneIterationNo=int(dataline[int(self.rir_data_field_numbers['microphoneMotorIterationNo'])])
       physicalSpeakerNo=int(dataline[int(self.rir_data_field_numbers['physicalSpeakerNo'])]) 
       micNo=dataline[int(self.rir_data_field_numbers['micNo'])] 

       record_name = f"SPEAKER_ITERATION-{speakerIterationNo}-MICROPHONE_ITERATION-{microphoneIterationNo}-PHYSICAL_SPEAKER_NO-{physicalSpeakerNo}-MICROPHONE_NO-{micNo}"
         
       wave_name=record_name+".wav"
       
       if not os.path.exists( roomWorkDir+"/"+wave_name ):
           NOT_EXISTING_WAV_FILE_COUNT+=1
           continue
       #print(roomWorkDir+"/"+wave_name+" filename="+essFilePath+"  rir_data rt60 : "+rt60)
        
       try:
         
         generated_data,rate=librosa.load(roomWorkDir+"/"+wave_name,sr=sr,mono=True)
         generated_data=generated_data[0:3500]  
         real_data=librosa.resample(self.rir_data[i][-1], orig_sr=44100, target_sr=sr) 
         real_data=real_data[:generated_data.shape[0]]
        
         
         generated_data,real_data=self.allignHorizontally(generated_data,real_data)         
         
         ######### BEGIN : YATAY ESITLEME (dikey esitleme zaten maksimum noktalarini esitliyerek yapilmisti)
         generated_data=generated_data-np.sum(generated_data)/generated_data.shape[0]
         ## bu sekilde ortalamasi 0'a denk gelecek
         ######### END: YATAY ESITLEME (dikey esitleme zaten maksimum noktalarini esitliyerek yapilmisti)

         generated_data,real_data=self.allignVertically(generated_data,real_data)         
         
         MSE=np.square(np.subtract(real_data,generated_data)).mean()
         
         
         generated_data_tiled=np.tile(generated_data, (2, 1)) ## duplicate 1d data to 2d
         real_data_tiled=np.tile(real_data, (2, 1)) ## duplicate 1d data to 2d

         generated_data_tiled=np.reshape(generated_data_tiled,(1,1,generated_data_tiled.shape[0],generated_data_tiled.shape[1]))
         real_data_tiled=np.reshape(real_data_tiled,(1,1,real_data_tiled.shape[0],real_data_tiled.shape[1]))

         generated_data_tensor=torch.from_numpy(generated_data_tiled)
         real_data_tensor=torch.from_numpy(real_data_tiled)


         # data_range  = np.max(real_data)-np.min(real_data) --> bunu bi 2 olarak set ediyoruz.
         #SSIM=ssim(generated_data_tensor,real_data_tensor, data_range=2.0,size_average=True).item()
         SSIM=ssim(generated_data_tensor.float(),real_data_tensor.float(),data_range=4.0,size_average=True).item()
         
         glitch_points=self.getGlitchPoints(generated_data,real_data)

         #crossCorrelation=self.getCrossCorrelation(generated_data,real_data)

         #title=record_name
         #title=f"RT60-{float(labels_embeddings_batch[9]):.2f}-MX-{float(labels_embeddings_batch[0]):.2f}-MY-{float(labels_embeddings_batch[1]):.2f}-MZ-{float(labels_embeddings_batch[2]):.2f}-SX-{float(labels_embeddings_batch[3]):.2f}-SY-{float(labels_embeddings_batch[4]):.2f}-SZ-{float(labels_embeddings_batch[5]):.2f}"
         title=""

         ## plot only 1 of 10 samples.
         if True or i%10 == 0 :
            self.plotWav(real_data,generated_data,MSE,SSIM,glitch_points,title,saveToPath=roomWorkDir+"/"+record_name+".wave.png")
         
         f = open(roomWorkDir+"/MSE.db.txt", "a")
         f.write(record_name+"="+str(MSE)+"\n")
         f.close()
         f = open(roomWorkDir+"/SSIM.db.txt", "a")
         f.write(record_name+"="+str(SSIM)+"\n")
         f.close()
         f = open(roomWorkDir+"/GLITCH_COUNT.db.txt", "a")
         f.write(record_name+"="+str(len(glitch_points))+"\n")
         f.close()
       except:
           print("Exception: roomId="+roomId+", record_name="+record_name)
           traceback.print_exc()

     open( self.report_dir+"/."+self.selected_room_id+".wavesAndSpectrogramsGenerated", 'a').close()   
     if NOT_EXISTING_WAV_FILE_COUNT > 0 :
        print(f" THERE WAS {NOT_EXISTING_WAV_FILE_COUNT} FILES THAT WERE NOT FOUND")
 
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

# def isBiggerThanNextN(self,checkIndex,N,threshold,generated,real):
#     isBigger=True
#     for i in range(N):
#         if abs(generated[checkIndex]-real[checkIndex+i]) < threshold:
#             isBigger=False
#     return isBigger

 def getLocalArgMax(self,limit,data):
     maximum_value=np.max(data[:limit])*4/5 # 20% error threshold for max
     return np.argmax(data[:limit]>=maximum_value)

        
 def localMaxDiffSum(self,signal1,signal2,numberOfChunks=64):
     maxDiffSum=0
     chunkSize=int(self.data_length/numberOfChunks)
     for i in range(numberOfChunks):
         #max1=np.max(np.abs(signal1[i:i+chunkSize]))
         #max2=np.max(np.abs(signal2[i:i+chunkSize]))
         max1=np.mean(np.abs(signal1[i:i+chunkSize]))
         max2=np.mean(np.abs(signal2[i:i+chunkSize]))
         maxDiff=abs(max2-max1)
         maxDiffSum=maxDiffSum+maxDiff
     return maxDiffSum/numberOfChunks



# def getCrossCorrelation(self,generated,real):
#     return scipy.signal.correlate(generated,real)[0]*10000
#



