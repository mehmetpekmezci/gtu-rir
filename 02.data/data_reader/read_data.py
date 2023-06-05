#!/usr/bin/env python3

import sys
import os
import pickle
#import scipy.io.wavfile
#from scipy import signal
#from scipy import stats
#import librosa.display
#import librosa
#import matplotlib.pyplot as plt
#from acoustics.utils import _is_1d
#from acoustics.signal import bandpass
#from acoustics.bands import (_check_band_type, octave_low, octave_high, third_low, third_high)
#np.set_printoptions(threshold=sys.maxsize)
#from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
#import torchaudio
#import torch


class RIRData :
 def __init__(self, dataFilePath):
 
   self.dataFilePath  = dataFilePath
   self.sampling_rate=44100 # 44100 sample points per second
   self.reduced_sampling_rate=16000
   self.rir_seconds=2
   self.track_length=self.rir_seconds*self.sampling_rate 
   self.final_sound_data_length=int(self.track_length/self.rir_seconds)
   self.roomProperties={}
   self.rooms_and_configs={}
   #self.rir_data_file_path=self.data_dir+"/RIR.pickle.dat"
   
   self.rir_data=[]  ##  "RIR.dat" --> list of list [34]
   if  os.path.exists(self.dataFilePath) :
         rir_data_file=open(self.dataFilePath,'rb')
         self.rir_data=pickle.load(rir_data_file)
         rir_data_file.close()
   else :
         print("Data file "+ self.dataFilePath + " not found, exiting ...")
         exit(1)

   print("rirData Length ="+str(len(self.rir_data)))
   self.rir_data_field_numbers={"timestamp":0,"speakerMotorIterationNo":1,"microphoneMotorIterationNo":2,"speakerMotorIterationDirection":3,"currentActiveSpeakerNo":4,"currentActiveSpeakerChannelNo":5,
                                "physicalSpeakerNo":6,"microphoneStandInitialCoordinateX":7,"microphoneStandInitialCoordinateY":8,"microphoneStandInitialCoordinateZ":9,"speakerStandInitialCoordinateX":10,
                                "speakerStandInitialCoordinateY":11,"speakerStandInitialCoordinateZ":12,"microphoneMotorPosition":13,"speakerMotorPosition":14,"temperatureAtMicrohponeStand":15,
                                "humidityAtMicrohponeStand":16,"temperatureAtMSpeakerStand":17,"humidityAtSpeakerStand":18,"tempHumTimestamp":19,"speakerRelativeCoordinateX":20,"speakerRelativeCoordinateY":21,
                                "speakerRelativeCoordinateZ":22,"microphoneStandAngle":23,"speakerStandAngle":24,"speakerAngleTheta":25,"speakerAnglePhi":26,"mic_RelativeCoordinateX":27,"mic_RelativeCoordinateY":28,
                                "mic_RelativeCoordinateZ":29,"mic_DirectionX":30,"mic_DirectionY":31,"mic_DirectionZ":32,"mic_Theta":33,"mic_Phi":34,"essFilePath":35,
                                "roomId":36,"configId":37,"micNo":38, 
                                "roomWidth":39,"roomHeight":40,"roomDepth":41, 
                                "rt60":42, 
                                "rirData":43 
                              } 
                             
   ## essFilePath =   <room_id> / <config_id> / <spkstep-SPKSTEPNO-micstep-MICSTEPNO-spkno-SPKNO> / receivedEssSignal-MICNO.wav
   
                                 
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

  





  
 def printAllToFile(self):
          all_record_file = open("all_records.txt", "w")

          for dataline in self.rir_data:
             
              CENT=100 ## M / CM 
          
              roomDepth=float(dataline[int(self.rir_data_field_numbers['roomDepth'])])/CENT # CM to M
              roomWidth=float(dataline[int(self.rir_data_field_numbers['roomWidth'])])/CENT # CM to M
              roomHeight=float(dataline[int(self.rir_data_field_numbers['roomHeight'])])/CENT # CM to M
                  
              microphoneCoordinatesX=float(dataline[int(self.rir_data_field_numbers['microphoneStandInitialCoordinateX'])])/CENT +float(dataline[int(self.rir_data_field_numbers['mic_RelativeCoordinateX'])])/CENT # CM to M 
              microphoneCoordinatesY=float(dataline[int(self.rir_data_field_numbers['microphoneStandInitialCoordinateY'])])/CENT +float(dataline[int(self.rir_data_field_numbers['mic_RelativeCoordinateY'])])/CENT # CM to M
              microphoneCoordinatesZ=float(dataline[int(self.rir_data_field_numbers['mic_RelativeCoordinateZ'])])/CENT


              speakerCoordinatesX=float(dataline[int(self.rir_data_field_numbers['speakerStandInitialCoordinateX'])])/CENT +float(dataline[int(self.rir_data_field_numbers['speakerRelativeCoordinateX'])])/CENT # CM to M
              speakerCoordinatesY=float(dataline[int(self.rir_data_field_numbers['speakerStandInitialCoordinateY'])])/CENT +float(dataline[int(self.rir_data_field_numbers['speakerRelativeCoordinateY'])])/CENT # CM to M
              speakerCoordinatesZ=float(dataline[int(self.rir_data_field_numbers['speakerRelativeCoordinateZ'])])/CENT

              rt60=float(dataline[int(self.rir_data_field_numbers['rt60'])])

              if 0.5 < rt60 < 0.6 :
                 rt60=rt60+0.1
              elif 0.6 < rt60:
                 rt60=rt60+0.2
                    
              speakerMotorIterationNo=int(dataline[int(self.rir_data_field_numbers['speakerMotorIterationNo'])])
              microphoneMotorIterationNo=int(dataline[int(self.rir_data_field_numbers['microphoneMotorIterationNo'])])
              currentActiveSpeakerNo=int(dataline[int(self.rir_data_field_numbers['currentActiveSpeakerNo'])])
              currentActiveSpeakerChannelNo=int(dataline[int(self.rir_data_field_numbers['currentActiveSpeakerChannelNo'])])
              physicalSpeakerNo=int(dataline[int(self.rir_data_field_numbers['physicalSpeakerNo'])]) 
              roomId=dataline[int(self.rir_data_field_numbers['roomId'])] 
              configId=dataline[int(self.rir_data_field_numbers['configId'])] 
              micNo=dataline[int(self.rir_data_field_numbers['micNo'])] 
              rirData=dataline[int(self.rir_data_field_numbers['rirData'])]
              lengthOfRirSignal=len(rirData)
              maxOfRirSignal=max(rirData)
              minOfRirSignal=min(rirData)

              all_record_file.write(f"#################################################\n")
              all_record_file.write(f"roomDepth={roomDepth} roomWidth={roomWidth} roomHeight={roomHeight} microphoneCoordinatesX={microphoneCoordinatesX} microphoneCoordinatesY={microphoneCoordinatesY} microphoneCoordinatesZ={microphoneCoordinatesZ} speakerCoordinatesX={speakerCoordinatesX} speakerCoordinatesY={speakerCoordinatesY} speakerCoordinatesZ={speakerCoordinatesZ} rt60={rt60} speakerMotorIterationNo={speakerMotorIterationNo} microphoneMotorIterationNo={microphoneMotorIterationNo} currentActiveSpeakerNo={currentActiveSpeakerNo} currentActiveSpeakerChannelNo={currentActiveSpeakerChannelNo} physicalSpeakerNo={physicalSpeakerNo} roomId={roomId} configId={configId} micNo={micNo} lengthOfRirSignal{lengthOfRirSignal} maxOfRirSignal={maxOfRirSignal} minOfRirSignal={minOfRirSignal}\n")
              
          all_record_file.close()


# def plotWavs(self,real_data,generated_data,MSE,SSIM,title,show=False,saveToPath=None):
#     plt.subplot(1,1,1)
#     minValue=np.min(real_data)
#     minValue2=np.min(generated_data)
#     if minValue2 < minValue:
#        minValue=minValue2
#     plt.text(3300, minValue+0.1, f"MSE={float(MSE):.4f}\nSSIM={float(SSIM):.4f}", style='italic',
#        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
#     plt.plot(real_data,color='r', label='real_data')
#     plt.plot(generated_data,color='b', label='generated_data')
#     plt.title(title)
#     plt.xlabel('Time')
#     plt.ylabel('Amlpitude')
#     plt.legend(loc = "upper right")
#     if show :
#        plt.show()
#     if saveToPath is not None :
#        plt.savefig(saveToPath)
#     plt.close()    
         
         
# def plotMfccSpectrogram(self,title,power_to_db,sr,show=False,saveToPath=None):
#     fig, ax = plt.subplots()
#     img=librosa.display.specshow(power_to_db, sr=sr, x_axis='time',  cmap='magma',  ax=ax)
#     fig.colorbar(img, ax=ax,label='dB')
#     plt.title('MFCC '+title, fontdict=dict(size=18))
#     plt.xlabel('', fontdict=dict(size=15))
#     plt.ylabel('', fontdict=dict(size=15))
#     plt.savefig(saveToPath)
#     plt.close()    
#         
# def getMfccSpectrogram(self,data,title=None,saveToPath=None):
#         sample_rate = 16000 ;  num_mfccs=4096
#         do_also_librosa_for_comparison=False
#         if do_also_librosa_for_comparison:
#           mfcc_librosa = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=num_mfccs)
#           if saveToPath is not None :
#            self.plotSpectrogram(title+"_librosa",mfcc_librosa,sample_rate,saveToPath=saveToPath+".librosa.png")
#         mfcc_transform_fn=torchaudio.transforms.MFCC(sample_rate=sample_rate,n_mfcc=num_mfccs,melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": num_mfccs, "center": False},)
#         mfccs= mfcc_transform_fn( torch.Tensor(data) ).numpy()
#         if saveToPath is not None :
#            self.plotiMfccSpectrogram(title,mfccs,sample_rate,saveToPath=saveToPath)
#         return mfccs
         

def main(dataFilePath):
  rirData=RIRData(dataFilePath)
  rirData.printAllToFile()
  print("SCRIPT IS FINISHED")

if __name__ == '__main__':
 main(str(sys.argv[1]).strip())



         




