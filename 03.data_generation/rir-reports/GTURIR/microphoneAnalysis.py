#!/usr/bin/env python3
import importlib
math        = importlib.import_module("math")
glob        = importlib.import_module("glob")
sys         = importlib.import_module("sys")
os          = importlib.import_module("os")
argparse    = importlib.import_module("argparse")
np          = importlib.import_module("numpy")
wave        = importlib.import_module("wave")
scipy       = importlib.import_module("scipy")
configparser= importlib.import_module("configparser")
subprocess  = importlib.import_module("subprocess")
pickle      = importlib.import_module("pickle")
matplotlib  = importlib.import_module("matplotlib")

import scipy.io.wavfile
from scipy import signal
from scipy import stats
import librosa.display
import librosa




np.set_printoptions(threshold=sys.maxsize)


from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

import torchaudio
import torch
import torch.nn.functional as TF


import scipy.fft
from scipy.spatial import distance

import traceback



gtu_rir_data_pickle_file=str(sys.argv[1]).strip()

list_of_room_ids=[ "room-207",  "room-208", "room-conferrence01", "room-sport01", "room-sport02", "room-z02", "room-z04" ,"room-z06" ,"room-z10", "room-z11" ,"room-z23" ]

def load_rir_data(roomId):
 rir_data=[]
 if os.path.exists(gtu_rir_data_pickle_file) :
         rir_data_file=open(gtu_rir_data_pickle_file+"."+roomId,'rb')
         rir_data=pickle.load(rir_data_file)
         rir_data_file.close()
 else :
         print(f"{gtu_rir_data_pickle_file} not exists")
         exit(1)

 return rir_data


def getGlitchPoints(generated,real):
     INSENSITIVITY=3
     glitchThreshold=np.std(np.abs(real))*INSENSITIVITY
     glitchPoints=[]
     for i in range(len(generated)):
         if  abs(abs(generated[i])-abs(real[i]) )> glitchThreshold :
             glitchPoints.append(i)
     return glitchPoints


rir_data_field_numbers={"timestamp":0,"speakerMotorIterationNo":1,"microphoneMotorIterationNo":2,"speakerMotorIterationDirection":3,"currentActiveSpeakerNo":4,"currentActiveSpeakerChannelNo":5,
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

#micDiff={}

micPairSimilarities={}

for selectedRoomId in list_of_room_ids:
    print(f"loading {selectedRoomId}")
    room_data=load_rir_data(selectedRoomId)

    print(f"Analyzing {selectedRoomId}")

    #print("DENEME START")
    #for key,value in micPairSimilarities.items():
    #   print(f"{key} : MSE: {np.mean(np.array(value['MSE']))} , SSIM: {np.mean(np.array(value['SSIM']))}, GLITCH: {np.mean(np.array(value['GLITCH']))}")
    #
    #print("DENEME END")

    for dataline1 in room_data:
       speakerMotorIterationNo_1=dataline1[rir_data_field_numbers['speakerMotorIterationNo']]
       microphoneMotorIterationNo_1=dataline1[rir_data_field_numbers['microphoneMotorIterationNo']]
       physicalSpeakerNo_1=dataline1[rir_data_field_numbers['physicalSpeakerNo']]
       micNo_1=dataline1[rir_data_field_numbers['micNo']]

       microphoneCoordinatesX_1=float(dataline1[rir_data_field_numbers['microphoneStandInitialCoordinateX']]) +float(dataline1[rir_data_field_numbers['mic_RelativeCoordinateX']])  
       microphoneCoordinatesY_1=float(dataline1[rir_data_field_numbers['microphoneStandInitialCoordinateY']]) +float(dataline1[rir_data_field_numbers['mic_RelativeCoordinateY']])  
       microphoneCoordinatesZ_1=float(dataline1[rir_data_field_numbers['mic_RelativeCoordinateZ']])  

       speakerCoordinatesX_1=float(dataline1[rir_data_field_numbers['speakerStandInitialCoordinateX']]) +float(dataline1[rir_data_field_numbers['speakerRelativeCoordinateX']])  
       speakerCoordinatesY_1=float(dataline1[rir_data_field_numbers['speakerStandInitialCoordinateY']]) +float(dataline1[rir_data_field_numbers['speakerRelativeCoordinateY']])  
       speakerCoordinatesZ_1=float(dataline1[rir_data_field_numbers['speakerRelativeCoordinateZ']])  

       for dataline2 in room_data:
           speakerMotorIterationNo_2=dataline2[rir_data_field_numbers['speakerMotorIterationNo']]
           microphoneMotorIterationNo_2=dataline2[rir_data_field_numbers['microphoneMotorIterationNo']]
           physicalSpeakerNo_2=dataline2[rir_data_field_numbers['physicalSpeakerNo']]
           micNo_2=dataline2[rir_data_field_numbers['micNo']]

           microphoneCoordinatesX_2=float(dataline2[rir_data_field_numbers['microphoneStandInitialCoordinateX']]) +float(dataline2[rir_data_field_numbers['mic_RelativeCoordinateX']])  
           microphoneCoordinatesY_2=float(dataline2[rir_data_field_numbers['microphoneStandInitialCoordinateY']]) +float(dataline2[rir_data_field_numbers['mic_RelativeCoordinateY']])  
           microphoneCoordinatesZ_2=float(dataline2[rir_data_field_numbers['mic_RelativeCoordinateZ']])  

           speakerCoordinatesX_2=float(dataline2[rir_data_field_numbers['speakerStandInitialCoordinateX']]) +float(dataline2[rir_data_field_numbers['speakerRelativeCoordinateX']])  
           speakerCoordinatesY_2=float(dataline2[rir_data_field_numbers['speakerStandInitialCoordinateY']]) +float(dataline2[rir_data_field_numbers['speakerRelativeCoordinateY']])  
           speakerCoordinatesZ_2=float(dataline2[rir_data_field_numbers['speakerRelativeCoordinateZ']])  

           DELTA=10
           if micNo_1!=micNo_2:
              if  ( 
                      abs(microphoneCoordinatesX_1-microphoneCoordinatesX_2)<DELTA and  ## DELTA in  CM
                      abs(microphoneCoordinatesY_1-microphoneCoordinatesY_2)<DELTA and
                      abs(microphoneCoordinatesZ_1-microphoneCoordinatesZ_2)<DELTA and
                      abs(speakerCoordinatesX_1-speakerCoordinatesX_2)<DELTA and
                      abs(speakerCoordinatesY_1-speakerCoordinatesY_2)<DELTA and
                      abs(speakerCoordinatesZ_1-speakerCoordinatesZ_2)<DELTA 
                  ):
                 # print (f"(spkItrNo-micItrNo-SpkNo-micNo)  {speakerMotorIterationNo_1}-{microphoneMotorIterationNo_1}-{physicalSpeakerNo_1}-{micNo_1}  and {speakerMotorIterationNo_2}-{microphoneMotorIterationNo_2}-{physicalSpeakerNo_2}-{micNo_2} are close : ")
                 # print (f"mic_XYZ_Diff={abs(microphoneCoordinatesX_1-microphoneCoordinatesX_2)}-{abs(microphoneCoordinatesY_1-microphoneCoordinatesY_2)}-{abs(microphoneCoordinatesZ_1-microphoneCoordinatesZ_2)}  spk_XYZ_Diff={abs(speakerCoordinatesX_1-speakerCoordinatesX_2)}-{abs(speakerCoordinatesY_1-speakerCoordinatesY_2)}-{abs(speakerCoordinatesZ_1-speakerCoordinatesZ_2)}")

                  mD ={}
                  #micDiff[f"{selectedRoomId}-{speakerMotorIterationNo_1}-{microphoneMotorIterationNo_1}-{physicalSpeakerNo_1}-{micNo_1}-{speakerMotorIterationNo_2}-{microphoneMotorIterationNo_2}-{physicalSpeakerNo_2}-{micNo_2}"]={}
                  #mD = micDiff[f"{selectedRoomId}-{speakerMotorIterationNo_1}-{microphoneMotorIterationNo_1}-{physicalSpeakerNo_1}-{micNo_1}-{speakerMotorIterationNo_2}-{microphoneMotorIterationNo_2}-{physicalSpeakerNo_2}-{micNo_2}"]
                  
                  rir_data_1=dataline1[rir_data_field_numbers['rirData']]
                  rir_data_2=dataline2[rir_data_field_numbers['rirData']]

                  #generated_data,real_data=self.allignHorizontally(generated_data,real_data)         

                  #generated_data,real_data=self.allignVertically(generated_data,real_data)         
                  ######### BEGIN : YATAY ESITLEME (dikey esitleme zaten maksimum noktalarini esitliyerek yapilmisti)
                  #generated_data=generated_data-np.sum(generated_data)/generated_data.shape[0]
                  ## bu sekilde ortalamasi 0'a denk gelecek
                  ######### END: YATAY ESITLEME (dikey esitleme zaten maksimum noktalarini esitliyerek yapilmisti)

                  rt60_1=dataline1[rir_data_field_numbers['rt60']]
                  rt60_2=dataline2[rir_data_field_numbers['rt60']]

                  mD["rt60"]=abs(rt60_1-rt60_2)
                  mD["mse"]=np.square(np.subtract(rir_data_1,rir_data_2)).mean()

                  rir_data_1_tiled=np.tile(rir_data_1, (2, 1)) ## duplicate 1d data to 2d
                  rir_data_2_tiled=np.tile(rir_data_2, (2, 1)) ## duplicate 1d data to 2d
                  rir_data_1_tiled=np.reshape(rir_data_1_tiled,(1,1,rir_data_1_tiled.shape[0],rir_data_1_tiled.shape[1]))
                  rir_data_2_tiled=np.reshape(rir_data_2_tiled,(1,1,rir_data_2_tiled.shape[0],rir_data_2_tiled.shape[1]))
                  rir_data_1_tensor=torch.from_numpy(rir_data_1_tiled)
                  rir_data_2_tensor=torch.from_numpy(rir_data_2_tiled)
                  mD["ssim"]=ssim(rir_data_1_tensor,rir_data_2_tensor,data_range=4.0,size_average=True).item()
                  mD["glitch_point_count"]=len(getGlitchPoints(rir_data_1,rir_data_2))
                  if micNo_1 < micNo_2:
                     mD["mics"]=f"{micNo_1}-{micNo_2}"
                  else:
                     mD["mics"]=f"{micNo_2}-{micNo_1}"
 
                  if mD["mics"] not in micPairSimilarities:
                     micPairSimilarities[mD["mics"]]={}
                     micPairSimilarities[mD["mics"]]["MSE"]=[]
                     micPairSimilarities[mD["mics"]]["SSIM"]=[]
                     micPairSimilarities[mD["mics"]]["GLITCH"]=[]
              
                  micPairSimilarities[mD["mics"]]["MSE"].append(mD["mse"])
                  micPairSimilarities[mD["mics"]]["SSIM"].append(mD["ssim"])
                  micPairSimilarities[mD["mics"]]["GLITCH"].append(mD["glitch_point_count"])

for key,value in micPairSimilarities.items():
       print(f"{key} : MSE: {np.mean(np.array(value['MSE']))} , SSIM: {np.mean(np.array(value['SSIM']))}, GLITCH: {np.mean(np.array(value['GLITCH']))}")


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

                  
         t2=time.time() 
         print(f"DeltaT.allignHorizontally={t2-t1}")
         print("1.MAX ALLIGNMENT : np.argmax(real_data[0:1000]):"+str(self.getLocalArgMax(1000,real_data)))
         print("1.MAX ALLIGNMENT : np.argmax(generated_data[0:1000]):"+str(self.getLocalArgMax(1000,generated_data)))
         #real_data1=librosa.resample(self.rir_data[i][-1], orig_sr=44100, target_sr=sr)
         #real_data1=real_data1[:generated_data.shape[0]]
         #print("1.MAX ALLIGNMENT : np.argmax(real_data1[0:1000]):"+str(self.getLocalArgMax(1000,real_data1)))
         # test edildi problem yok :)
         return generated_data,real_data
         

         
         
