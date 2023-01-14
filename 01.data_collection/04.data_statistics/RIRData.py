#!/usr/bin/env python3
import gc
from RIRHeader import *
import scipy.io.wavfile
from scipy import signal
from scipy import stats

import matplotlib.pyplot as plt


from acoustics.utils import _is_1d
from acoustics.signal import bandpass
from acoustics.bands import (_check_band_type, octave_low, octave_high, third_low, third_high)


np.set_printoptions(threshold=sys.maxsize)

class RIRData :
 def __init__(self, logger):
 
   
 
   self.logger  = logger
   self.script_dir=os.path.dirname(os.path.realpath(__file__))
   self.sampling_rate=44100 # 44100 sample points per second
   self.rir_seconds=2
   self.track_length=self.rir_seconds*self.sampling_rate 
   self.final_sound_data_length=int(self.track_length/self.rir_seconds)
   self.number_of_microphones=6
   self.number_of_speakers=4
   self.data_dir=self.script_dir+'/../../../data/single-speaker/'
   self.data_dir_visualize=self.data_dir+"/visualize/"
   self.roomProperties={}
   self.rooms_and_configs={}
   
          
   if not os.path.exists(self.data_dir_visualize) : 
          os.makedirs(self.data_dir_visualize)
          
   self.rir_data_file_path=self.data_dir+"/RIR.pickle.dat"
   self.rir_8k_data_file_path=self.data_dir+"/RIR.pickle.8k.dat"
   self.ess_db_csv_file_path=self.data_dir+"/ess_db.csv"
   self.song_db_csv_file_path=self.data_dir+"/song_db.csv"
   
   self.rir_data=[]  ##  "RIR.dat" --> list of list [34]


   ## 06.data.post.processing/run.sh icinden
   '''
   timestamp speakerMotorIterationNo microphoneMotorIterationNo speakerMotorIterationDirection currentActiveSpeakerNo currentActiveSpeakerChannelNo physicalSpeakerNo microphoneStandInitialCoordinateX microphoneStandInitialCoordinateY 
   microphoneStandInitialCoordinateZ speakerStandInitialCoordinateX speakerStandInitialCoordinateY speakerStandInitialCoordinateZ microphoneMotorPosition speakerMotorPosition temperatureAtMicrohponeStand humidityAtMicrohponeStand 
   temperatureAtMSpeakerStand humidityAtSpeakerStand tempHumTimestamp speakerRelativeCoordinateX speakerRelativeCoordinateY speakerRelativeCoordinateZ microphoneStandAngle speakerStandAngle speakerAngleTheta speakerAnglePhi 
   mic_RelativeCoordinateX mic_RelativeCoordinateY mic_RelativeCoordinateZ mic_DirectionX mic_DirectionY mic_DirectionZ mic_Theta mic_Phi filepath
   '''
                  
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

  
 def visualizeAllTheData(self):
 
     self.logger.info ("loadData function started ...")
     
            
     if  os.path.exists( self.rir_data_file_path) :
         os.remove( self.rir_data_file_path)
     if  os.path.exists( f"{self.data_dir_visualize}/mean.wav") :
         os.remove( f"{self.data_dir_visualize}/mean.wav")

     if  os.path.exists( self.rir_data_file_path) :
         rir_data_file=open(self.rir_data_file_path,'rb')
         self.rir_data=pickle.load(rir_data_file)
         rir_data_file.close()
     else :
         
         if not os.path.exists(self.ess_db_csv_file_path) :
            self.logger.info (self.ess_db_csv_file_path+" file does not exist ...")
            exit(1)
             
         if not os.path.exists(self.song_db_csv_file_path) :
            self.logger.info (self.song_db_csv_file_path+" file does not exist ...")
            exit(1)
            
            
         self.clearSingleQuotes(self.ess_db_csv_file_path)
         self.clearSingleQuotes(self.song_db_csv_file_path)
         
         ess_db_csv_data=np.array(np.loadtxt(open(self.ess_db_csv_file_path, 'r'), delimiter=",",dtype=str))
         song_db_csv_data=np.array(np.loadtxt(open(self.song_db_csv_file_path, 'r'), delimiter=",",dtype=str))
     
         self.logger.info ("ess_db_csv.shape="+str(ess_db_csv_data.shape))
         self.logger.info ("song_db_csv.shape="+str(song_db_csv_data.shape))
     
         self.logger.info ("transmittedEssSignal.wav file location="+self.data_dir+"/transmittedEssSignal.wav")

         transmittedEssWav,rate=librosa.load(self.data_dir+"/transmittedEssSignal.wav",sr=self.sampling_rate, mono=True)
         self.transmittedEssWav=transmittedEssWav
         self.logger.info ("default.transmittedWav.shape="+str(self.transmittedEssWav.shape))
         #self.transmittedWav=librosa.resample(transmittedWav, orig_sr=samp_rate, target_sr=self.sampling_rate)
         #self.logger.info ("self.transmittedWav.resampled.shape="+str(self.transmittedWav.shape))

         
     
         self.logger.info ("Building self.rir_data ...")
         for essCsvRowNumber in range(1,int(ess_db_csv_data.shape[0])) :

            if essCsvRowNumber%100 == 0 :
               self.logger.info (f"essCsvRowNumber={essCsvRowNumber}")
               
               
 
            essDataLine=ess_db_csv_data[essCsvRowNumber].tolist()
  
            

            essFilePath=essDataLine[int(self.rir_data_field_numbers["essFilePath"])]

            essFilePathElements=essFilePath.split("/")
            
            roomId=essFilePathElements[0] ## room-sport01
            essDataLine.append(roomId) 
        
            configId=essFilePathElements[1] ## micx-215.0-micy-510.0-spkx-470.0-spky-500.0-2022.05.09_12.15.55
            essDataLine.append(configId)
        
            #recordingStep=essFilePathElements[2] ## micstep-9-spkstep-9-spkno-3
        
            wavFileName=essFilePathElements[3] ## receivedEssSignal-5.wav
            micNo=wavFileName.split(".")[0].split("-")[1]
        
            essDataLine.append(micNo) ## micNo


            if not roomId  in self.roomProperties :
               if not os.path.exists(self.data_dir+"/"+roomId+"/properties/record.ini") :
                  self.logger.info (self.data_dir+"/"+roomId+"/properties/record.ini  file does not exist ...")
                  exit(1)        
               config = configparser.ConfigParser()
               config.read(self.data_dir+"/"+roomId+"/properties/record.ini")
               ROOM_DIM_WIDTH=float(config['room.dimensions']['room_width'])
               ROOM_DIM_DEPTH=float(config['room.dimensions']['room_depth'])
               ROOM_DIM_HEIGHT=float(config['room.dimensions']['room_height'])
               self.roomProperties[roomId]=[ROOM_DIM_WIDTH,ROOM_DIM_HEIGHT,ROOM_DIM_DEPTH]
            essDataLine=essDataLine+self.roomProperties[roomId]   



            if not os.path.exists(self.data_dir+"/"+essFilePath+".ir.wav"):
               if not os.path.exists(self.data_dir+"/"+essFilePath+".ir.wav.bz2") :
                  self.logger.info (self.data_dir+"/"+essFilePath+" file does not exist (neither *.bz2 file exists) ...")
                  continue                
               subproc=subprocess.Popen(["bunzip2", self.data_dir+"/"+essFilePath+".ir.wav.bz2"])
               subproc.wait()

            wavRIRData,sampling_rate=librosa.load(self.data_dir+"/"+essFilePath+".ir.wav",sr=self.sampling_rate)
            
            t60_impulse=self.t60_impulse(wavRIRData,sampling_rate)
            
            essDataLine.append(t60_impulse)
            
            self.logger.info (self.data_dir+"/"+essFilePath+".ir.wav  t60 = "+str(t60_impulse))
            
            essDataLine.append(wavRIRData)
           
            if t60_impulse < 2 :
               self.rir_data.append(essDataLine)   
                           
         if len(self.rir_data) > 0 : 
            rir_data_file=open(self.rir_data_file_path,'wb')
            pickle.dump(self.rir_data,rir_data_file)
            rir_data_file.close()


     self.rir_data_len=len(self.rir_data)
     self.logger.info (f"len(rir_data.shape) = {self.rir_data_len}")
   
     ## prepare training data numbers

     self.testing_rir_data_indexes=random.sample(range(self.rir_data_len), int(self.rir_data_len/10))    
     self.training_rir_data_indexes=np.arange(self.rir_data_len).tolist()
     for i in self.testing_rir_data_indexes :
         self.training_rir_data_indexes.remove(i)
         
     self.logger.info ("Number of testing data line indexes : "+str(len(self.testing_rir_data_indexes)))
     self.logger.info ("Number of training data line indexes : "+str(len(self.training_rir_data_indexes)))

     for dataline in self.rir_data:
         roomId=dataline[int(self.rir_data_field_numbers['roomId'])]
         configId=dataline[int(self.rir_data_field_numbers['configId'])]
         if roomId not in self.rooms_and_configs:
            self.rooms_and_configs[roomId]=[]
         if configId not in self.rooms_and_configs[roomId]:
            self.rooms_and_configs[roomId].append(configId)


     self.takeMeanOfAllRecordsAndSave()
     self.visualizeData()


     #return  self.testing_rir_data_indexes,self.training_rir_data_indexes,self.rir_data



 def clearSingleQuotes(self,filepath):
     #input file
     fin = open(filepath, "rt")
     #output file to write the result to
     fout = open(filepath+".temp", "wt")
     #for each line in the input file
     for line in fin:
       #read replace the string and write to output file
       fout.write(line.replace('\'', ''))
     #close input and output files
     fin.close()
     fout.close()
     os.remove(filepath)
     os.rename(filepath+".temp",filepath)
     

 def normalize(self,data):
    normalized_data=data
    return normalized_data
    
 def denormalize(self,data):
    denormalized_data = data
    return denormalized_data

        
 def allign_referring_max(self,data):
    #np.set_printoptions(threshold=sys.maxsize)

    self.logger.info ("alligning process is started")
    
    new_data=np.zeros((data.shape))
    
    allign_index = self.sampling_rate

    for i in range(data.shape[0]):
        if i%1000 == 0 :
           self.logger.info(f"allign data : {i} ")
        max_index=np.argmax(abs(data[i]))
        #self.plotWav(data[i],f"Sound Record Number : {i}",max_index,data[i][max_index],show=False,saveToPath=f"/tmp/rir.{i}.png")
        #print(allign_index)
        #print(max_index)
        
        diff=allign_index-max_index
        
        #print (diff)
        if diff > 0 :
            new_data[i][abs(diff):]=data[i][0:int(data.shape[1]-abs(diff))]
        else :
            new_data[i][0:int(data.shape[1]-abs(diff))]=data[i][abs(diff):]
        
        
        max_index=np.argmax(abs(new_data[i]))
        #self.plotWav(new_data[i],f"Sound Record Number : {i}",max_index,new_data[i][max_index],show=False,saveToPath=f"/tmp/rir.alligned.{i}.png")
    
    self.logger.info ("alligning process is finished")
    
    return new_data



 def resample_04k(self,data):
    #np.set_printoptions(threshold=sys.maxsize)

    self.logger.info (f"resampling process is started ({self.sampling_rate} to {self.reduced_sampling_rate})")
    
    new_data=np.ones((data.shape[0],self.track_length))
    
    for i in range(data.shape[0]):
       
        if i%100 == 0 :
           self.logger.info(f"resampling data : {i} ")
           
           
           
           
        new_data[i]= librosa.resample(data[i], orig_sr=self.sampling_rate, target_sr=self.reduced_sampling_rate)
       
   
    self.logger.info ("resampling process is finished")
    
    return new_data



 def plotWav(self,data,title,max_index,max_y,show=False,saveToPath=None):
    
     plt.clf()
     plt.plot(data)
     plt.scatter(max_index,max_y,color='red')
     plt.title(title)
     plt.ylabel("Amplitude")
     plt.xlabel("Time")
     
     if show :
        plt.show()
     if saveToPath is not None :
        plt.savefig(saveToPath)



           
 def writeToFile(self,content,fileName):
      file1 = open(fileName, "w")
      file1.writelines(content)
      file1.close()
  
 def takeMeanOfAllRecordsAndSave(self):
     filename=f"{self.data_dir_visualize}/mean.wav"
     print(f"filename={filename}")
     if not os.path.exists(filename):  
          mean_of_data=np.zeros(len(self.rir_data[0][-1])).tolist()
          for dataline in self.rir_data:
            mean_of_data+=dataline[-1]
          mean_of_data=mean_of_data/len(self.rir_data)
          mean_of_data=mean_of_data* 1/ np.max(mean_of_data)
          scipy.io.wavfile.write(filename, self.sampling_rate,mean_of_data)  
     else :
          ## self.data_dir_visualize directory is not empty , so visualization is already made.
          return
  

 def visualizeData(self):
     for roomId in  self.rooms_and_configs:
        for configId in  self.rooms_and_configs[roomId]:
             self.visualizeHeatMap(roomId,configId)
             self.visualizeWaves(roomId,configId)
             
 def visualizeHeatMap(self,roomId,configId):

          mic_heatmapDataSet={}
          spk_heatmapDataSet={}
          ## RESHAPE DATA
          for dataline in self.rir_data:
              dataRoomId=dataline[int(self.rir_data_field_numbers['roomId'])]
              dataConfigId=dataline[int(self.rir_data_field_numbers['configId'])]
              if dataRoomId != roomId or dataConfigId != configId :
                 continue
              #self.logger.info (f"roomId={roomId} configId={configId}")
              stepId=dataline[int(self.rir_data_field_numbers['speakerMotorIterationNo'])]+"-"+dataline[int(self.rir_data_field_numbers['microphoneMotorIterationNo'])]
              #self.logger.info (f"stepId={stepId}")
              micNo=int(dataline[int(self.rir_data_field_numbers['micNo'])])
              spkNo=int(dataline[int(self.rir_data_field_numbers['physicalSpeakerNo'])])
              key=str(roomId)+"-"+str(configId)

              mic_heatmapDatas=None
              spk_heatmapDatas=None
         
              if key not in mic_heatmapDataSet:
            
                  HEATMAP_RESOLUTION_IN_CM=50    
                  roomWidth=int(dataline[int(self.rir_data_field_numbers['roomWidth'])]/HEATMAP_RESOLUTION_IN_CM)
                  roomHeight=int(dataline[int(self.rir_data_field_numbers['roomHeight'])]/HEATMAP_RESOLUTION_IN_CM)
                  roomDepth=int(dataline[int(self.rir_data_field_numbers['roomDepth'])]/HEATMAP_RESOLUTION_IN_CM)
                  mic_heatmapData=np.zeros((roomDepth,roomWidth))
                  spk_heatmapData=np.zeros((roomDepth,roomWidth))
                  mic_heatmapDatas={}
                  spk_heatmapDatas={}
                  self.logger.info (f"key={key}")
                  mic_heatmapDataSet[key]=mic_heatmapData
                  spk_heatmapDataSet[key]=spk_heatmapData
              else :

                  mic_heatmapData=mic_heatmapDataSet[key]
                  spk_heatmapData=spk_heatmapDataSet[key]
            
              microphoneCoordinatesX=int(float(dataline[int(self.rir_data_field_numbers['microphoneStandInitialCoordinateX'])]))+int(float(dataline[int(self.rir_data_field_numbers['mic_RelativeCoordinateX'])]))
              microphoneCoordinatesY=int(float(dataline[int(self.rir_data_field_numbers['microphoneStandInitialCoordinateY'])]))+int(float(dataline[int(self.rir_data_field_numbers['mic_RelativeCoordinateY'])]))
              microphoneCoordinatesZ=int(float(dataline[int(self.rir_data_field_numbers['mic_RelativeCoordinateZ'])]))
              microphoneDirectionX=int(float(dataline[int(self.rir_data_field_numbers['mic_DirectionX'])]))
              microphoneDirectionY=int(float(dataline[int(self.rir_data_field_numbers['mic_DirectionY'])]))
              microphoneDirectionZ=int(float(dataline[int(self.rir_data_field_numbers['mic_DirectionZ'])]))
              speakerCoordinatesX=int(float(dataline[int(self.rir_data_field_numbers['speakerStandInitialCoordinateX'])]))+int(float(dataline[int(self.rir_data_field_numbers['speakerRelativeCoordinateX'])]))
              speakerCoordinatesY=int(float(dataline[int(self.rir_data_field_numbers['speakerStandInitialCoordinateY'])]))+int(float(dataline[int(self.rir_data_field_numbers['speakerRelativeCoordinateY'])]))
              speakerCoordinatesZ=int(float(dataline[int(self.rir_data_field_numbers['speakerRelativeCoordinateZ'])]))
                        
              mic_heatmapData[int(microphoneCoordinatesX/HEATMAP_RESOLUTION_IN_CM),int(microphoneCoordinatesY/HEATMAP_RESOLUTION_IN_CM)]+=1
              spk_heatmapData[int(speakerCoordinatesX/HEATMAP_RESOLUTION_IN_CM),int(speakerCoordinatesY/HEATMAP_RESOLUTION_IN_CM)]+=1
            

         

  
          self.logger.info (f" PLOT  HEATMAP")                      

          ## PLOT DATA AND HEATMAP
          for roomId_configId in mic_heatmapDataSet:
              self.logger.info (f"roomId_configId={roomId_configId}")
              key=roomId_configId
              ## PLOT HEATMAP

              mic_heatmapData=mic_heatmapDataSet[key]
              spk_heatmapData=spk_heatmapDataSet[key]
              self.heatmap(mic_heatmapData,f"{key}.mic_heatmap.")    
              self.heatmap(spk_heatmapData,f"{key}.spk_heatmap")    
              
              
              
 def visualizeWaves(self,roomId,configId):
          gc.collect()
          figureDataSet={}
          titleDataSet={}

          ## RESHAPE DATA
          for dataline in self.rir_data:
              dataRoomId=dataline[int(self.rir_data_field_numbers['roomId'])]
              dataConfigId=dataline[int(self.rir_data_field_numbers['configId'])]
              if dataRoomId != roomId or dataConfigId != configId :
                 continue
              #self.logger.info (f"roomId={roomId} configId={configId}")
              stepId=dataline[int(self.rir_data_field_numbers['speakerMotorIterationNo'])]+"-"+dataline[int(self.rir_data_field_numbers['microphoneMotorIterationNo'])]
              #self.logger.info (f"stepId={stepId}")
              micNo=int(dataline[int(self.rir_data_field_numbers['micNo'])])
              spkNo=int(dataline[int(self.rir_data_field_numbers['physicalSpeakerNo'])])
              key=str(roomId)+"-"+str(configId)
              figureData=None
              figureDatas=None
              titleDatas=None

         
              if key not in figureDataSet:
             
                  figureDatas={}
                  titleDatas={}

                  self.logger.info (f"key={key}")
                  figureDataSet[key]=figureDatas
                  titleDataSet[key]=titleDatas
                  
              else :
                  figureDatas=figureDataSet[key]
                  titleDatas=titleDataSet[key]
            
              microphoneCoordinatesX=int(float(dataline[int(self.rir_data_field_numbers['microphoneStandInitialCoordinateX'])]))+int(float(dataline[int(self.rir_data_field_numbers['mic_RelativeCoordinateX'])]))
              microphoneCoordinatesY=int(float(dataline[int(self.rir_data_field_numbers['microphoneStandInitialCoordinateY'])]))+int(float(dataline[int(self.rir_data_field_numbers['mic_RelativeCoordinateY'])]))
              microphoneCoordinatesZ=int(float(dataline[int(self.rir_data_field_numbers['mic_RelativeCoordinateZ'])]))
              microphoneDirectionX=int(float(dataline[int(self.rir_data_field_numbers['mic_DirectionX'])]))
              microphoneDirectionY=int(float(dataline[int(self.rir_data_field_numbers['mic_DirectionY'])]))
              microphoneDirectionZ=int(float(dataline[int(self.rir_data_field_numbers['mic_DirectionZ'])]))
              speakerCoordinatesX=int(float(dataline[int(self.rir_data_field_numbers['speakerStandInitialCoordinateX'])]))+int(float(dataline[int(self.rir_data_field_numbers['speakerRelativeCoordinateX'])]))
              speakerCoordinatesY=int(float(dataline[int(self.rir_data_field_numbers['speakerStandInitialCoordinateY'])]))+int(float(dataline[int(self.rir_data_field_numbers['speakerRelativeCoordinateY'])]))
              speakerCoordinatesZ=int(float(dataline[int(self.rir_data_field_numbers['speakerRelativeCoordinateZ'])]))

            
              if (stepId not in titleDatas) or (micNo == 0 and spkNo == 0) :
                 self.logger.info (f"stepId={stepId}")
                 titleDatas[stepId]=f"(Spk-Mic:{stepId}) SPK:{speakerCoordinatesX}.{speakerCoordinatesY}.{speakerCoordinatesZ} - MIC-COORD:{microphoneCoordinatesX}.{microphoneCoordinatesY}.{microphoneCoordinatesZ} - MIC-DIR:{microphoneDirectionX}.{microphoneDirectionY}.{microphoneDirectionZ}"



              if stepId not in figureDatas:
                 figureData=np.zeros(int(self.number_of_speakers*self.number_of_microphones*self.track_length))
                 figureDatas[stepId]=figureData

              else :
                 figureData=figureDatas[stepId]

            
              partNo=int(micNo+spkNo*self.number_of_microphones)   
              figureData[partNo*self.track_length:(partNo+1)*self.track_length]=dataline[-1]
         

  
          self.logger.info (f" PLOT DATA AND HEATMAP")                      
          
          ## PLOT DATA AND HEATMAP
          for roomId_configId in figureDataSet:
              self.logger.info (f"roomId_configId={roomId_configId}")
              key=roomId_configId
              figureDatas=figureDataSet[key]
              titleDatas=titleDataSet[key]
         
              filename=f"{self.data_dir_visualize}/data_plot.{key}.png"
              print(f"filename={filename}")


              X = np.arange(0,int(self.number_of_speakers*self.number_of_microphones*self.track_length))
              WIDTH=1
              #figure, axis =  plt.subplots(int(len(list(figureDatas.keys()))/WIDTH),WIDTH,figsize=(10,650))
              figure, axis =  plt.subplots(int(len(list(figureDatas.keys()))),figsize=(10,200))
              figure.subplots_adjust(hspace=0.4, wspace=0.4)   
              i=0 
              for key2 in figureDatas.keys():
                      if i%3 == 0 :
                         color='r'
                      elif i%3 == 1 :
                         color='g'
                      elif i%3 == 2 :
                         color='b'       
                      print(key2)
                      figureData=figureDatas[key2].tolist()
                      #print(figureData[0:100])
                      #print("B")
          
                      axis[i].plot(X,figureData ,color)
                      axis[i].set_title(titleDatas[key2])
                      i=i+1 


                    
              figure.tight_layout()
              ## PLOT DATA 
              plt.savefig(filename)     
              figure.clf()
              plt.close()    
          
    

    


      
 def heatmap(self,matrix, title):
         #print(np.amax(matrix))
         #min_value=np.min(matrix)
         #max_value=np.max(matrix)
         #plt.imshow(matrix, cmap='RdBu', interpolation='nearest',vmin=min_value, vmax=max_value)
         #plt.show()
         #print(matrix)
         #plt.axis('square')
         min_value=np.min(matrix)
         max_value=np.max(matrix)
         fig, ax = plt.subplots()
         cmap=matplotlib.cm.Blues(np.linspace(0,1,10))
         cmap[0,:]=np.array([108/256, 108/256, 108/256, 1])
         cmap=matplotlib.colors.ListedColormap(cmap)
         c = ax.pcolormesh(matrix, cmap=cmap, vmin=min_value, vmax=max_value)
         ax.invert_xaxis()
         ax.set_title(title)
         ax.set_aspect('equal', 'box')
         #ax.set_xticks(np.arange(matrix.shape[1]))
         #ax.set_yticks(np.arange(matrix.shape[0]))


         #ax.axis([0, matrix.shape[0], 0, matrix.shape[1]])
         fig.colorbar(c, ax=ax)
         #plt.show()
         plt.savefig(f"{self.data_dir_visualize}/{title}.png")
         
 def t60_impulse(self,raw_signal,fs):  # pylint: disable=too-many-locals
    """
    Reverberation time from a WAV impulse response.
    :param file_name: name of the WAV file containing the impulse response.
    :param bands: Octave or third bands as NumPy array.
    :param rt: Reverberation time estimator. It accepts `'t30'`, `'t20'`, `'t10'` and `'edt'`.
    :returns: Reverberation time :math:`T_{60}`
    """
    bands =np.array([62.5 ,125, 250, 500,1000, 2000])

    if np.max(raw_signal)==0 and np.min(raw_signal)==0:
        print('came 1')
        return .5
    
    # fs, raw_signal = wavfile.read(file_name)
    band_type = _check_band_type(bands)

    # if band_type == 'octave':
    low = octave_low(bands[0], bands[-1])
    high = octave_high(bands[0], bands[-1])
    # elif band_type == 'third':
    #     low = third_low(bands[0], bands[-1])
    #     high = third_high(bands[0], bands[-1])

    
    init = -0.0
    end = -60.0
    factor = 1.0
    bands =bands[3:5]
    low = low[3:5]
    high = high[3:5]

    t60 = np.zeros(bands.size)

    for band in range(bands.size):
        # Filtering signal
        filtered_signal = bandpass(raw_signal, low[band], high[band], fs, order=8)
        abs_signal = np.abs(filtered_signal) / np.max(np.abs(filtered_signal))

        # Schroeder integration
        sch = np.cumsum(abs_signal[::-1]**2)[::-1]
        sch_db = 10.0 * np.log10(sch / np.max(sch))
        if math.isnan(sch_db[1]):
            print('came 2')
            return .5
        # print("leng sch_db ",sch_db.size)
        # print("sch_db ",sch_db)
        # Linear regression
        sch_init = sch_db[np.abs(sch_db - init).argmin()]
        sch_end = sch_db[np.abs(sch_db - end).argmin()]
        init_sample = np.where(sch_db == sch_init)[0][0]
        end_sample = np.where(sch_db == sch_end)[0][0]
        x = np.arange(init_sample, end_sample + 1) / fs
        y = sch_db[init_sample:end_sample + 1]
        slope, intercept = stats.linregress(x, y)[0:2]

        # Reverberation time (T30, T20, T10 or EDT)
        db_regress_init = (init - intercept) / slope
        db_regress_end = (end - intercept) / slope
        t60[band] = factor * (db_regress_end - db_regress_init)
    mean_t60 =(t60[1]+t60[0])/2
    # print("meant60 is ", mean_t60)
    if math.isnan(mean_t60):
        print('came 3')
        return .5
    return mean_t60


